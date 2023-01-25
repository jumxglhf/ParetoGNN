import dgl
import torch
import src.utils as utils
import torch.nn.functional as F
import math
import copy
from torch import Tensor 
import numpy as np
import torch.nn as nn

class Link_Pred(torch.nn.Module):
    def __init__(self, in_channels):
        super(Link_Pred, self).__init__()
        self.linear = torch.nn.Linear(in_channels, 1)
        self.linear_ =  torch.nn.Linear(in_channels, in_channels)

    def forward(self, h):
        return self.linear(h).squeeze()


class PretrainModule(torch.nn.Module):
    def __init__(self, big_model, predictor_dim):
        super(PretrainModule, self).__init__()
        hid_dim = big_model.hid_dim
        self.big_model = big_model
        
        # link prediction head
        self.link_predictor_hid = torch.nn.Linear(hid_dim, predictor_dim)
        self.link_predictor_class = torch.nn.Linear(predictor_dim, 1)
        
        # graph matching head
        self.graph_matcher = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(predictor_dim, hid_dim))
        # discriminator for ming
        self.discriminator = Discriminator(hid_dim)

        # head for metis partition cls
        self.metis_cls = torch.nn.Linear(hid_dim, 10)

        # head for metis partition clsss
        self.par_cls = torch.nn.Linear(hid_dim, 20)

        # head for minsg
        self.minsg = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(predictor_dim, hid_dim))

        # head for decor
        self.decor = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(predictor_dim, hid_dim))

        # head for feature reconstruction
        self.recon_mask = torch.nn.Parameter(torch.zeros(1, big_model.node_module.in_feats))
        self.recon_enc_dec = torch.nn.Linear(hid_dim, hid_dim, bias=False)
        self.decoder = dgl.nn.GraphConv(hid_dim, big_model.node_module.in_feats, allow_zero_in_degree=True)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):

        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def compute_representation(self, g, X):
        self.train(False)
        with torch.no_grad():
            h = self.big_model(g, X)
        self.train(True)
        return h.detach()

    def forward(self, sample, opt):
        res = {}
        if 'p_link' in sample:
            data = sample['p_link']
            res['p_link'] = self.p_link(data[0], data[1], data[2], data[3], data[4])
        
        if 'p_ming' in sample:
            data = sample['p_ming']
            res['p_ming'] = self.p_ming(data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device))

        if 'p_minsg' in sample:
            data = sample['p_minsg']
            res['p_minsg'] = self.p_minsg(data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device), data[3].to(opt.device), temperature=opt.temperature_minsg)
        
        if 'p_decor' in sample:
            data = sample['p_decor']
            res['p_decor'] = self.p_decor(data[0].to(opt.device), data[1].to(opt.device), lambd=opt.decor_lamb)

        if 'p_recon' in sample:
            data = sample['p_recon']
            res['p_recon'] = self.p_recon(data[0].to(opt.device), data[1])

        return res 


    def p_link(self, sg, pos_u, pos_v, neg_u, neg_v):
        h = self.big_model(sg)
        h = F.normalize(h, dim=1)
        h = F.relu(self.link_predictor_hid(h))
        h_pos = h[pos_u] * h[pos_v]
        h_neg = h[neg_u] * h[neg_v]
        pos_logits = self.link_predictor_class(h_pos).squeeze()
        neg_logits = self.link_predictor_class(h_neg).squeeze()
        logits = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)])
        target = torch.cat([torch.ones_like(pos_logits),torch.zeros_like(neg_logits)])
        return F.binary_cross_entropy(logits, target)
    
    def p_ming(self, bg, feat, cor_feat):
        positive = self.big_model(bg, feat)
        negative = self.big_model(bg, cor_feat)

        summary = torch.sigmoid(positive.mean(dim=0)) 
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        l1 = F.binary_cross_entropy(torch.sigmoid(positive), torch.ones_like(positive))
        l2 = F.binary_cross_entropy(torch.sigmoid(negative), torch.zeros_like(negative))
        return l1 + l2

    def p_minsg(self, g1, feat1, g2, feat2, temperature):
        
        def get_loss(h1, h2, temperature):
            f = lambda x: torch.exp(x / temperature)
            refl_sim = f(utils.sim_matrix(h1, h1))        # intra-view pairs
            between_sim = f(utils.sim_matrix(h1, h2))     # inter-view pairs
            x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
            loss = -torch.log(between_sim.diag() / x1)
            return loss

        h1 = self.minsg(self.big_model(g1, feat1))
        h2 = self.minsg(self.big_model(g2, feat2))
        l1 = get_loss(h1, h2, temperature)
        l2 = get_loss(h2, h1, temperature)
        ret = (l1 + l2) * 0.5
        return ret.mean()  # utils.constrastive_loss(h1, h2, temperature=temperature)

    def p_decor(self, g1, g2, lambd=1e-3):
        N = g1.number_of_nodes()
        h1 = self.big_model(g1, g1.ndata['feat'])
        h2 = self.big_model(g2, g2.ndata['feat'])

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        c = (z1 - z2) / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = torch.linalg.matrix_norm(c)
        iden = torch.tensor(np.eye(c1.shape[0])).to(h1.device)
        loss_dec1 = torch.linalg.matrix_norm(iden - c1)
        loss_dec2 = torch.linalg.matrix_norm(iden - c2)

        return loss_inv + lambd * (loss_dec1 + loss_dec2)

    def p_recon(self, g, mask_nodes):
        x_target = g.ndata['feat'][mask_nodes].clone()
        feat = g.ndata['feat'].clone()
        feat[mask_nodes] = 0
        feat[mask_nodes] += self.recon_mask
        h = self.big_model(g, feat)
        h = self.recon_enc_dec(h)
        h[mask_nodes] = 0
        x_pred = self.decoder(g, h)[mask_nodes]
        return sce_loss(x_pred, x_target)

class BigModel(torch.nn.Module):
    def __init__(self, node_module, graph_module, hid_dim):
        super(BigModel, self).__init__()

        self.node_module = node_module
        self.graph_module = graph_module
        self.hid_dim = node_module.n_classes
        self.inter_mid = hid_dim
        # this is a universal projection head, agnostic of downstream task
        if hid_dim > 0:
            if graph_module != None:
                self.projection = torch.nn.Linear(node_module.n_classes + graph_module.hid_dim , hid_dim)
            else:
                self.projection = torch.nn.Sequential(torch.nn.Linear(node_module.n_classes, hid_dim),
                                                    torch.nn.PReLU(),
                                                    torch.nn.Linear(hid_dim, node_module.n_classes),
                                                    torch.nn.PReLU()
                )

        for m in self.modules():
            self.weights_init(m)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, G, X=None):
        if type(G) is list:
            node = self.node_module(G)
            if self.graph_module == None:
                if self.inter_mid > 0:
                    return self.projection(node)
                else:
                    return node
            graph = self.graph_module(G, X)
            h = torch.cat([node, graph], dim= -1)
            if self.inter_mid > 0:
                return self.projection(node)
            else:
                return node
        else:
            node = self.node_module(G, X)
            if self.graph_module == None:
                if self.inter_mid > 0:
                    return self.projection(node)
                else:
                    return node
            graph = self.graph_module(G, X)
            h = torch.cat([node, graph], dim= -1)
            if self.inter_mid > 0:
                return self.projection(node)
            else:
                return node


class GCN(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_lst,
                 dropout,
                 norm,
                 prelu):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.in_feats = in_feats
        hidden_lst = [in_feats] + hidden_lst
        for in_, out_ in zip(hidden_lst[:-1], hidden_lst[1:]):
            self.layers.append(dgl.nn.GraphConv(in_, out_, allow_zero_in_degree=True))
            self.norms.append(torch.nn.BatchNorm1d(out_, momentum=0.99) if norm == 'batch' else \
                              torch.nn.LayerNorm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = hidden_lst[-1]

    def forward(self, g, features=None):
        if type(g) is list:
            h = g[0].ndata['feat']['_N'].to(self.layers[-1].weight.device)
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g[i].to(self.layers[-1].weight.device), h)
                h = self.activations[i](self.norms[i](h))
        else:
            h = features
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g, h)
                h = self.activations[i](self.norms[i](h))
        return h



class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features

class BatchNorm(torch.nn.Module):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.module = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                                           track_running_stats)

    def reset_parameters(self):
        self.module.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        """"""
        return self.module(x)


    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.num_features})'

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss
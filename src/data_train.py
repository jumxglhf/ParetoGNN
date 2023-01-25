import torch
import numpy as np
import dgl
import copy


class Graph_Dataset(torch.utils.data.Dataset):
    def __init__(self, g):
        self.len = g.number_of_edges()

    def __len__(self):
        return 10000000

    def __getitem__(self, idx):
        return self.len

class Universal_Collator(object):
    def __init__(self, g, use_saint, batch_size, device, tasks,
                minsg_edge_drop_ratio, minsg_feat_drop_ratio, minsg_batch_size_multiplier, minsg_k_hop, \
                ming_k_hop, ming_batch_size_multiplier, \
                gm_sub_graph_size, gm_edge_drop_ratio, gm_node_drop_ratio, gm_feat_drop_ratio, \
                link_negative_ratio, \
                decor_size, der, dfr, \
                recon_size, recon_mask_rate
                ):

        self.g = g
        self.use_saint = use_saint
        self.batch_size = batch_size
        self.device = device
        self.tasks = tasks

        if 'p_link' in tasks:
            # link
            sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
            self.link_sampler = dgl.dataloading.as_edge_prediction_sampler(
                            sampler, negative_sampler=dgl.dataloading.negative_sampler.GlobalUniform(link_negative_ratio))

        if 'p_minsg' in tasks:
            # minsg
            self.minsg_k_hop = minsg_k_hop
            self.minsg_use_sample = None
            self.minsg_batch_size_multiplier = minsg_batch_size_multiplier
            self.minsg_edge_drop_ratio = minsg_edge_drop_ratio
            self.minsg_feat_drop_ratio = minsg_feat_drop_ratio
            if use_saint and self.minsg_batch_size_multiplier != 0 :
                self.minsg_sampler = dgl.dataloading.SAINTSampler('node', budget=self.minsg_batch_size_multiplier*self.batch_size)
            self.minsg_augmentations = [dgl.transforms.DropEdge(minsg_edge_drop_ratio), \
                                    dgl.transforms.FeatMask(minsg_feat_drop_ratio, ['feat'])]
        
        if 'p_ming' in tasks:
            # ming
            self.ming_k_hop = ming_k_hop
            self.ming_use_sample = None
            self.ming_batch_size_multiplier = ming_batch_size_multiplier
            if use_saint:
                self.ming_sampler = dgl.dataloading.SAINTSampler('node', budget=ming_batch_size_multiplier*batch_size)
        
        if 'p_gm' in tasks:
            # gm
            self.gm_sampler = dgl.dataloading.SAINTSampler('node', budget=gm_sub_graph_size) 
            self.gm_augmentations = [dgl.transforms.DropEdge(gm_edge_drop_ratio), \
                                dgl.transforms.DropNode(gm_node_drop_ratio), \
                            dgl.transforms.FeatMask(gm_feat_drop_ratio, ['feat'])]
        
        if 'p_decor' in tasks:
            self.decor_size = decor_size
            self.der = der
            self.dfr = dfr
            if decor_size > 0:
                self.decor_sampler = dgl.dataloading.SAINTSampler('node', budget=decor_size) 
            self.decor_augmentations = [dgl.transforms.DropEdge(self.der), \
                                    dgl.transforms.FeatMask(self.dfr, ['feat'])]

        if 'p_recon' in tasks:
            self.recon_size = recon_size
            self.mask_rate = recon_mask_rate
            if recon_size > 0:
                self.recon_sampler = dgl.dataloading.SAINTSampler('node', budget=recon_size) 
            self.recon_augmentations = dgl.transforms.DropEdge(0.35)

    def __call__(self, indices):
        outputs = {}

        if 'p_link' in self.tasks:
            # link
            indices = torch.randperm(indices[0])[:len(indices)*10]
            outputs['p_link'] = link_prediction_data_process(self.link_sampler.sample(self.g, indices), self.device)
            
        if 'p_minsg' in self.tasks:
            # minsg
            if self.use_saint and self.minsg_batch_size_multiplier != 0:
                graphs_v1 = self.minsg_sampler.sample(self.g, 0)
            else:
                if self.minsg_use_sample == None:
                    if self.g.number_of_nodes() > self.batch_size*self.minsg_batch_size_multiplier and self.minsg_batch_size_multiplier != 0:
                        self.minsg_use_sample = True
                    else:
                        self.minsg_use_sample = False
                if self.minsg_use_sample:
                    node_idx = np.random.choice(self.g.number_of_nodes(), self.batch_size, replace=False)
                    graphs_v1 = dgl.khop_in_subgraph(self.g, node_idx, k=self.minsg_k_hop)[0]
                else:
                    graphs_v1 = dgl_graph_copy(self.g) # copy.deepcopy(self.g)
            g1, g2 = dgl_graph_copy(graphs_v1), dgl_graph_copy(graphs_v1) # copy.deepcopy(graphs_v1), copy.deepcopy(graphs_v1)
            for aug in self.minsg_augmentations: aug(g1)
            for aug in self.minsg_augmentations: aug(g2)
            outputs['p_minsg'] = [g1, g1.ndata['feat'], g2, g2.ndata['feat']]

        if 'p_ming' in self.tasks:
            # ming
            if self.use_saint and self.ming_batch_size_multiplier != 0:
                g = self.ming_sampler.sample(self.g, 0)
                X = g.ndata['feat']
                perm = torch.randperm(X.shape[0])
                outputs['p_ming'] = [g, X, X[perm]]
            else:
                if self.ming_use_sample == None:
                    if self.g.number_of_nodes() > self.batch_size*self.ming_batch_size_multiplier and self.ming_batch_size_multiplier != 0:
                        self.ming_use_sample = True
                    else:
                        self.ming_use_sample = False
                if self.ming_use_sample:
                    node_idx = np.random.choice(self.g.number_of_nodes(), self.batch_size, replace=False)
                    g = dgl.khop_in_subgraph(self.g, node_idx, k=self.ming_k_hop)[0]

                    X = g.ndata['feat']
                    perm = torch.randperm(X.shape[0])
                    outputs['p_ming'] = [g, X, X[perm]]
                else:
                    X = self.g.ndata['feat']
                    perm = torch.randperm(X.shape[0])
                    outputs['p_ming'] = [self.g, X, X[perm]]

        if 'p_gm' in self.tasks:
            # gm
            graphs_v1 = [self.gm_sampler.sample(self.g, 0) for _ in range(self.batch_size)]
            aug_type = np.random.choice(3, self.batch_size, replace=True)
            aug_type = {k:aug_type[k] for k in range(self.batch_size)}
            graphs_v2 = [dgl_graph_copy(g) for g in graphs_v1]
            for i, g in enumerate(graphs_v1):
                self.gm_augmentations[aug_type[i]](g)
            for i, g in enumerate(graphs_v2):
                self.gm_augmentations[aug_type[i]](g) 
            bg1, bg2 = dgl.batch(graphs_v1), dgl.batch(graphs_v2)
            outputs['p_gm'] = [bg1, bg2]
        
        if 'p_decor' in self.tasks:
            if self.decor_size > 0 and self.g.number_of_nodes() > self.decor_size: 
                g_v1 = self.decor_sampler.sample(self.g, 0)
            else:
                g_v1 = dgl_graph_copy(self.g)

            g_v2 = dgl_graph_copy(g_v1)
            for aug in self.decor_augmentations: aug(g_v1)
            for aug in self.decor_augmentations: aug(g_v2)
            outputs['p_decor'] = [g_v1, g_v2]

        if 'p_recon' in self.tasks:
            if self.recon_size > 0 and self.g.number_of_nodes() > self.recon_size: 
                g = self.recon_sampler.sample(self.g, 0)
            else:
                g = dgl_graph_copy(self.g)
            self.recon_augmentations(g)
            num_nodes = g.num_nodes()
            perm = torch.randperm(num_nodes)
            num_mask_nodes = int(self.mask_rate * num_nodes)
            mask_nodes = perm[: num_mask_nodes]
            outputs['p_recon'] = [g, mask_nodes]
            
        return outputs


# Link Prediction
def get_link_prediction_dataloader(g, batch_size, n_workers, device):
    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, exclude='self',
        negative_sampler=dgl.dataloading.negative_sampler.GlobalUniform(2))
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(g.number_of_edges()), sampler, device=device,
        batch_size=batch_size*20, shuffle=True, drop_last=False, num_workers=0)
    return dataloader

def link_prediction_data_process(outputs_from_dataloader, device, mask_ratio=0.4):
    input_nodes, pos_pair_graph, neg_pair_graph, sg = outputs_from_dataloader
    pos_edges = pos_pair_graph.edges()
    neg_edges = neg_pair_graph.edges()
    pos_u, pos_v = pos_edges[0], pos_edges[1]
    neg_u, neg_v = neg_edges[0], neg_edges[1]
    sg[0].ndata['feat']['_N'] = torch.nn.functional.dropout(sg[0].ndata['feat']['_N'], mask_ratio)
    return sg, pos_u, pos_v, neg_u, neg_v


# minsg
def get_minsg_dataloader(g, batch_size, batch_size_multiplier, buffer_size, use_saint, n_workers, khop):
    dataset = minsg_Dataset(g, buffer_size)
    collator = minsg_Collator(batch_size, batch_size_multiplier, use_saint, 0.3, 0.3, khop)
    dataloader = dgl.dataloading.GraphDataLoader(
            dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0, collate_fn=collator, pin_memory=False)
    return dataloader

class minsg_Dataset(torch.utils.data.Dataset):
    def __init__(self, g, buffer_size=100):
        self.buffer_size = buffer_size
        self.g = g
            
    def __len__(self):
        return self.buffer_size #len(self.sub_graphs)

    def __getitem__(self, idx):
        return self.g

class minsg_Collator(object):
    def __init__(self, batch_size, batch_size_multiplier, use_saint,
                 edge_drop_ratio,
                 feat_drop_ratio,
                 k_hop,
                 ):
        self.augmentations = [dgl.transforms.DropEdge(edge_drop_ratio), \
                            dgl.transforms.FeatMask(feat_drop_ratio, ['feat'])]
        self.k_hop = k_hop
        self.use_saint = use_saint
        self.use_sample = None
        self.batch_size = batch_size*batch_size_multiplier

        if use_saint and self.batch_size != 0 :
            self.sampler = dgl.dataloading.SAINTSampler('node', budget=self.batch_size)

    def __call__(self, graphs_v1):
        assert len(graphs_v1) == 1, f'minsg ONLY TAKES ONE SUBGRAPH, But get {len(graphs_v1)} graphs'
        graphs_v1 = graphs_v1[0]
        if self.use_saint and self.batch_size != 0:
            graphs_v1 = self.sampler.sample(graphs_v1, 0)
        else:
            if self.use_sample == None:
                if graphs_v1.number_of_nodes() > self.batch_size and self.batch_size != 0:
                    self.use_sample = True
                else:
                    self.use_sample = False
            if self.use_sample:
                node_idx = np.random.choice(graphs_v1.number_of_nodes(), self.batch_size, replace=False)
                graphs_v1 = dgl.khop_in_subgraph(graphs_v1, node_idx, k=self.k_hop)[0]
        g1, g2 = copy.deepcopy(graphs_v1), copy.deepcopy(graphs_v1)

        for aug in self.augmentations: aug(g1)
        for aug in self.augmentations: aug(g2)
        return g1, g1.ndata['feat'], g2, g2.ndata['feat']


# ming
def get_ming_dataloader(g, batch_size, batch_size_multiplier, buffer_size, use_saint, n_workers, khop):
    dataset = ming_Dataset(g, buffer_size)
    collator = ming_Collator(batch_size, batch_size_multiplier, use_saint, khop)
    dataloader = dgl.dataloading.GraphDataLoader(
            dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0, collate_fn=collator, pin_memory=False)

    return dataloader

class ming_Dataset(torch.utils.data.Dataset):
    def __init__(self, g, buffer_size=100):
        self.buffer_size = buffer_size
        self.g = g
            
    def __len__(self):
        return self.buffer_size #len(self.sub_graphs)

    def __getitem__(self, idx):
        return self.g

class ming_Collator(object):
    def __init__(self, batch_size, batch_size_multiplier, use_saint, k_hop=3):
        self.k_hop = k_hop
        self.use_saint = use_saint
        self.use_sample = None
        self.batch_size = batch_size*batch_size_multiplier

        if use_saint:
            self.sampler = dgl.dataloading.SAINTSampler('node', budget=batch_size_multiplier*batch_size)

    def __call__(self, g):
        assert len(g) == 1, f'ming ONLY TAKES ONE SUBGRAPH, But get {len(g)} graphs'
        g = copy.deepcopy(g[0])
        if self.use_saint and self.batch_size != 0:
            g = self.sampler.sample(g, 0)
        else:
            if self.use_sample == None:
                if g.number_of_nodes() > self.batch_size and self.batch_size != 0:
                    self.use_sample = True
                else:
                    self.use_sample = False
            if self.use_sample:
                node_idx = np.random.choice(g.number_of_nodes(), self.batch_size, replace=False)
                g = dgl.khop_in_subgraph(g, node_idx, k=self.k_hop)[0]

        X = g.ndata['feat']
        perm = torch.randperm(X.shape[0])
        return g, X, X[perm]


# GM
def get_gm_dataloader(g, batch_size, sub_graph_size, buffer_size, n_workers):
    dataset = GM_Dataset(g, sub_graph_size, buffer_size)
    dataloader = dgl.dataloading.GraphDataLoader(
            dataset, pin_memory=False, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers, collate_fn=GM_Collator(0.1, 0.1, 0.1, sub_graph_size))
    return dataloader

class GM_Dataset(torch.utils.data.Dataset):
    def __init__(self, g, sub_graph_size=256, buffer_size=50000):
        self.sampler = dgl.dataloading.SAINTSampler('node', budget=sub_graph_size) # 
        self.graph = g
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return self.graph 

class GM_Collator(object):
    def __init__(self,
                 edge_drop_ratio,
                 node_drop_ratio,
                 feat_drop_ratio,
                 sub_graph_size
                 ):
        self.augmentations = [dgl.transforms.DropEdge(edge_drop_ratio), \
                            dgl.transforms.DropNode(node_drop_ratio), \
                            dgl.transforms.FeatMask(feat_drop_ratio, ['feat'])]
        self.sampler =  dgl.dataloading.SAINTSampler('node', budget=sub_graph_size)

    def __call__(self, graphs_v1):
        batch_size = len(graphs_v1)
        graphs_v1 = [self.sampler.sample(g, 0) for g in graphs_v1]
        aug_type = np.random.choice(3, batch_size, replace=True)
        aug_type = {k:aug_type[k] for k in range(batch_size)}
        graphs_v2 = [copy.deepcopy(g) for g in graphs_v1]
        for i, g in enumerate(graphs_v1):
            self.augmentations[aug_type[i]](g)
        for i, g in enumerate(graphs_v2):
            self.augmentations[aug_type[i]](g) 
        bg1, bg2 = dgl.batch(graphs_v1), dgl.batch(graphs_v2)
        return bg1, bg2

def dgl_graph_copy(g):
    edges = g.edges()
    new_g = dgl.graph(edges)
    new_g.ndata['feat'] = g.ndata['feat']
    return new_g
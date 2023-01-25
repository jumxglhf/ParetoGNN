from pickle import HIGHEST_PROTOCOL
import torch
import numpy as np
import torch
import dgl
import pickle
from sklearn.preprocessing import StandardScaler

def preprocess(graph, no_self_loop=False):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat
    if not no_self_loop:
        graph = graph.remove_self_loop().add_self_loop()
    else:
        graph = graph.remove_self_loop()
    graph.create_formats_()
    return graph

def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def cross_validation_gen(y, k_fold=10):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k_fold)
    train_splits = []
    val_splits = []
    test_splits = []

    for larger_group, smaller_group in skf.split(y, y):
        train_y = y[smaller_group]
        sub_skf = StratifiedKFold(n_splits=k_fold)
        train_split, val_split = next(iter(sub_skf.split(train_y, train_y)))
        train = torch.zeros_like(y, dtype=torch.bool)
        train[smaller_group[train_split]] = True
        val = torch.zeros_like(y, dtype=torch.bool)
        val[smaller_group[val_split]] = True
        test = torch.zeros_like(y, dtype=torch.bool)
        test[larger_group] = True
        train_splits.append(train.unsqueeze(1))
        val_splits.append(val.unsqueeze(1))
        test_splits.append(test.unsqueeze(1))
    
    return torch.cat(train_splits, dim=1), torch.cat(val_splits, dim=1), torch.cat(test_splits, dim=1)

def load_data(data_name, pretrain_label_dir, mask_edge, tvt_addr, split='random', hetero_graph_path = None, no_self_loop=False):
    if data_name == 'wiki_cs':
        dataset = dgl.data.WikiCSDataset()
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        if split == 'random':
            train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
            g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask
        else:
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = \
                g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'].unsqueeze(1).expand_as(g.ndata['val_mask'])

    elif data_name == 'co_cs':
        dataset = dgl.data.CoauthorCSDataset()
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        # no public split is given
        train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask

    elif data_name == 'co_phy':
        dataset = dgl.data.CoauthorPhysicsDataset()
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        # no public split is given
        train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask

    elif data_name == 'co_photo':
        dataset = dgl.data.AmazonCoBuyPhotoDataset()
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        # no public split is given
        train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask

    elif data_name == 'co_computer':
        dataset = dgl.data.AmazonCoBuyComputerDataset()
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        # no public split is given
        train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = train_mask, val_mask, test_mask

    elif data_name == 'cora':
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        if split == 'random':
            train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = train_mask, val_mask, test_mask
        else:
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = \
                g.ndata['train_mask'].unsqueeze(1), g.ndata['val_mask'].unsqueeze(1),  g.ndata['test_mask'].unsqueeze(1)

    elif data_name == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        if split == 'random':
            train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = train_mask, val_mask, test_mask
        else:
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = \
                g.ndata['train_mask'].unsqueeze(1), g.ndata['val_mask'].unsqueeze(1),  g.ndata['test_mask'].unsqueeze(1)

    elif data_name == 'pubmed':
        dataset = dgl.data.PubmedGraphDataset()
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / std
        if split == 'random':
            train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = train_mask, val_mask, test_mask

    elif data_name == 'actor':
        dataset, _ = dgl.load_graphs(hetero_graph_path + '/actor.bin')
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        if split == 'random':
            train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = train_mask, val_mask, test_mask
    
    elif data_name == 'chameleon':
        dataset, _ = dgl.load_graphs(hetero_graph_path + '/chameleon.bin')
        g = dataset[0]
        # std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        # g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        if split == 'random':
            train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = train_mask, val_mask, test_mask

    elif data_name == 'squirrel':
        dataset, _ = dgl.load_graphs(hetero_graph_path + '/squirrel.bin')
        g = dataset[0]
        std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
        g.ndata['feat'] = (g.ndata['feat'] - mean) / (std + 1e-8)
        if split == 'random':
            train_mask, val_mask, test_mask = cross_validation_gen(g.ndata['label'])
            g.ndata['train_mask'], g.ndata['val_mask'],  g.ndata['test_mask'] = train_mask, val_mask, test_mask

    elif data_name == 'arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name = 'ogbn-arxiv')
        g = dataset[0][0]
        g.ndata['label'] = dataset[0][1]
        # split_idx = dataset.get_idx_split()
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # g.ndata['train_mask'] = torch.zeros_like(g.ndata['label'], dtype=torch.bool)
        # g.ndata['train_mask'][train_idx] = True
        # g.ndata['val_mask'] = torch.zeros_like(g.ndata['label'], dtype=torch.bool)
        # g.ndata['val_mask'][valid_idx] = True
        # g.ndata['test_mask'] = torch.zeros_like(g.ndata['label'], dtype=torch.bool)
        # g.ndata['test_mask'][test_idx] = True
    elif data_name == 'products':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name = 'ogbn-products')
        g = dataset[0][0]
        g.ndata['label'] = dataset[0][1]
    else:
        assert Exception('Invalid Dataset')

    g.ndata['node_assignment'] = torch.load(pretrain_label_dir+'/metis_label_{}.pt'.format(data_name))
    g = preprocess(g, no_self_loop)
    # normalize graphs with discrete features
    norm = StandardScaler()
    norm.fit(g.ndata['feat'])
    g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
    
    if mask_edge:
        _, _, val_edges, _, test_edges, _ = pickle.load(open(tvt_addr, 'rb'))
        lst = []
        lst.append(g.edge_ids(val_edges[:,0], val_edges[:,1]))
        lst.append(g.edge_ids(val_edges[:,1], val_edges[:,0]))
        lst.append(g.edge_ids(test_edges[:,0], test_edges[:,1]))
        lst.append(g.edge_ids(test_edges[:,1], test_edges[:,0]))
        lst = torch.cat(lst)
        g.remove_edges(lst)
    
    return g
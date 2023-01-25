from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset, WikiCSDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorPhysicsDataset
import dgl
import torch
import pickle
from copy import deepcopy
import scipy.sparse as sp
import numpy as np
import os 

def mask_test_edges(adj_orig, val_frac, test_frac):

    # Remove diagonal elements
    adj = deepcopy(adj_orig)
    # set diag as all zero
    adj.setdiag(0)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj, 1)
    edges = sparse_to_tuple(adj_triu)[0]
    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * val_frac))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = edges[all_edge_idx[num_val + num_test:]]

    noedge_mask = np.ones(adj.shape) - adj
    noedges = np.asarray(sp.triu(noedge_mask, 1).nonzero()).T
    all_edge_idx = list(range(noedges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges_false = noedges[test_edge_idx]
    val_edges_false = noedges[val_edge_idx]

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T


    train_mask = np.ones(adj_train.shape)
    for edges_tmp in [val_edges, val_edges_false, test_edges, test_edges_false]:
        for e in edges_tmp:
            assert e[0] < e[1]
        train_mask[edges_tmp.T[0], edges_tmp.T[1]] = 0
        train_mask[edges_tmp.T[1], edges_tmp.T[0]] = 0

    train_edges = np.asarray(sp.triu(adj_train, 1).nonzero()).T
    train_edges_false = np.asarray((sp.triu(train_mask, 1) - sp.triu(adj_train, 1)).nonzero()).T

    # NOTE: all these edge lists only contain single direction of edge!
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

dataset_classes = {
    'cora':CoraGraphDataset,
    'pubmed':PubmedGraphDataset, 
    'citeseer':CiteseerGraphDataset, 
    'wiki_cs':WikiCSDataset,
    'co_cs':CoauthorCSDataset, 
    'co_computer':AmazonCoBuyComputerDataset, 
    'co_photo':AmazonCoBuyPhotoDataset,
    'co_phy':CoauthorPhysicsDataset
}
os.mkdir('links')
os.mkdir('pretrain_labels')

for k, v in dataset_classes.items():
    g = v()[0]
    total_pos_edges = torch.randperm(g.num_edges())
    adj_train = g.adjacency_matrix(scipy_fmt='csr')
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_train, 0.1, 0.2)
    tvt_edges_file = f'links/{k}_tvtEdges.pkl'
    pickle.dump((train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false), open(tvt_edges_file, 'wb'))
    node_assignment = dgl.metis_partition_assignment(g, 10)
    torch.save(node_assignment, f'pretrain_labels/metis_label_{k}.pt')

for dataset in ['chameleon', 'film', 'squirrel']:
    g, _ = dgl.load_graphs(f'hetero_graphs/{dataset}.bin')
    g = g[0]
    total_pos_edges = torch.randperm(g.num_edges())
    adj_train = g.adjacency_matrix(scipy_fmt='csr')
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_train, 0.1, 0.2)
    tvt_edges_file = f'links/{dataset}_tvtEdges.pkl'
    pickle.dump((train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false), open(tvt_edges_file, 'wb'))
    node_assignment = dgl.metis_partition_assignment(g, 10)
    torch.save(node_assignment, f'pretrain_labels/metis_label_{dataset}.pt')

# arxiv

from ogb.nodeproppred import DglNodePropPredDataset
dataset = DglNodePropPredDataset(name = 'ogbn-arxiv')
g = dataset[0][0]
total_edges = torch.stack(g.edges()).t()

num_val_links = 30000
num_test_links = 60000
num_train_links = 210000

assert num_train_links + num_test_links + num_val_links < len(total_edges), 'Not enough edges to sample '

# *100 here means we can set negative ratio for upto 100 when training for link prediciton downstream task
negatives = torch.stack(g.global_uniform_negative_sampling(num_val_links + num_test_links + num_train_links*100)).t()
test_edges_false, val_edges_false, train_edges_false = negatives[:num_test_links].numpy(), \
    negatives[num_test_links:num_test_links+num_val_links].numpy(), negatives[num_test_links+num_val_links:].numpy()

indices = torch.randperm(len(total_edges)).numpy()
total_edges = total_edges[indices]
test_edges, val_edges, train_edges = total_edges[:num_test_links].numpy(), \
    total_edges[num_test_links:num_test_links+num_val_links].numpy(), total_edges[num_test_links+num_val_links:num_test_links+num_val_links+num_train_links].numpy()

dataset = 'arxiv'
tvt_edges_file = f'links/{dataset}_tvtEdges.pkl'
pickle.dump((train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false), open(tvt_edges_file, 'wb'))
node_assignment = dgl.metis_partition_assignment(g, 10)
torch.save(node_assignment, f'pretrain_labels/metis_label_{dataset}.pt')

# products

from ogb.nodeproppred import DglNodePropPredDataset
dataset = DglNodePropPredDataset(name = 'ogbn-products')
g = dataset[0][0]
total_edges = torch.stack(g.edges()).t()

num_val_links = 30000
num_test_links = 60000
num_train_links = 210000

assert num_train_links + num_test_links + num_val_links < len(total_edges), 'Not enough edges to sample '

# *100 here means we can set negative ratio for upto 100 when training for link prediciton downstream task
negatives = torch.stack(g.global_uniform_negative_sampling(num_val_links + num_test_links + num_train_links*100)).t()
test_edges_false, val_edges_false, train_edges_false = negatives[:num_test_links].numpy(), \
    negatives[num_test_links:num_test_links+num_val_links].numpy(), negatives[num_test_links+num_val_links:].numpy()

indices = torch.randperm(len(total_edges)).numpy()
total_edges = total_edges[indices]
test_edges, val_edges, train_edges = total_edges[:num_test_links].numpy(), \
    total_edges[num_test_links:num_test_links+num_val_links].numpy(), total_edges[num_test_links+num_val_links:num_test_links+num_val_links+num_train_links].numpy()

dataset = 'products'
tvt_edges_file = f'links/{dataset}_tvtEdges.pkl'
pickle.dump((train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false), open(tvt_edges_file, 'wb'))
node_assignment = dgl.metis_partition_assignment(g, 10)
torch.save(node_assignment, f'pretrain_labels/metis_label_{dataset}.pt')

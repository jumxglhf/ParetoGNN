import dgl
import src.evaluator
import torch
from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset, WikiCSDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorPhysicsDataset
import os, sys
import statistics
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0,
                    help='which GPU to run, -1 for cpu')
parser.add_argument('--batch_size', type=int, default=10240,
                    help='batch size for link prediciton.')
parser.add_argument('--neg_rate', type=int, default=1,
                    help='negative rate for link prediction.')
parser.add_argument('--data', type=str, required=True, 
                    help='Dataset to evaluate.')
parser.add_argument('--embedding_path_node', type=str, required=True, 
                    help='path for saved node embedding.')
parser.add_argument('--embedding_path_link', type=str, required=True, 
                    help='path for save node embedding (intended for link prediction downstream task).')

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

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
args = parser.parse_args()

device = f'cuda:{args.device}' if args.device != -1 else 'cpu'
batch_size = args.batch_size
neg_rate = args.neg_rate
dataset = args.data

if dataset in ['chameleon', 'squirrel', 'actor']:
    d, _ = dgl.load_graphs('hetero_graphs' + '/{}.bin'.format(dataset))
    g = d[0]
elif dataset == 'arxiv':
        dataset = DglNodePropPredDataset(name = 'ogbn-arxiv')
        g = dataset[0][0]
        g.ndata['label'] = dataset[0][1]
        num_nodes = g.num_nodes()
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        g.ndata["label"] = dataset[0][1].view(-1)
        g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"] = train_mask, val_mask, test_mask
        dataset = 'arxiv'
elif dataset == 'products':
    dataset = DglNodePropPredDataset(name = 'ogbn-products')
    g = dataset[0][0]
    g.ndata['label'] = dataset[0][1]
    num_nodes = g.num_nodes()
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)

    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    g.ndata["label"] = dataset[0][1].view(-1)
    g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"] = train_mask, val_mask, test_mask
    dataset = 'products'
else:
    g = dataset_classes[dataset]()[0]
    
metis_label = torch.load(f'pretrain_labels/metis_label_{dataset}.pt', map_location='cpu')
embedding_path_node = args.embedding_path_node
embedding_path_link = args.embedding_path_link
tvt_edges_file = f'links/{dataset}_tvtEdges.pkl'

with HiddenPrints():
    X_link = torch.load(embedding_path_link, map_location=device)
    X_node = torch.load(embedding_path_node, map_location=device)
    auc, auc_std, hits20, hits20_std = src.evaluator.fit_link_predictor(X_link, tvt_edges_file, device, batch_size, neg_rate, es_metric='auc', epochs=200, patience=50, repeat=2)
    if dataset in ['products', 'arxiv']:
        ssnc_acc, ssnc_acc_std = src.evaluator.fit_logistic_regression_neural_net_preset_splits(X_node, g.ndata['label'], \
            g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'], repeat=10, device=device)
        metis, metis_std = src.evaluator.fit_logistic_regression_neural_net(X_node, metis_label, device=device)
    else:
        ssnc_acc, ssnc_acc_std = src.evaluator.fit_logistic_regression(X_node.cpu().numpy(), g.ndata['label'], repeat=10)
        metis, metis_std = src.evaluator.fit_logistic_regression(X_node.cpu().numpy(), metis_label, repeat=10)
    nmi, nmi_std = src.evaluator.fit_node_clustering(X_node.cpu().numpy(), g.ndata['label'])

print('MEAN: AUC: {:.4f}, Hits@20: {:.4f}, ACC: {:.4f}, NMI:{:.4f}, METIS: {:0.4f}, HMEAN:{:0.4f}'.format(auc, hits20, ssnc_acc, nmi, metis, np.mean([auc, ssnc_acc, nmi, metis])))
print('STD : AUC: {:.4f}, Hits@20: {:.4f}, ACC: {:.4f}, NMI:{:.4f}, METIS: {:0.4f}'.format(auc_std, hits20_std, ssnc_acc_std, nmi_std, metis_std))
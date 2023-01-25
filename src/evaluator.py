import numpy as np
from sklearn import metrics
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
import logging
import sys 
import pickle 
from torch.utils.data import DataLoader
import torch
import src.model
import torch.nn.functional as F
import src.utils_lp
from tqdm import tqdm 
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score

def init_logger(filename=None):
    logger = logging.getLogger(__name__)
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    return logger

def fit_logistic_regression(X, y, data_random_seed=1, repeat=1, gpu=False):
     # transform targets to one-hot vector

     one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

     y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

     # normalize x
     X = normalize(X, norm='l2')

     accuracies = []
     for r in range(repeat):
         # set random state, this will ensure the dataset will be split exactly the same throughout training
         rng = np.random.RandomState(data_random_seed+r)
         # different random split after each repeat
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

         # grid search with one-vs-rest classifiers
         if not gpu:
             logreg = LogisticRegression(solver='liblinear', max_iter=10000)
         else:
             import cuml
             logreg = cuml.linear_model.Logistic_regression()
         c = 2.0 ** np.arange(-10, 11)
         cv = ShuffleSplit(n_splits=5, test_size=0.5)
         clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                            n_jobs=-1, cv=cv, verbose=0)
         clf.fit(X_train, y_train)

         y_pred = clf.predict_proba(X_test)
         y_pred = np.argmax(y_pred, axis=1)
         y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)

         test_acc = metrics.accuracy_score(y_test, y_pred)
         accuracies.append(test_acc)
     return np.mean(accuracies), np.std(accuracies)
     
def fit_node_clustering(kmeans_input, y, repeat=5):
    kmeans_input = normalize(kmeans_input, norm='l2')
    nclass = y.max().item()+1
    nmis = []
    for i in range(repeat):
        kmeans = KMeans(n_clusters=nclass, random_state=i).fit(kmeans_input)
        pred = kmeans.predict(kmeans_input)
        nmis.append(v_measure_score(y, pred))
    return np.mean(nmis), np.std(nmis)


def fit_logistic_regression_preset_splits(X, y, train_mask, val_mask, test_mask, opt, repeat=5):
    # normalize x
    # if opt != None:
    #     logger = init_logger(opt.checkpoint_path / 'run.log')
    #     logger.info('START EVALUATION on SSNC.')
    # else:
    #     print('START EVALUATION on SSNC.')
    if len(train_mask.shape) == 1:
        train_mask = train_mask.unsqueeze(1).cpu().numpy()
        train_mask = train_mask.unsqueeze(1).cpu().numpy()
        train_mask = train_mask.unsqueeze(1).cpu().numpy()

    X = normalize(X, norm='l2')
    accuracies = []
    for _ in range(repeat):
        for split_id in range(train_mask.shape[1]):
            # get train/val/test masks
            tmp_train_mask, tmp_val_mask, tmp_test_mask = train_mask[:, split_id], val_mask[:, split_id], test_mask[:, split_id]

            # make custom cv
            X_train, y_train = X[tmp_train_mask], y[tmp_train_mask]
            X_val, y_val = X[tmp_val_mask], y[tmp_val_mask]
            X_test, y_test = X[tmp_test_mask], y[tmp_test_mask]

            # grid search with one-vs-rest classifiers
            best_test_acc, best_acc = 0, 0
            for c in 2.0 ** np.arange(-10, 11):
                clf = OneVsRestClassifier(LogisticRegression(solver='liblinear',  C=c,))
                # clf = LogisticRegression(solver='liblinear', C=c, multi_class='ovr')
                clf.fit(X_train, y_train)

                y_pred = clf.predict_proba(X_val)
                y_pred = np.argmax(y_pred, axis=1)
                val_acc = metrics.accuracy_score(y_val, y_pred)
                if val_acc > best_acc:
                    best_acc = val_acc
                    y_pred = clf.predict_proba(X_test)
                    y_pred = np.argmax(y_pred, axis=1)
                    best_test_acc = metrics.accuracy_score(y_test, y_pred)
                    print(c, '{:.4f}'.format(best_test_acc))

            accuracies.append(best_test_acc)
    return np.mean(accuracies)
    # if opt != None:
    #     logger.info(f"Best Dev ACC: {best_acc}, Testing ACC: {np.mean(accuracies)}")
    # else:
    #     print(f"Best Dev ACC: {best_acc}, Testing ACC: {np.mean(accuracies)}")


def fit_link_predictor(X, tvt_edges_file, device, batch_size, neg_rate, es_metric='hits@20', epochs=200, patience=50, repeat=5):
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = pickle.load(open(tvt_edges_file, 'rb'))
    val_labels = np.concatenate((np.ones(val_edges.shape[0]), np.zeros(val_edges_false.shape[0])), axis=0)
    test_labels = np.concatenate((np.ones(test_edges.shape[0]), np.zeros(test_edges_false.shape[0])), axis=0)
    val_pairs = np.concatenate((val_edges, val_edges_false), axis=0)
    test_pairs = np.concatenate((test_edges, test_edges_false), axis=0)
    train_edges = torch.tensor(train_edges, dtype=torch.long)#.to(device)
    train_edges_false = torch.tensor(train_edges_false, dtype=torch.long)#.to(device)
    res_auc = []
    res_hits20 = []
    for _ in range(repeat):
        model = src.model.Link_Pred(X.shape[1])
        model = model.to(device)

        optim = torch.optim.Adam(model.parameters(),
                                lr=0.01,
                                weight_decay=0.01)
        model.zero_grad()
        X = X.to(device)
        X = F.normalize(X)
        labels_batch = torch.cat([torch.ones(batch_size), torch.zeros(batch_size*neg_rate)]).to(device)

        best_val_res = 0.0
        cnt_wait = 0
        neg_indices = torch.randperm(train_edges_false.size(0))
        if train_edges.shape[0] * neg_rate > train_edges_false.size(0):
            print('not much negative edges to sample')
            neg_indices = torch.cat([torch.randperm(train_edges_false.size(0)) for _ in range(int(train_edges_false.size(0)/(train_edges.shape[0] * neg_rate)+1))])
        for epoch in range(epochs):
            total_loss = 0
            dataloader = DataLoader(range(train_edges.shape[0]), batch_size, num_workers=0,
                        shuffle=True, drop_last=True, )#collate_fn=Collator(train_edges, train_edges_false, neg_rate))
            for i, perm in enumerate(dataloader):
                model.zero_grad()
                pos_edges =  train_edges[perm]
                neg_sample_idx = neg_indices[(neg_rate * len(perm))*i:(neg_rate * len(perm))*(i+1)]
                neg_edges = train_edges_false[neg_sample_idx]
                train_edges_batch = torch.cat([pos_edges, neg_edges], dim=0).t()
                z = torch.mul(X[train_edges_batch[0]], X[train_edges_batch[1]])
                logits = model(z)
                loss = F.binary_cross_entropy_with_logits(logits, labels_batch, \
                    pos_weight= torch.FloatTensor([neg_rate]).to(device))
                loss.backward()
                optim.step()

            with torch.no_grad():
                z = X[val_pairs.T[0]] * X[val_pairs.T[1]]
                logits_val = model(z.to(device)).detach().cpu()
                z = X[test_pairs.T[0]] * X[test_pairs.T[1]]
                logits_test = model(z.to(device)).detach().cpu()
            
            val_res = src.utils_lp.eval_ep_batched(logits_val, val_labels, val_edges.shape[0])
            if val_res[es_metric] >= best_val_res:
                cnt_wait = 0
                best_val_res = val_res[es_metric]
                test_res = src.utils_lp.eval_ep_batched(logits_test, test_labels, test_edges.shape[0])
                test_res['best_val'] = val_res[es_metric]
                print('Epoch {} Loss: {:.4f} lr: {:.4f} val: {:.4f} test: {:.4f}'.format(
                            epoch+1, total_loss, 0.0001, val_res[es_metric], test_res[es_metric]))
                print('Epoch {} Loss: {:.4f} lr: {:.4f} val: {:.4f} test: {:.4f}'.format(
                            epoch+1, total_loss, 0.0001, val_res['hits@20'], test_res['hits@20']))
            else:
                cnt_wait += 1
            
            if cnt_wait >= patience:
                break
        res_auc.append(test_res['auc'])
        res_hits20.append(test_res['hits@20'])
    return np.mean(res_auc), np.std(res_auc), np.mean(res_hits20), np.std(res_hits20)


# this function is used for the evaluation of ogbn arxiv or datasets larger
def fit_logistic_regression_neural_net(X, y, hid_dim=None, opt=None, data_random_seed=1, epoch=5000, patience=500, device='cpu'):

    class LogReg(torch.nn.Module):
        def __init__(self, nfeat, nclass):
            super(LogReg, self).__init__()
            self.fc = torch.nn.Linear(nfeat, nclass)

            for m in self.modules():
                self.weights_init(m)

        def weights_init(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        def forward(self, seq):
            ret = self.fc(seq)
            return ret

    # normalize x
    X = F.normalize(X)

    test_accs = []
    # set random state, this will ensure the dataset will be split exactly the same throughout training
    rng = np.random.RandomState(data_random_seed)
    # different random split after each repeat
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)
    X_test = X_test.to(device)

    cv = ShuffleSplit(n_splits=5, test_size=0.5)
    for train_idx, val_idx in cv.split(X_train):
        # torch.nn.Linear(X.shape[1], y.max()+1).to(device)
        loss = torch.nn.CrossEntropyLoss()

        X_train_ = X_train[train_idx].to(device)
        X_val_ = X_train[val_idx].to(device)
        Y_train_ = y_train[train_idx].to(device)
        Y_val_ = y_train[val_idx]

        global_best = 0
        test_acc = 0
        for weight_decay in 2.0 ** np.arange(-10, 11, 2):
            model = LogReg(X.shape[1], y.max()+1).to(device) # LR(X.shape[1], y.max()+1, hid_dim).to(device)
            optim = torch.optim.AdamW(model.parameters(),
                                lr=0.01,
                                weight_decay=weight_decay)
            cnt = 0
            local_best = 0
            local_test_acc = 0
            for i in range(epoch):
                model.train()
                cnt += 1
                model.zero_grad()
                logits = model(X_train_)
                l = loss(logits, Y_train_)
                l.backward()
                optim.step()

                model.eval()
                with torch.no_grad():
                    logits_val = model(X_val_)
                    val_pred = torch.argmax(logits_val, dim=1).cpu()
                    val_acc = metrics.accuracy_score(Y_val_, val_pred)
                    if val_acc > local_best:
                        local_best = val_acc
                        cnt = 0
                        test_pred = torch.argmax(model(X_test), dim=1).cpu()
                        local_test_acc = metrics.accuracy_score(y_test, test_pred)

                if cnt == patience:
                    if local_best > global_best:
                        global_best = local_best
                        test_acc = local_test_acc
                    break

            if cnt != patience:
                if local_best > global_best:
                    global_best = local_best
                    test_acc = local_test_acc
        
        test_accs.append(test_acc)
    return np.mean(test_accs), np.std(test_accs)

# this function is used for the evaluation of ogbn arxiv or datasets larger
def fit_logistic_regression_neural_net_preset_splits(X, y, train_idx, val_idx, test_idx, hid_dim=None, opt=None, epoch=50000, patience=500, device='cpu', repeat=5):

    class LogReg(torch.nn.Module):
        def __init__(self, nfeat, nclass):
            super(LogReg, self).__init__()
            self.fc = torch.nn.Linear(nfeat, nclass)

            for m in self.modules():
                self.weights_init(m)

        def weights_init(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        def forward(self, seq):
            ret = self.fc(seq)
            return ret

    # normalize x
    X = F.normalize(X)
    y = y.squeeze()
    test_accs = []
    loss = torch.nn.CrossEntropyLoss()

    X_train_ = X[train_idx].to(device)
    X_val_ = X[val_idx].to(device)
    Y_train_ = y[train_idx].to(device)
    Y_val_ = y[val_idx]
    X_test = X[test_idx].to(device)
    Y_test = y[test_idx]

    for r in range(repeat):
        global_best = 0
        test_acc = 0
        for weight_decay in 2.0 ** np.arange(-10, 11, 2):
            model = LogReg(X.shape[1], y.max()+1).to(device) # LR(X.shape[1], y.max()+1, hid_dim).to(device)
            optim = torch.optim.AdamW(model.parameters(),
                                lr=0.01,
                                weight_decay=weight_decay)
            cnt = 0
            local_best = 0
            local_test_acc = 0
            for i in range(epoch):
                model.train()
                cnt += 1
                model.zero_grad()
                logits = model(X_train_)
                l = loss(logits, Y_train_)
                l.backward()
                optim.step()

                model.eval()
                with torch.no_grad():
                    logits_val = model(X_val_)
                    val_pred = torch.argmax(logits_val, dim=1).cpu()
                    val_acc = metrics.accuracy_score(Y_val_, val_pred)
                    if val_acc > local_best:
                        local_best = val_acc
                        cnt = 0
                        test_pred = torch.argmax(model(X_test), dim=1).cpu()
                        local_test_acc = metrics.accuracy_score(Y_test, test_pred)

                if cnt == patience:
                    if local_best > global_best:
                        global_best = local_best
                        test_acc = local_test_acc
                        print(r, weight_decay, i, local_best, test_acc)
                    break

            if cnt != patience:
                if local_best > global_best:
                    global_best = local_best
                    test_acc = local_test_acc
                    print(r, weight_decay, i, local_best, test_acc)
        
        test_accs.append(test_acc)

    return np.mean(test_accs), np.std(test_accs)

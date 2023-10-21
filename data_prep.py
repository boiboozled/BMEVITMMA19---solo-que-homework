import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs
import scipy.sparse as sp

data_path = './data/facebook'
save_path_graphs = './data/facebook/graphs/graphs.bin'
USER = 0

def load_data(data_path, user_id):
    file_edges = f'{data_path}/{user_id}.edges'
    file_feats = f'{data_path}/{user_id}.feat'

    # load edges
    src = []
    dst = []
    with open(file_edges) as f:
        for line in f:
            e1, e2 = tuple(int(x) -1 for x in line.strip().split())
            src.append(e1)
            dst.append(e2)
    src = np.array(src)
    dst = np.array(dst)
    # load features
    num_nodes = 0
    feats = []
    with open(file_feats) as f:
        for line in f:
            num_nodes += 1
            # remove first column (node id)
            a = [int(x) for x in line.strip().split()[1:]]
            feats.append(torch.tensor(a, dtype=torch.float32))

    feats = torch.stack(feats)
    # build graph
    g = dgl.graph((src, dst))
    g.ndata['feat'] = feats

    print(f'Loaded data int DGL graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.')
    return g

def split_data(g,test_size=0.3):
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * test_size)  # number of edges in test set
    train_size = g.number_of_edges() - test_size  # number of edges in train set

    # get positive edges for test and train
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    # split the negative edges for training and testing 
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # construct positive and negative graphs for training and testing
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    # training graph
    train_g = dgl.remove_edges(g, eids[:test_size])
    train_g = dgl.add_self_loop(train_g)

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g


# load data
g = load_data(data_path, USER)
# split data
train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = split_data(g)

# save graphs
save_graphs(save_path_graphs, [train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g])
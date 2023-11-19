import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs
import scipy.sparse as sp
import pickle
import pandas as pd
import networkx as nx

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
data_path = './data/facebook/'
save_path_graphs = './data/facebook-processed/'
USER = 0
FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]

def load_data(data_path, user_id):
    """Load data for a single user from raw files
    
    Arguments:
        data_path {str} -- path to folder with data
        user_id {int} -- user id
        
    Returns:
        g {networkx graph} -- graph with features"""
    file_edges = f'{data_path}{user_id}.edges'
    file_feats = f'{data_path}{user_id}.feat'

    # load edges
    with open(file_edges) as f:
        g = nx.read_edgelist(f, nodetype=int)

    # Add ego user (directly connected to all other nodes)
    g.add_node(user_id)
    for node in g.nodes():
        if node != user_id:
            g.add_edge(user_id, node)
    

    # load features
    df = pd.read_table(file_feats, sep=' ', header=None,index_col=0)
    nx.set_node_attributes(g,df.to_dict('index')) # add features to graph

    assert nx.is_connected(g), 'Graph is not connected'
    print(f'Loaded data from {data_path}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.')

    return g

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.
    
    
    Arguments:
        sparse_mx {sparse matrix} -- sparse matrix
        
    Returns:
        coords {array} -- coordinates of non-zero elements
        values {array} -- values of non-zero elements
        shape {array} -- shape of matrix
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj, test_frac=.2, val_frac=.1):
    """ Function to build test set with
    
    Arguments:
        adj {sparse matrix} -- adjacency matrix
        test_frac {float} -- fraction of edges to be used for test set
        val_frac {float} -- fraction of edges to be used for validation set
    
    Returns:
        adj_train {sparse matrix} -- adjacency matrix with hidden edges removed
        train_edges {array} -- array of hidden edges
        train_edges_false {array} -- array of false edges
        val_edges {array} -- array of validation edges
        val_edges_false {array} -- array of false validation edges
        test_edges {array} -- array of test edges
        test_edges_false {array} -- array of false test edges
    """
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * val_frac))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def save_ego_user_data(user_id, data_path, save_path_graphs):
    """Save data for a single user
    
    Arguments:
        user_id {int} -- user id
        data_path {str} -- path to folder with data
        save_path_graphs {str} -- path to folder where graphs should be saved
    """
    g_user = load_data(data_path, user_id)
    adj_sparse = nx.to_scipy_sparse_array(g_user)
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=0.15, val_frac=0.15)
    g_user_train = nx.from_scipy_sparse_array(adj_train) # new graph object with only non-hidden edges

    # save graphs with pickle
    with open(save_path_graphs+f'{user_id}_train.pickle', 'wb') as f:
        pickle.dump(g_user_train, f)
    # save edges with numpy
    np.save(save_path_graphs+f'{user_id}_train_edges.npy', train_edges)
    np.save(save_path_graphs+f'{user_id}_train_edges_false.npy', train_edges_false)
    np.save(save_path_graphs+f'{user_id}_val_edges.npy', val_edges)
    np.save(save_path_graphs+f'{user_id}_val_edges_false.npy', val_edges_false)
    np.save(save_path_graphs+f'{user_id}_test_edges.npy', test_edges)
    np.save(save_path_graphs+f'{user_id}_test_edges_false.npy', test_edges_false)
    print(f'Graph {user_id} saved')

# Loading function for later use
def load_user_data(user_id, load_path_graphs):
    """Load data for a single user
    
    Arguments:
        user_id {int} -- user id
        load_path_graphs {str} -- path to folder with graphs
    
    Returns:
        g_user_train {networkx graph} -- graph with features
        train_edges {array} -- array of hidden edges
        train_edges_false {array} -- array of false edges
        val_edges {array} -- array of validation edges
        val_edges_false {array} -- array of false validation edges
        test_edges {array} -- array of test edges
        test_edges_false {array} -- array of false test edges
    """
    # load graphs with pickle
    with open(load_path_graphs+f'{user_id}_train.pickle', 'rb') as f:
        g_user_train = pickle.load(f)
    # load edges with numpy
    train_edges = np.load(load_path_graphs+f'{user_id}_train_edges.npy')
    train_edges_false = np.load(load_path_graphs+f'{user_id}_train_edges_false.npy')
    val_edges = np.load(load_path_graphs+f'{user_id}_val_edges.npy')
    val_edges_false = np.load(load_path_graphs+f'{user_id}_val_edges_false.npy')
    test_edges = np.load(load_path_graphs+f'{user_id}_test_edges.npy')
    test_edges_false = np.load(load_path_graphs+f'{user_id}_test_edges_false.npy')
    print(f'Graph {user_id} loaded')
    return g_user_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


for user in FB_EGO_USERS:
    save_ego_user_data(user, data_path, save_path_graphs)
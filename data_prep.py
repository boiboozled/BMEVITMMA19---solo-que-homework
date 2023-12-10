import numpy as np
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.seed import seed_everything   
import pandas as pd
import networkx as nx

#setting random seed for everything for reproducibility
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)

# data paths
data_path = './data/facebook/'
save_path_graphs = './data/facebook-processed/'
# create folder if it doesn't exist 
import os
if not os.path.exists(save_path_graphs):
    os.makedirs(save_path_graphs)

# setting variables
USER = 0
FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
val_frac = 0.15
test_frac = 0.15

def load_data(data_path, user_id):
    """Load data for a single user from raw files
    
    Arguments:
        data_path {str} -- path to folder with data
        user_id {int} -- user id
        
    Returns:
        g {networkx graph} -- graph with features"""
    file_edges = f'{data_path}{user_id}.edges'
    file_feats = f'{data_path}{user_id}.feat'
    file_ego_feats = f'{data_path}{user_id}.egofeat'

    # load edges
    with open(file_edges) as f:
        g = nx.read_edgelist(f, nodetype=int)

    # Add ego user (directly connected to all other nodes)
    ego_df = pd.read_table(file_ego_feats, sep=' ', header=None)
    ego_df.index = [user_id]
    ego_df.columns = [x+1 for x in ego_df.columns]

    g.add_node(user_id)
    for node in g.nodes():
        if node != user_id:
            g.add_edge(user_id, node)
    

    # load features
    df = pd.read_table(file_feats, sep=' ', header=None,index_col=0)
#    nx.set_node_attributes(g,df.to_dict('index')) # add features to graph
#    nx.set_node_attributes(g,ego_df.to_dict('index')) # add ego features to graph

    assert nx.is_connected(g), 'Graph is not connected'
    print(f'Loaded data from {data_path}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.')

    # create torch_geometric data object
    g_pyg = from_networkx(g)
    # add features to data object
    df = pd.concat([ego_df,df])
    node_feat_array = np.array(df.values, dtype=np.float32)
    g_pyg.x = torch.tensor(node_feat_array, dtype=torch.float)

    #save graph
    torch.save(g_pyg, f'{save_path_graphs}{user_id}_graph.pt')

    return g_pyg

def get_pos_neg_edges(graph):
    """Get positive and negative edges from graph
    
    Arguments:
        graph {torch_geometric.Data graph} -- graph
    
    Returns:
        pos_edges {np.array} -- array of positive edges
        neg_edges {np.array} -- array of negative edges
    """
    pos_edges = graph.edge_label_index[:,graph.edge_label==1].t().numpy()
    neg_edges = graph.edge_label_index[:,graph.edge_label==0].t().numpy()
    return pos_edges, neg_edges

def save_ego_user_data(user_id, data_path, save_path_graphs,val_frac=0.15,test_frac=0.15):
    """Save data for a single user
    
    Arguments:
        user_id {int} -- user id
        data_path {str} -- path to folder with data
        save_path_graphs {str} -- path to folder where graphs should be saved
    """
    g_user = load_data(data_path, user_id)
    
    # create train-val-test split
    g_user_train, g_user_val, g_user_test = RandomLinkSplit(num_val=val_frac, num_test=test_frac,is_undirected=True)(g_user)

    # get positive and negative edges
    pos_edges_train, neg_edges_train = get_pos_neg_edges(g_user_train)
    pos_edges_val, neg_edges_val = get_pos_neg_edges(g_user_val)
    pos_edges_test, neg_edges_test = get_pos_neg_edges(g_user_test)
    
    #adj_sparse = nx.to_scipy_sparse_array(g_user)
    #adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=0.15, val_frac=0.15)
    #g_user_train = nx.from_scipy_sparse_array(adj_train) # new graph object with only non-hidden edges

    # save graphs 
    torch.save(g_user_train, save_path_graphs+f'{user_id}_train_graph.pt')
    torch.save(g_user_val, save_path_graphs+f'{user_id}_val_graph.pt')
    torch.save(g_user_test, save_path_graphs+f'{user_id}_test_graph.pt')
    # save edges with numpy
    np.save(save_path_graphs+f'{user_id}_train_edges.npy', pos_edges_train)
    np.save(save_path_graphs+f'{user_id}_train_edges_false.npy', neg_edges_train)
    np.save(save_path_graphs+f'{user_id}_val_edges.npy', pos_edges_val)
    np.save(save_path_graphs+f'{user_id}_val_edges_false.npy', neg_edges_val)
    np.save(save_path_graphs+f'{user_id}_test_edges.npy', pos_edges_test)
    np.save(save_path_graphs+f'{user_id}_test_edges_false.npy', neg_edges_test)
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
    # load graphs
    g_user_train = torch.load(load_path_graphs+f'{user_id}_train_graph.pt')
    g_user_val = torch.load(load_path_graphs+f'{user_id}_val_graph.pt')
    g_user_test = torch.load(load_path_graphs+f'{user_id}_test_graph.pt')
    # load edges with numpy
    train_edges = np.load(load_path_graphs+f'{user_id}_train_edges.npy')
    train_edges_false = np.load(load_path_graphs+f'{user_id}_train_edges_false.npy')
    val_edges = np.load(load_path_graphs+f'{user_id}_val_edges.npy')
    val_edges_false = np.load(load_path_graphs+f'{user_id}_val_edges_false.npy')
    test_edges = np.load(load_path_graphs+f'{user_id}_test_edges.npy')
    test_edges_false = np.load(load_path_graphs+f'{user_id}_test_edges_false.npy')
    print(f'Graph {user_id} loaded')
    return g_user_train, g_user_val, g_user_test, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

# only run if main
if __name__ == "__main__":
    # save data for all users
    for user in FB_EGO_USERS:
        save_ego_user_data(user, data_path, save_path_graphs,val_frac=val_frac,test_frac=test_frac)
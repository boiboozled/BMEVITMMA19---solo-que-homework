import torch
from torch_geometric.nn import GCNConv, VGAE
from typing import Optional, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import numpy as np
import torch_geometric



class VGraphEncoder(torch.nn.Module):
    """Encoder layer for VGAE."""
    def __init__(self, input_channels:int, hidden_channels1: Optional[int]=64, hidden_channels2:Optional[int]=8, output_channels:Optional[int]=2):
        """Initialization function.
        
        Arguments:
            input_channels {int} -- number of input channels
            hidden_channels1 {int} -- number of hidden channels in first layer
            hidden_channels2 {int} -- number of hidden channels in second layer
            output_channels {int} -- number of output channels
        """
        super(VGraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.conv3 = GCNConv(hidden_channels2, 2*output_channels)

        self.mu = GCNConv(2*output_channels, output_channels)
        self.logvar = GCNConv(2*output_channels, output_channels)

    def forward(self, x:torch.Tensor, edge_index:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Arguments:
            x {torch.tensor} -- input tensor
            edge_index {torch.tensor} -- edge index
            
        Returns:
            mu {torch.tensor} -- mean of latent space
            logvar {torch.tensor} -- log variance of latent space
        """
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        mu = self.mu(x, edge_index)
        logvar = self.logvar(x, edge_index)
        return mu, logvar
    

class my_VGAE(VGAE):
    """Variational Graph Auto-Encoder from the pytorch geometric version, with a different test function."""
    def __init__(self, encoder: torch.nn.Module, decoder: Optional[torch.nn.Module]=None):
        """Initialization function.
        
        Arguments:
            encoder {torch.nn.Module} -- encoder module
            decoder {torch.nn.Module} -- decoder module
            
        """
        super(my_VGAE, self).__init__(encoder, decoder)

    def test(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> Tuple[float, float, np.array]:
        """Test function.
        
        Arguments:
            z {torch.tensor} -- latent space representations of nodes
            pos_edge_index {torch.tensor} -- positive edges
            neg_edge_index {torch.tensor} -- negative edges
            
        Returns:
            roc {float} -- ROC AUC score
            ap {float} -- average precision score
            conf_mtx {np.array} -- confusion matrix
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        roc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        conf_mtx = confusion_matrix(y, np.round(pred))
        return roc, ap, conf_mtx

def train_step_vgae(model: my_VGAE, x: torch.Tensor, num_nodes:int, edge_index: torch.Tensor, optimizer: torch.optim.Optimizer):
    """Train function for VGAE. Performs one training step.
    
    Arguments:
        model {my_VGAE} -- model
        x {torch.tensor} -- input tensor of node features
        edge_index {torch.tensor} -- edge indeces
        optimizer {torch.optim.Optimizer} -- optimizer
        
    Returns:
        loss {torch.tensor} -- loss of the training step
    """
    model.train()
    optimizer.zero_grad()

    z = model.encode(x, edge_index)
    loss = model.recon_loss(z, edge_index)

    loss = loss + (1 / num_nodes) * model.kl_loss()

    loss.backward()
    optimizer.step()
    return loss

def test_vgae(graph: torch_geometric.data.Data,pos_edge_index: torch.Tensor, neg_edge_index:torch.Tensor, model: my_VGAE) -> Tuple[float, float, np.array]:
    """Test function for VGAE.
    
    Arguments:
        graph {torch_geometric.data.Data} -- graph
        pos_edge_index {torch.tensor} -- positive edges
        neg_edge_index {torch.tensor} -- negative edges
        
    Returns:
        roc {float} -- ROC AUC score
        ap {float} -- average precision score
        conf_mtx {np.array} -- confusion matrix
    """
    model.eval()
    x = graph.x
    with torch.no_grad():
        z = model.encode(x, graph.edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

def grid_search_vgae(graph: torch_geometric.data.Data, graph_train: torch_geometric.data.Data, graph_val: torch_geometric.data.Data, hidden_channels1_list: Optional[list]=[8, 16, 32, 64],\
                     hidden_channels2_list:Optional[list]=[2, 4, 8, 16], num_epochs:int=100, learning_rate_list:Optional[list]=[0.005, 0.002, 0.001, 0.0008, 0.0005],\
                     weight_decay_list:Optional[list]=[0, 0.0005, 0.001, 0.005, 0.01], verbose:Optional[int]=1) :
    """Grid search for VGAE.
    
    Arguments:
        graph {torch_geometric.data.Data} -- graph
        graph_train {torch_geometric.data.Data} -- training graph
        graph_val {torch_geometric.data.Data} -- validation graph
        hidden_channels1_list {list} -- list of hidden channels for first layer
        hidden_channels2_list {list} -- list of hidden channels for second layer
        num_epochs {int} -- number of epochs
        learning_rate_list {list} -- list of learning rates
        weight_decay_list {list} -- list of weight decays
        verbose {int} -- verbosity level during grid search.
        
    Returns:
        best_h_ch1 {int} -- number of hidden channels for first layer in model with best AUC
        best_h_ch2 {int} -- number of hidden channels for second layer in model with best AUC
        best_lr {float} -- learning rate in model with best AUC
        best_wd {float} -- weight decay in model with best AUC
    """
    # grid search
    final_aucs = []
    final_aps = []
    final_cfmtxs = []
    print("Start grid search for VGAE")
    print(f"Number of hyperparameter combinations: {len(hidden_channels1_list)*len(hidden_channels2_list)*len(learning_rate_list)*len(weight_decay_list)}")
    for h_ch1 in hidden_channels1_list:
        for h_ch2 in hidden_channels2_list:
            for lr in learning_rate_list:
                for wd in weight_decay_list:
                    # setting hyperparameters
                    num_epochs = 100
                    out_channels = 2
                    hidden_channels1 = h_ch1
                    hidden_channels2 = h_ch2
                    input_channels = graph.num_node_features

                    # model
                    model = my_VGAE(VGraphEncoder(input_channels,hidden_channels1=hidden_channels1, hidden_channels2=hidden_channels2, output_channels=out_channels))

                    # data
                    x = graph.x
                    edge_index = graph_train.edge_index

                    # optimizer
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,  weight_decay=wd)

                    for epoch in range(num_epochs):
                        loss = train_step_vgae(model, x, graph_train.num_nodes, edge_index, optimizer)
                        auc, ap, conf_mtx = test_vgae(graph_val,graph_val.edge_label_index[:,graph_val.edge_label==1], graph_val.edge_label_index[:,graph_val.edge_label==0], model)
                

                    final_aucs.append(auc)
                    final_aps.append(ap)
                    final_cfmtxs.append(conf_mtx)
                    if verbose>=1:
                        print(f'hidden_channels1: {h_ch1}, hidden_channels2: {h_ch2}, learning_rate: {lr}, weight_decay: {wd}')
                        print(f'Final loss: {loss.item()}, AUC: {auc}, AP: {ap}, Confusion Matrix: {conf_mtx}')
    # get index of max auc
    max_auc_index = np.argmax(final_aucs)
    # save hyperparameters of max auc
    # Determine the corresponding hyperparameters using the indices
    best_hidden_channels1_index = (max_auc_index  % len(hidden_channels1_list))
    best_hidden_channels2_index = (max_auc_index // len(hidden_channels1_list)) % len(hidden_channels2_list)
    best_learning_rate_index = (max_auc_index // (len(hidden_channels1_list) * len(hidden_channels2_list))) % len(learning_rate_list)
    best_weight_decay_index = (max_auc_index // (len(hidden_channels1_list) * len(hidden_channels2_list) * len(learning_rate_list))) % len(weight_decay_list)

    # Retrieve the best hyperparameters
    best_h_ch1 = hidden_channels1_list[best_hidden_channels1_index]
    best_h_ch2 = hidden_channels2_list[best_hidden_channels2_index]
    best_lr = learning_rate_list[best_learning_rate_index]
    best_wd = weight_decay_list[best_weight_decay_index]

    if verbose>=1:
        # print max auc
        print(f"Max AUC: {final_aucs[max_auc_index]}")
        # get max auc's ap
        print(f"AP of max AUC: {final_aps[max_auc_index]}")
        # get max auc's confusion matrix
        print(f"Confusion matrix of max AUC: {final_cfmtxs[max_auc_index]}")
        print(f"Best hidden_channels1: {best_h_ch1}, Best hidden_channels2: {best_h_ch2}, Best learning_rate: {best_lr}, Best weight_decay: {best_wd}")

    return best_h_ch1, best_h_ch2, best_lr, best_wd

import numpy as np
import torch
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
def train(model, train_data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    out = model.decode(z, train_data.edge_label_index).view(-1)
    loss = criterion(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return loss

def test(data, model):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    auc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    ap = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    conf_mtx = confusion_matrix(data.edge_label.cpu().numpy(), np.round(out.cpu().numpy()))
    return auc, ap, conf_mtx

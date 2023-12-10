from data_prep import load_user_data
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from GNN import GNN, train, test
from torch_geometric.seed import seed_everything   
from matplotlib import pyplot as plt
import pandas as pd

#setting random seed for everything for reproducibility
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)

# load data
user_id = 0
data_path = './data/facebook-processed/'
g_user = torch.load(data_path+f'{user_id}_graph.pt')
adj_sparse = to_scipy_sparse_matrix(g_user.edge_index)
adj_sparse = adj_sparse.tocsr()
g_user_train, g_user_val, g_user_test, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = load_user_data(user_id, data_path)

# print info
print("Total nodes:", adj_sparse.shape[0])
print("Total edges:", int(adj_sparse.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print("Training edges (positive):", len(train_edges))
print("Training edges (negative):", len(train_edges_false))
print("Validation edges (positive):", len(val_edges))
print("Validation edges (negative):", len(val_edges_false))
print("Test edges (positive):", len(test_edges))
print("Test edges (negative):", len(test_edges_false))


# RUN GNN

print("\n\n")
print(40*"-"+" Running GNN "+40*"-")
gnn_model = GNN(g_user_train.num_node_features, 16, 1)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# train
losses = []
aucs = []
aps = []
for epoch in range(1, 101):
    loss = train(gnn_model, g_user_train, optimizer, criterion)
    val_auc, val_ap, _ = test(g_user_val, gnn_model)
    losses.append(loss.item())
    aucs.append(val_auc)
    aps.append(val_ap)
#    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')

test_auc, test_ap, conf_mtx_gnn = test(g_user_test, gnn_model)
print(f'GNN Test AUC: {test_auc:.4f}')
print(f'GNN Test AP: {test_ap:.4f}')
print('GNN Confusion matrix (test): \n', str(conf_mtx_gnn))

# plot losses

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.savefig('./plots/gnn_losses.png', dpi=300, bbox_inches='tight')
plt.clf()

# plot aucs and aps
plt.plot(aucs, label='auc')
plt.plot(aps, label='ap')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('ROC AUC and AP scores over epochs')
plt.savefig('./plots/gnn_scores.png', dpi=300, bbox_inches='tight')

# save test scores in csv
scores = pd.DataFrame({'auc': [test_auc], 'ap': [test_ap], 'conf_mtx': [conf_mtx_gnn]})
scores.to_csv('./scores/gnn_scores.csv', index=False)
scores.to_pickle('./scores/gnn_scores.pkl')
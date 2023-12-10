from vgae import my_VGAE, VGraphEncoder, test_vgae, grid_search_vgae,train_step_vgae
import torch
from data_prep import load_user_data
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import matplotlib.pyplot as plt
from torch_geometric.seed import seed_everything
import pandas as pd


#setting random seed for everything for reproducibility
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)

DO_HYPERPARAMETER_TUNING = False


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


# RUN VGAE
print("\n\n")
print(40*"-"+" Running VGAE "+40*"-")

if DO_HYPERPARAMETER_TUNING:
    hidden_channels1_list = [8, 16, 32, 64]
    hidden_channels2_list = [2, 4, 8, 16]
    learning_rate_list = [0.005, 0.002, 0.001, 0.0008, 0.0005]
    weight_decay_list = [0, 0.0005, 0.001, 0.005, 0.01]

    hidden_channels1,hidden_channels2,learning_rate,weight_decay = grid_search_vgae(g_user,g_user_train, g_user_val, hidden_channels1_list=hidden_channels1_list, hidden_channels2_list=hidden_channels2_list, learning_rate_list=learning_rate_list, weight_decay_list=weight_decay_list, num_epochs=100, verbose=1)
else:
    hidden_channels1 = 64
    hidden_channels2 = 8
    learning_rate = 0.01
    weight_decay = 0.0005
# setting hyperparameters
num_epochs = 100
out_channels = 2
input_channels = g_user.num_node_features

# model
model = my_VGAE(VGraphEncoder(input_channels,hidden_channels1=hidden_channels1,hidden_channels2=hidden_channels2, output_channels=out_channels))

# data
x = g_user.x
edge_index = g_user_train.edge_index

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=weight_decay)

losses = []
aucs = []
aps = []
for epoch in range(num_epochs):
    loss = train_step_vgae(model, x, g_user_train.num_nodes, edge_index, optimizer)
    auc, ap, _ = test_vgae(g_user_val,g_user_val.edge_label_index[:,g_user_val.edge_label==1], g_user_val.edge_label_index[:,g_user_val.edge_label==0], model)
    
    losses.append(loss.item())
    aucs.append(auc)
    aps.append(ap)
    #print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss.item(), auc, ap))


# test
test_auc, test_ap, test_conf_mtx = test_vgae(g_user_test,g_user_test.edge_label_index[:,g_user_test.edge_label==1], g_user_test.edge_label_index[:,g_user_test.edge_label==0], model)
print(f'VGAE Test AUC: {test_auc:.4f}')
print(f'VGAE Test AP: {test_ap:.4f}')
print('VGAE Confusion matrix (test): \n', str(test_conf_mtx))


# plot losses

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.savefig('./plots/vgae_losses.png', dpi=300, bbox_inches='tight')
plt.clf()

# plot aucs and aps
plt.plot(aucs, label='auc')
plt.plot(aps, label='ap')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('ROC AUC and AP scores over epochs')
plt.savefig('./plots/vgae_scores.png', dpi=300, bbox_inches='tight')

# save test scores in csv
scores = pd.DataFrame({'auc': [test_auc], 'ap': [test_ap], 'conf_mtx': [test_conf_mtx]})
scores.to_csv('./scores/vgae_scores.csv', index=False)
scores.to_pickle('./scores/vgae_scores.pkl')
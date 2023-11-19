from data_prep import load_user_data
import networkx as nx
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix


# load data
user_id = 0
data_path = './data/facebook-processed/'
with open(data_path+'0.p', 'rb') as f:
    g_user = pickle.load(f)
adj_sparse = nx.to_scipy_sparse_array(g_user)
g_user_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = load_user_data(user_id, data_path)

# print info
print("Total nodes:", adj_sparse.shape[0])
print("Total edges:", int(adj_sparse.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print("Training edges (positive):", len(train_edges))
print("Training edges (negative):", len(train_edges_false))
print("Validation edges (positive):", len(val_edges))
print("Validation edges (negative):", len(val_edges_false))
print("Test edges (positive):", len(test_edges))
print("Test edges (negative):", len(test_edges_false))

# RUN NODE2VEC
import node2vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

print("\n\n")
print(40*"-"+" Running node2vec "+40*"-")
# node2vec settings
# NOTE: When p = q = 1, this is equivalent to DeepWalk

P = 1 # Return hyperparameter
Q = 1 # In-out hyperparameter
WINDOW_SIZE = 10 # Context size for optimization
NUM_WALKS = 10 # Number of walks per source
WALK_LENGTH = 80 # Length of walk per source
DIMENSIONS = 128 # Embedding dimension
DIRECTED = False # Graph directed/undirected
WORKERS = 1 # Num. parallel workers
ITER = 1 # SGD epochs

# Preprocessing, generate walks
g_n2v = node2vec.Graph(g_user_train, DIRECTED, P, Q) # create node2vec graph instance
g_n2v.preprocess_transition_probs()
walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
#walks = [map(str, walk) for walk in walks]

# Train skip-gram model
model = Word2Vec(walks, vector_size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS)#, iter=ITER)

# Store embeddings mapping
emb_mappings = model.wv

# Create node embeddings matrix (rows = nodes, columns = embedding features)
emb_list = []
for node_index in range(0, adj_sparse.shape[0]):
#    node_str = str(node_index)
    node_emb = emb_mappings[node_index]
    emb_list.append(node_emb)
emb_matrix = np.vstack(emb_list)

# Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
def get_edge_embeddings(edge_list):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = emb_matrix[node1]
        emb2 = emb_matrix[node2]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs

# Train-set edge embeddings
pos_train_edge_embs = get_edge_embeddings(train_edges)
neg_train_edge_embs = get_edge_embeddings(train_edges_false)
train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

# Create train-set edge labels: 1 = real edge, 0 = false edge
train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

# Val-set edge embeddings, labels
pos_val_edge_embs = get_edge_embeddings(val_edges)
neg_val_edge_embs = get_edge_embeddings(val_edges_false)
val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

# Test-set edge embeddings, labels
pos_test_edge_embs = get_edge_embeddings(test_edges)
neg_test_edge_embs = get_edge_embeddings(test_edges_false)
test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

# Create val-set edge labels: 1 = real edge, 0 = false edge
test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

# Train logistic regression classifier on train-set edge embeddings
edge_classifier = LogisticRegression(random_state=0)
edge_classifier.fit(train_edge_embs, train_edge_labels)


# Predicted edge scores: probability of being of class "1" (real edge)
val_preds_n2v = edge_classifier.predict_proba(val_edge_embs)[:, 1]
val_roc_n2v = roc_auc_score(val_edge_labels, val_preds_n2v)
val_ap_n2v = average_precision_score(val_edge_labels, val_preds_n2v)

# Predicted edge scores: probability of being of class "1" (real edge)
test_preds_n2v = edge_classifier.predict_proba(test_edge_embs)[:, 1]
test_roc_n2v = roc_auc_score(test_edge_labels, test_preds_n2v)
test_ap_n2v = average_precision_score(test_edge_labels, test_preds_n2v)

# Create confusion matrix
conf_mtx_n2v = confusion_matrix(test_edge_labels, edge_classifier.predict(test_edge_embs))


print('node2vec Validation ROC score: ', str(val_roc_n2v))
print('node2vec Validation AP score: ', str(val_ap_n2v))
print('node2vec Test ROC score: ', str(test_roc_n2v))
print('node2vec Test AP score: ', str(test_ap_n2v))
print('node2vec Confusion matrix (test): \n', str(conf_mtx_n2v))

# RUN SPECTRAL CLUSTERING
from sklearn.manifold import spectral_embedding

print("\n\n")
print(40*"-"+" Running spectral clustering "+40*"-")
def get_roc_score(edges_pos, edges_neg, embeddings):
    score_matrix = np.dot(embeddings, embeddings.T)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # create confusion matrix
    conf_mtx = confusion_matrix(labels_all, np.round(preds_all))

    return roc_score, ap_score, conf_mtx


# Get spectral embeddings (16-dim)
emb = spectral_embedding(adj_sparse, n_components=16, random_state=0)
# Calculate ROC AUC and Average Precision
roc_sc, ap_sc, conf_mtx_sc = get_roc_score(test_edges, test_edges_false, emb)


print('Spectral Clustering Test ROC score: ', str(roc_sc))
print('Spectral Clustering Test AP score: ', str(ap_sc))
print('Spectral Clustering Confusion matrix (test): \n', str(conf_mtx_sc))


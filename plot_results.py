import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# load data
n2v_scores = pd.read_pickle('./scores/node2vec_scores.pkl')
spectral_scores = pd.read_pickle('./scores/spectral_clustering_scores.pkl')
vgae_scores = pd.read_pickle('./scores/vgae_scores.pkl')
gnn_scores = pd.read_pickle('./scores/gnn_scores.pkl')

# add model name
n2v_scores['model'] = 'node2vec'
spectral_scores['model'] = 'spectral clustering'
vgae_scores['model'] = 'VGAE'
gnn_scores['model'] = 'GNN'

# combine scores
scores = pd.concat([n2v_scores,spectral_scores,vgae_scores,gnn_scores])

# plot ROC
plt.figure(figsize=(10,6))
plt.title('ROC AUC scores on test set')
plt.ylabel('ROC AUC score')
plt.ylim(0.5,1)
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.grid(axis='x')
plt.bar(scores['model'], scores['auc'])
plt.savefig('./plots/roc_auc_scores.png', dpi=300, bbox_inches='tight')

# plot AP
plt.figure(figsize=(10,6))
plt.title('Average precision scores on test set')
plt.ylabel('Average precision score')
plt.ylim(0.5,1)
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.grid(axis='x')
plt.bar(scores['model'], scores['ap'])
plt.savefig('./plots/ap_scores.png', dpi=300, bbox_inches='tight')
plt.clf()

# plot confusion matrices
for model in scores.model.unique():
    conf_mtx = scores[scores.model==model]['conf_mtx'].values[0]
    #conf_mtx = conf_mtx.replace('\n',',')
    #conf_mtx = np.array(ast.literal_eval(conf_mtx))
#    plt.figure(figsize=(10,6))
    classes = ['False edge', 'Real edge']
    sns.heatmap(conf_mtx, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
#    plt.imshow(conf_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion matrix for {model} on test set')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#    plt.xticks(np.arange(len(classes)) + 0.5, classes)
#    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)
    plt.xticks(rotation=0)
#    plt.imshow(conf_mtx, cmap='Blues')
#    plt.colorbar()
    plt.savefig(f'./plots/conf_mtx_{model}.png', dpi=300, bbox_inches='tight')
    plt.clf()

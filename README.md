# BMEVITMMA19---solo-que-homework
This is the repository of the homework of team solo que for the BMEVITMMA19 course.


## Team name
solo que

## Group members
Mihályi Balázs Márk - [J8KAR3]

## Project describtion
Friend recommendation with graph neural networks
The goal of this project is to develop a personalized friend recommendation system by using Graph Neural Networks (GNNs). You have to analyze data from Facebook, Google+, or Twitter to suggest meaningful connections based on user profiles and interactions. This project offers a hands-on opportunity to deepen your deep learning and network analysis skills.

## Table of contects
### data_acq.py
This script downloads the [facebook dataset](https://snap.stanford.edu/data/ego-Facebook.html) into the data folder, then unzips it there.

### data_prep.py
This script loads the data of all of the ego-networks from the facebook dataset, creates a networkx graph from the data, then creates training, validation, and test splits with negative edge sampling. It then saves the train networkx graph, and the numpy arrays of training, validation and test edges (positive and negative).

### run_baselines.py
This script loads one ego-network, and it's train, validation, and test sets. It then runs three baseline approaches, and prints their ROC and AP scores on the console as well as their confusion matrices. It also saves these metrics into pandas dataframes.
The baseline models are:
- Node2Vec (with Logistic Regression)
- Spectral Clustering
- GNN (simple GCN)

### GNN.py
This script contains the definition along with training and testing functions for a simple Graph Auto-Encoder (GAE) model. The defined GAE model has two Graph convolutional layers - one for encoding, and one for decoding.

## run_GNN.py
This scripts does the same things as _run_baselines.py_, but with a GAE model. It trains the model with the specified hyperparameters and prints the ROC AUC and AP scores, along with the confusion matrix. It also saves how the loss and the two scores changed throughout the training phase. It then also saves the three metrics (ROC AUC, AP, confusion matrix) of the model to a pandas dataframe.

## vgae.py
This sscript contains the definition of the Variational Graph Auto-Encoder (VGAE) class, its encoder's class, along with training and testing functions. It also contains the definition of a function that performs a simple hyperparameter grid search on a VGAE model given lists of parameters to try.

## run_vgae.py
This scripts does exactly the same things as _run_GNN.py_. The only addition is the hyperparameter grid search, which will occur if the _DO_HYPERPARAMETER_TUNING_ global variable at the beginning of the script is set to True.

## plot_results.py
This script reads all the dataframes saved during the training of the baseline and GNN methods and plots them. The ROC AUC and AP scores are plotted on bar plots, the confusion matrices are plotted independently.

## Dockerfile
This Dockerfile contains the definition of a docker image that's needed to run all the major scripts in this project.

## Related works
- [Node2Vec](https://arxiv.org/abs/1607.00653)
- [Friend Recommendation using GraphSAGE](https://medium.com/stanford-cs224w/friend-recommendation-using-graphsage-ffcda2aaf8d6)
- [GitHub Repository on link prediciton](https://github.com/lucashu1/link-prediction)
- [GitHub Repository on Graph Auto Encoders](https://github.com/tkipf/gae/tree/master)


## Usage
To get the train, validation, and test splits, just run these two scripts in order: 
- _data_acq.py_,
- _data_prep.py_.

To run baseline models, simply run the _run_baselines.py_ script.
To run the GNN-based methods, run the _run_GNN.py_ and _run_vgae.py_ scripts.
### Dockerization
To dockerize the project run the following command in the project folder (or where the Dockerfile is saved):
```
docker build -t dl-hw-soloque
```
This will build a docker image. To run the modelling sequence as a whole, run this command:
```
docker run -d dl-hw-soloque
```
The docker container will run the data acquisition and preparation scripts, then the modelling ones, and finally the plotting script.

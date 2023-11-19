# BMEVITMMA19---solo-que-homework
This is the repository of the homework of team solo que for the BMEVITMMA19 course.


## Team name
solo que

## Group members
Mihályi Balázs Márk - [REDACTED]

## Project describtion
Friend recommendation with graph neural networks
The goal of this project is to develop a personalized friend recommendation system by using Graph Neural Networks (GNNs). You have to analyze data from Facebook, Google+, or Twitter to suggest meaningful connections based on user profiles and interactions. This project offers a hands-on opportunity to deepen your deep learning and network analysis skills.

## Table of contects
### data_acq.py
This script downloads the [facebook dataset](https://snap.stanford.edu/data/ego-Facebook.html) into the data folder, then unzips it there.

### data_prep.py
This script loads the data of all of the ego-networks from the facebook dataset, creates a networkx graph from the data, then creates training, validation, and test splits with negative edge sampling. It then saves the train networkx graph, and the numpy arrays of training, validation and test edges (positive and negative).

## Related works
- [Node2Vec](https://arxiv.org/abs/1607.00653)
- [Friend Recommendation using GraphSAGE](https://medium.com/stanford-cs224w/friend-recommendation-using-graphsage-ffcda2aaf8d6)
- [GitHub Repository on link prediciton](https://github.com/lucashu1/link-prediction)
- [GitHub Repository on Graph Auto Encoders](https://github.com/tkipf/gae/tree/master)


## Usage
To get the train, validation, and test splits, just run these two scripts in order: 
- _data_acq.py_,
- _data_prep.py_.

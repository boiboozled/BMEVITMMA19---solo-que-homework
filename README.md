# BMEVITMMA19---solo-que-homework
This is the repository of the homework of team solo que for the BMEVITMMA19 course.


## Team name
solo que

## Group members
Mihályi Balázs Márk - J8KAR3

## Project describtion
Friend recommendation with graph neural networks
The goal of this project is to develop a personalized friend recommendation system by using Graph Neural Networks (GNNs). You have to analyze data from Facebook, Google+, or Twitter to suggest meaningful connections based on user profiles and interactions. This project offers a hands-on opportunity to deepen your deep learning and network analysis skills.

## Table of contects
### data_acq.py
This script downloads the [facebook dataset](https://snap.stanford.edu/data/ego-Facebook.html) into the data folder, then unzips it there. Later versions will take an argument, that will decide which social dataset to download. Other options will be the [twitter dataset](), and the [Google+]() dataset.

### data_prep.py
This script loads the data of one ego-network from the facebook dataset, creates a dgl graph from the data, then splits that graph into two train, validation, and test graphs. One with positive edges and one with negative ones for all three. Finally, it saves the whole graph, and it's splits. For now, the ego-network id, and the load and save paths are not given as arguments, but given in the script. This will change in the future to make containerization easier.

## Related works
- [Node2Vec](https://arxiv.org/abs/1607.00653)
- [Friend Recommendation using GraphSAGE](https://medium.com/stanford-cs224w/friend-recommendation-using-graphsage-ffcda2aaf8d6)


## Usage
To get the train, validation, and test graph sets, you just need to run the two scripts in order. First _data_acq.py_, _then data_prep.py_.

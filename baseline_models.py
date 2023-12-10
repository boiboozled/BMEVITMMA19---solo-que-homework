


class Node2Vec:
    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, workers=1, p=1, q=1, weight_key=None):
        """"""
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers

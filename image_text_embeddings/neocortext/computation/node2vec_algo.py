from node2vec import Node2Vec
import numpy as np
import networkx as nx


def node2vec_algo(data):
    # Transform an edge list into embeddings with Node2Vec
    G = nx.from_pandas_edgelist(data, "source", "target", "weight")
    node2vec = Node2Vec(G, dimensions=20, walk_length=20, num_walks=30, workers=6)
    model = node2vec.fit(window=30, min_count=1)
    nodes = list(map(str, G.nodes()))
    embeddings = np.array([model.wv[x] for x in nodes])

    return nodes, embeddings

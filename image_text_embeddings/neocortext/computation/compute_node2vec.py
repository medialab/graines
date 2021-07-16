from .node2vec_algo import node2vec_algo
from .reduction import make_reduction
from  .cluster import make_clustering
import pickle
import os

def compute(edge_list, path):
    # Node2Vec embedding
    if not os.path.isfile(path + "embeddings_node2vec.pkl"):
        print('starting embeddings...')


        nodes, embeddings = node2vec_algo(edge_list)

        with open(path + "embeddings_node2vec.pkl", "wb") as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path + "embeddings_nodes.pkl", "wb") as f:
            pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        print("The embeddings have already been computed...")

        with open(path + "embeddings_node2vec.pkl", "rb") as f:
            embeddings = pickle.load(f)

        with open(path + "embeddings_nodes.pkl", "rb") as f:
            nodes = pickle.load(f)


    # 2D dimensionility dimensions
    if not os.path.isfile(path + "reduction.pkl"):
        print("Reduction Algorithms...")
        reduction = make_reduction(embeddings, save = True)

    else:
        print("The reduction has already been operated...")

        with open(path + "reduction.pkl", "rb") as f:
            reduction = pickle.load(f)

    # Cluster embeddings
    if not os.path.isfile(path + "clusters.pkl"):
        print("Clustering Algorithms...")
        clusters = make_clustering(embeddings, save = True)

    else:
        print("The clustering has already been operated...")

        with open(path + "clusters.pkl", "rb") as f:
            clusters = pickle.load(f)

    return reduction, clusters, nodes
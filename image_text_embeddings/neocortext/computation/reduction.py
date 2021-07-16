from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
import umap
from node2vec import Node2Vec
import numpy as np
from sklearn.manifold import TSNE
import pickle 

def make_reduction(embeddings: np.array, save =True) -> dict:
    '''Reduce dimensions with different algorithms
    Outputs the 2D Embeddings per embedding style'''
    reduction_algorithms = {
        "tsne": TSNE(n_components=2),
        "pca": PCA(n_components=2),
        "kpca": KernelPCA(n_components=2),
        "svd": TruncatedSVD(n_components=2),
        "umap": umap.UMAP(n_components=2, n_neighbors=15)
    }

    reduction_embeddings = {}
    for key, value in reduction_algorithms.items():
        res = reduction_algorithms[key].fit_transform(embeddings)
        reduction_embeddings[key] = res

    if save is True:
        with open('reduction.pickle', 'wb') as f:
	        pickle.dump(reduction_embeddings, f)

    return reduction_embeddings

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import hdbscan
import umap
import numpy as np
import pickle

umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, 
                            metric='cosine')

hdbscan_clustering = hdbscan.HDBSCAN(min_cluster_size=12,
                          metric='euclidean',                      
                          cluster_selection_method='eom')


def make_clustering(embeddings: np.array, save = True)-> dict:
    # Make clusters
    clustering_algorithms = {
        "km": KMeans(n_clusters=15),
        "hdbscan": make_pipeline(umap_embeddings, hdbscan_clustering)
    }

    clustering_embeddings = {}
    for key, value in clustering_algorithms.items():
        res = clustering_algorithms[key].fit_predict(embeddings)
        clustering_embeddings[key] = res

    if save is True:
        with open('clustering.pickle', 'wb') as f:
            pickle.dump(clustering_embeddings, f)

    return clustering_embeddings
 
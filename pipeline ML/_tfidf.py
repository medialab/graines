from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd


def tfidf_pipeline(X: list) -> np.array:
    """This function embed text based on tf-idf functions

    Args:
        corpus ([type]): list of texts to embed

    Returns:
         np.array: the tfidf embeddings
    """

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    svd = TruncatedSVD(n_components=512, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    emb = lsa.fit_transform(X)
    return emb


if __name__ == "__main__":
    data = pd.read_csv("data/data_ready.csv", index_col=[0])
    data_tfidf = data[["screen_name", "description", "name"]]
    data_tfidf = data_tfidf.fillna(" ")
    data_tfidf = (
        data_tfidf["screen_name"]
        + "//"
        + data_tfidf["description"]
        + "//"
        + data_tfidf["name"]
    )
    X = list(data_tfidf.values)
    emb = tfidf_pipeline(X)
    np.save("embeddings/tfidf.npy", emb)

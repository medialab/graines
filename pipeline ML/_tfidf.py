from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd


def tfidf_pipeline(X: list, n_components: int) -> np.array:
    """This function embed text based on tf-idf functions

    Args:
        corpus ([list]): list of texts to embed

    Returns:
         np.array: the tfidf embeddings
    """

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    emb = lsa.fit_transform(X)
    return emb


def tfidf_pipeline_final_predict(X: list, X_predict: list, n_components: int) -> np.array:
    """This function embed text based on tf-idf functions

    Args:
        corpus ([list]): list of texts to embed

    Returns:
         np.array: the tfidf embeddings
    """

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X_predict = vectorizer.transform(X_predict)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    emb = lsa.fit_transform(X_predict)
    return emb


def load_data(data):
    data_tfidf = data[["screen_name", "description", "name", "location"]]
    data_tfidf = data_tfidf.fillna(" ")
    data_tfidf = (
        data_tfidf["screen_name"]
        + "//"
        + data_tfidf["description"]
        + "//"
        + data_tfidf["name"]
        + "//"
        + data_tfidf["location"]
    )
    X = list(data_tfidf.values)
    return X


if __name__ == "__main__":
    n_components = 10
    annotated_data = pd.read_csv("data/data_ready.csv", index_col=[0])
    X = load_data(annotated_data)
    emb = tfidf_pipeline(X, n_components)
    np.save("embeddings/tfidf.npy", emb)

    all_data = pd.read_csv("data/followers_metadata_version_2021_10_19.csv")
    X_predict = load_data(all_data)
    emb = tfidf_pipeline_final_predict(X, X_predict, n_components)
    np.save("embeddings/tfidf_final_predict.npy", emb)

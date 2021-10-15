from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd


def bayesian_pipeline(
    X: list, classifier="MultinomialNB", seeds=[3, 7, 8, 9, 10, 11]
) -> np.array:
    """Produce an embedding based on the probability of being a seed computed with a Bayesian classifier

    Args:
        X (list): list of texts to embed
        classifier (str): 'MultinomialNB' or 'GaussianNB'
        seeds (list):

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
    data_graines = data["graines_in_friends"].str.split("|")
    X = list(data_graines.values)
    emb = bayesian_pipeline(X)
    np.save("embeddings/tfidf.npy", emb)

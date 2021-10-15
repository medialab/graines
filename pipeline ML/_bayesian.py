from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import scipy


def bayesian_pipeline(
    all_followers: list,
    X: list,
    y: list,
    classifier="MultinomialNB",
    seeds=[12, 13, 14, 15],
):
    """Produce an embedding based on the probability of being a seed computed with a Bayesian classifier
    Args:
        all_followers: list of graines in friends for the entire dataset
        X (list): llist of graines in friends for the annotated dataset
        y (list): target values
        classifier (str): 'MultinomialNB' or 'GaussianNB'
        seeds (list):

    """

    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_followers)
    X = vectorizer.transform(X)
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        if classifier == "MultinomialNB":
            clf = naive_bayes.MultinomialNB()
        elif classifier == "GaussianNB":
            clf = naive_bayes.GaussianNB()
            if scipy.sparse.issparse(X):
                X = X.todense()
                X_train = X_train.todense()
        else:
            print("Error, choose one of 'MultinomialNB', 'GaussianNB")
        clf.fit(X_train, y_train)
        emb = clf.predict_log_proba(X)
        np.save("embeddings/bayesian_proba_{}_{}.npy".format(classifier, seed), emb)


if __name__ == "__main__":
    all_followers = pd.read_csv("data/followers_metadata_version_2021_09_21.csv")
    data = pd.read_csv("data/data_ready.csv", index_col=[0])
    all_followers_graines = all_followers["graines_in_friends"].str.replace("|", " ").values
    data_graines = data["graines_in_friends"].apply(lambda x: " ".join(eval(x))).values
    all_followers_friends = list(all_followers_graines)
    X = list(data_graines)
    y = list(data["label"].values)
    bayesian_pipeline(all_followers_friends, X, y, classifier="MultinomialNB")

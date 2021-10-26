from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy
from config import seeds


def bayesian_pipeline(
    all_friends: list,
    annotated_friends: list,
    all_followers: list,
    annotated_followers: list,
    y: list,
    classifier="MultinomialNB",
    seeds=[12, 13, 14, 15],
):
    """Produce an embedding based on the probability of being a seed computed with a Bayesian classifier
    Args:
        all_followers: list of graines in friends for the entire dataset
        X (list): list of graines in friends for the annotated dataset
        y (list): target values
        classifier (str): 'MultinomialNB' or 'GaussianNB'
        seeds (list):

    Returns:


    """

    vectorizer_friends, vectorizer_followers = TfidfVectorizer(), TfidfVectorizer()
    vectorizer_friends.fit(all_friends)
    vectorizer_followers.fit(all_followers)
    vectorized_friends = vectorizer_friends.transform(annotated_friends)
    vectorized_followers = vectorizer_followers.transform(annotated_followers)
    X = scipy.sparse.hstack([vectorized_friends, vectorized_followers])
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
    all_data = pd.read_csv("data/followers_metadata_version_2021_10_19.csv")
    annotated_data = pd.read_csv("data/data_ready.csv", index_col=[0])
    all_friends = all_data["graines_in_friends"].str.replace("|", " ", regex=False).values
    annotated_friends = annotated_data["graines_in_friends"].str.replace("|", " ", regex=False).values
    all_followers = all_data[all_data.graines_in_followers.notna()]["graines_in_followers"].str.replace("|", " ").values
    annotated_followers = annotated_data["graines_in_followers"].fillna("").str.replace("|", " ").values
    y = annotated_data["label"].values
    bayesian_pipeline(all_friends,
                      annotated_friends,
                      all_followers,
                      annotated_followers,
                      y,
                      classifier="MultinomialNB",
                      seeds=seeds
                      )

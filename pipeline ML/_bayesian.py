from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv
import scipy
from collections import defaultdict
from config import seeds, objective
import array
from tqdm import tqdm

BATCH_SIZE = 100000


def count_mentions(vocabulary, all_screen_names, mentions_text):
   vectorizer = CountVectorizer(vocabulary=vocabulary)
   X = vectorizer.fit_transform((" ".join(v*(k+" ") for k,v in mentions_text[screen_name].items()) for screen_name in all_screen_names))
   return X


def fit_classifier(X, classifier, X_train, y_train):
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
    return clf, X, X_train, y_train


def bayesian_pipeline(
    all_screen_names: list,
    annottated_indices: list,
    all_friends: list,
    annotated_friends: list,
    all_followers: list,
    annotated_followers: list,
    y: list,
    classifier="MultinomialNB",
    seeds=[12, 13, 14, 15],
    objective="report"
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
    # create a vocabulary of graines' names
    with open("data/VF-Carte Raison - Corpus-final.csv") as f:
        reader = csv.reader(f)
        next(reader)
        vocab = set(row[1].lower() for row in reader)
    vocabulary = {k: v for v, k in enumerate(sorted(vocab))}

    mentions_dict = defaultdict(dict)
    mentioned_dict = defaultdict(dict)
    with open("data/tweets_mentionnant_les_graines_depuis_20200101.csv") as f:
        reader = csv.reader(f)
        headers = next(reader)
        mentions_pos = headers.index("mentioned_names")
        screen_name_pos = headers.index("user_screen_name")
        # read all tweets
        print("\n Find mentions in tweets\n ")
        for row in tqdm(reader, total=4749615):
            if row[mentions_pos]:
                for mention in row[mentions_pos].split("|"):
                    # if a graine is mentioned in a tweet, add it to the mentions_dict
                    if mention in vocabulary:
                        mentions_user = mentions_dict[row[screen_name_pos]]
                        if mention not in mentions_user:
                            mentions_user[mention] = 1
                        else:
                            mentions_user[mention] += 1
                # if someone is mentioned by a graine, add it to the mentioned_dict
                if row[screen_name_pos] in vocabulary:
                    for mention in row[mentions_pos].split("|"):
                        mentioned_user = mentioned_dict[mention]
                        if row[screen_name_pos] not in mentioned_user:
                            mentioned_user[row[screen_name_pos]] = 1
                        else:
                            mentioned_user[row[screen_name_pos]] += 1
    count_matrix_mentions = scipy.sparse.csr_matrix((0, len(vocabulary)))
    count_matrix_mentioned = scipy.sparse.csr_matrix((0, len(vocabulary)))
    print("\n Create mentions count matrix\n ")
    for i in tqdm(range(0, len(all_screen_names), BATCH_SIZE)):
        count_matrix_mentions = scipy.sparse.vstack(
            [count_matrix_mentions, count_mentions(vocabulary, all_screen_names[i:i+BATCH_SIZE], mentions_dict)]
        )
        count_matrix_mentioned = scipy.sparse.vstack(
            [count_matrix_mentioned, count_mentions(vocabulary, all_screen_names[i:i+BATCH_SIZE], mentioned_dict)]
        )
    transformer_mentions, transformer_mentioned = TfidfTransformer(), TfidfTransformer()
    vectorized_mentions = transformer_mentions.fit_transform(count_matrix_mentions)
    vectorized_mentioned = transformer_mentioned.fit_transform(count_matrix_mentioned)
    vectorizer_friends, vectorizer_followers = TfidfVectorizer(), TfidfVectorizer()
    vectorizer_friends.fit(all_friends)
    vectorizer_followers.fit(all_followers)
    vectorized_friends = vectorizer_friends.transform(annotated_friends)
    vectorized_followers = vectorizer_followers.transform(annotated_followers)
    X = scipy.sparse.hstack([
        vectorized_friends,
        vectorized_followers,
        vectorized_mentions[annotated_indices],
        vectorized_mentioned[annotated_indices]
    ])
    if objective == "report":
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )
            clf, X, X_train, y_train = fit_classifier(X, classifier, X_train, y_train)
            emb = clf.predict_log_proba(X)
            np.save("embeddings/bayesian_proba_{}_{}.npy".format(classifier, seed), emb)
    elif objective == "classification":

        # save an embedding of the training set
        X_train, y_train = X, y
        clf, X, X_train, y_train = fit_classifier(X, classifier, X_train, y_train)
        emb = clf.predict_log_proba(X)
        np.save("embeddings/bayesian_proba_{}_final_train.npy".format(classifier), emb)

        # save an embedding of all followers
        vectorized_friends = vectorizer_friends.transform(all_friends)
        vectorized_followers = vectorizer_followers.transform(all_followers)
        X = scipy.sparse.hstack([
            vectorized_friends,
            vectorized_followers,
            vectorized_mentions,
            vectorized_mentioned
        ])
        emb = clf.predict_log_proba(X)
        np.save("embeddings/bayesian_proba_{}_final_predict.npy".format(classifier), emb)


if __name__ == "__main__":
    all_data = pd.read_csv("data/followers_metadata_version_2021_10_19.csv")
    annotated_data = pd.read_csv("data/data_ready.csv", index_col=[0])
    all_friends = all_data["graines_in_friends"].str.replace("|", " ", regex=False).values
    annotated_friends = annotated_data["graines_in_friends"].str.replace("|", " ", regex=False).values
    all_data["graines_in_followers"] = all_data["graines_in_followers"].fillna(" ")
    all_followers = all_data["graines_in_followers"].str.replace("|", " ").values
    annotated_followers = annotated_data["graines_in_followers"].fillna("").str.replace("|", " ").values
    all_screen_names = all_data["screen_name"].values
    annotated_screen_names = set(annotated_data["screen_name"].values)
    annotated_indices = all_data[all_data.screen_name.isin(annotated_screen_names)].index
    y = annotated_data["label"].values
    bayesian_pipeline(all_screen_names,
                      annotated_indices,
                      all_friends,
                      annotated_friends,
                      all_followers,
                      annotated_followers,
                      y,
                      classifier="MultinomialNB",
                      seeds=seeds,
                      objective=objective
                      )

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, euclidean_distances
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from config import seeds, type_of_model, objective
from tqdm import tqdm

BATCH_SIZE = 1000

def triangular_kernel(X, Y):
    return 1 - abs(euclidean_distances(X, Y))


classifiers = {
    "SVM_triangular_kernel": SVC(kernel=triangular_kernel, C=3),
    "SVM_RBF_kernel": SVC(),
}


def add_bayes(X, type_of_algo, seed=0):
    bayes_file_path = "embeddings/bayesian_proba_MultinomialNB_{}.npy".format(seed)
    if type_of_algo == ["bayesian"]:
        X = np.load(bayes_file_path)
    elif "bayesian" in type_of_algo and os.path.isfile(bayes_file_path):
        X_bayes = np.load(bayes_file_path)
        X = np.concatenate((X, X_bayes), axis=1)
    return X


def train_classifier(X_train, y_train, classifier_model):

    # Chose classifier
    clf = classifiers[classifier_model]

    # Fit
    clf.fit(X_train, y_train)
    return clf


def classifier_pipeline(
    type_of_algo: list,
    X: np.array,
    y: np.array,
    classifier_model: str = "SVM_triangular_kernel",
    seeds: list = [12, 13, 14, 15],
    objective: str = "report",
    X_predict: np.array = None
) -> pd.DataFrame:

    """This function trains a triangular classifier and outputs a report or the results of the prediction

    Args:
        labels_file (str): the path of the file with the labels
        classifier_model (str, optional): The triangulat model to chsoe. Defaults to "SVM_triangular_kernel".
        seeds (list, optional): a list of seed values . Defaults to [2, 3, 4, 5, 6].
        report_file_path (str, optional): path of the report file if it exists . Defaults to None.
        embedding_file (str, optional): path of the embeddings file as a numpy array. Defaults to "bert_fitted_on_descriptions.npy".
        objective (str, optional): . Defaults to "report". There are two main objectives:
            - create a report
            - predict

    Returns:
        if objective = 'report', it outputs a DataFrame with the result of the report or reads a previous report and
        append the new results
        if objective  = 'classification', it outputs the initial file with an extra column called 'predicted'
        with the results of the algorithm
    """

    if objective == "report":  # Only compute the report
        display_df = pd.DataFrame()
        for seed in seeds:
            X = add_bayes(X, type_of_algo, seed=seed)

            # Train Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=seed
            )

            X_test = X_test
            y_test = np.array(y_test)

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            clf = train_classifier(X_train, y_train, classifier_model)

            # Predict
            y_pred = clf.predict(X_test)

            # Evaluate the results
            current_results = dict()

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, pos_label=1, average="binary"
            )

            current_results["precision"] = precision
            current_results["recall"] = recall
            current_results["f1"] = f1
            current_results["seed"] = seed
            current_results["datetime"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            current_results = pd.DataFrame(current_results, index=[0])
            current_results["type_of_algo"] = "|".join(type_of_algo)

            display_df = display_df.append(current_results)
            display_df = display_df.reset_index(drop=True)

        return display_df

    # Only predict the values
    elif objective == "classification":
        X = add_bayes(X, type_of_algo, seed="final_train")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_predict = scaler.fit_transform(X_predict)
        clf = train_classifier(X, y, classifier_model)

        y_pred = clf.predict(X_predict[:BATCH_SIZE])
        for i in tqdm(range(BATCH_SIZE, X_predict.shape[0], BATCH_SIZE)):
            y_pred = np.concatenate([y_pred, clf.predict(X_predict[i:i+BATCH_SIZE])], axis=0)
        return y_pred

    else:
        print("Chose a right objective")


if __name__ == "__main__":

    # Load Computed embeddings
    dict_emb = {
    "tfidf": "tfidf.npy",
    "bert": "bert.npy",
    "image": "full_profile_pictures.npy",
    "features": "features.npy",
    "topology": "topo.npy",
    "doc2vec": "doc2vecs.npy",
    "activity": "tweets_content.npy"
    }

    data = pd.read_csv("data/data_ready.csv")
    y = list(data["label"])

    full_report = pd.read_csv("report.csv", index_col=[0])
    mean_report = pd.read_csv("mean_report.csv")

    if type_of_model != ["bayesian"]:
        X = np.concatenate(
            [np.load
             (os.path.join("embeddings", dict_emb[model]), allow_pickle=True
              ) for model in type_of_model if model != "bayesian"
             ],
            axis=1
        )
    else:
        X = None

    if objective == "report":
        output = classifier_pipeline(
            type_of_algo=type_of_model,
            X=X,
            y=y,
            seeds=seeds,
            objective=objective,
        )
        full_report = full_report.append(output)
        full_report.to_csv("report.csv")

        mean = {}
        for metric in ["precision", "recall", "f1"]:
            mean[metric] = "{}??{}".format(
                output[[metric]].mean().round(2).values[0],
                output[[metric]].std().round(2).values[0]
            )
        for info in ["datetime", "type_of_algo"]:
            mean[info] = output.iloc[0][info]

        mean = pd.DataFrame(mean, index=[0])

        mean_report = mean_report.append(mean)
        mean_report.to_csv("mean_report.csv", index=False)

        print(output[["precision", "recall", "f1", "type_of_algo"]])
        print("Average on {} runs:".format(output.shape[0]))
        print(mean[["precision", "recall", "f1", "type_of_algo"]])

    elif objective == "classification":
        dict_emb.update({'bayesian': 'bayesian_proba_MultinomialNB.npy'})
        X_predict = np.concatenate(
            [np.load(
                os.path.join("embeddings", dict_emb[model].replace(".npy", "_final_predict.npy"))
            ) for model in type_of_model],
            axis=1
        )
        output = classifier_pipeline(
            type_of_algo=type_of_model,
            X=X,
            y=y,
            seeds=[],
            objective=objective,
            X_predict=X_predict
        )
        all_data = pd.read_csv("data/followers_metadata_version_2021_10_19.csv")
        all_data["prediction"] = output
        all_data.to_csv("prediction_{}.csv".format("-".join(type_of_model)), index=False)
        print("Saved predictions file to prediction_{}.csv".format("-".join(type_of_model)))
        nb_positives = all_data[all_data.prediction == 1].shape[0]
        print("Nb of accounts predicted as galaxy members: {} ({}% of all accounts)".format(
            nb_positives, round(nb_positives * 100 / all_data.shape[0], 2)
        ))

    else:
        print("objective parameter should be either report or classification")



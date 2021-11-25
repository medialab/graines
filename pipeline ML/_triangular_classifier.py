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


def triangular_kernel(X, Y):
    return 1 - abs(euclidean_distances(X, Y))


classifiers = {
    "SVM_triangular_kernel": SVC(kernel=triangular_kernel, C=3),
    "SVM_RBF_kernel": SVC(),
}


def add_bayes(X, type_of_algo, seed):
    bayes_file_path = "embeddings/bayesian_proba_MultinomialNB_{}.npy".format(seed)
    if type_of_algo == ["bayesian"]:
        X = np.load(bayes_file_path)
    elif "bayesian" in type_of_algo and os.path.isfile(bayes_file_path):
        X_bayes = np.load(bayes_file_path)
        X = np.concatenate((X, X_bayes), axis=1)
    return X


def train_classifier(X, y, classifier_model, type_of_algo, seed):
    scaler = StandardScaler()
    # Chose classifier
    clf = classifiers[classifier_model]

    X = add_bayes(X, type_of_algo, seed)

    # Train Test Split
    X_train, X_test, y_train, y_test, flamboyant_train, flamboyant_test = train_test_split(
        X, y, not_flamboyant, test_size=0.3, random_state=seed
    )

    X_test = X_test[flamboyant_test]
    y_test = np.array(y_test)[flamboyant_test]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Fit
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def classifier_pipeline(
    type_of_algo: list,
    X: np.array,
    y: np.array,
    not_flamboyant: np.array,
    classifier_model: str = "SVM_triangular_kernel",
    seeds: list = [12, 13, 14, 15],
    objective: str = "report",
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
            clf, X_test, y_test = train_classifier(X, y, classifier_model, type_of_algo, seed)

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
        seed = seeds[1]
        scaler = StandardScaler()
        clf, _, _ = train_classifier(X, y, classifier_model, type_of_algo, seed)
        X = add_bayes(X, type_of_algo, seed)
        X = scaler.fit_transform(X)
        y_pred = clf.predict(X)
        return y_pred

    else:
        print("Chose a right objective")


if __name__ == "__main__":

    # Load Computed embeddings
    X_tfidf = np.load("embeddings/tfidf.npy", allow_pickle=True)
    X_bert = np.load("embeddings/bert.npy", allow_pickle=True)
    X_image = np.load("embeddings/full_profile_pictures.npy", allow_pickle=True)
    X_features = np.load("embeddings/features.npy", allow_pickle=True)
    X_topo = np.load("embeddings/topo.npy", allow_pickle=True)
    X_d2v = np.load("embeddings/doc2vecs.npy", allow_pickle=True)

    dict_emb = {
        "tfidf": X_tfidf,
        "bert": X_bert,
        "images": X_image,
        "features": X_features,
        "topology": X_topo,
        "doc2vec": X_d2v
    }

    '''def make_classification(emb_type: list, dict_emb: dict, y: list):
        """This functions takes as an input the desired embeddings and outputs
        the results of the classification as a report and as a prediction

        Args:
            emb_type (list): example: ['bert', 'tfidf]
            dict_emb (dict): the table of name and related embeddings
            y (list): the labels

        Returns:
            [type]: outputs a report and the predictions of the algorithm
        """

        X = np.array(object)
        for x in emb_type:
            emb = dict_emb[x]
            X = np.concatenate((X, emb), axis=1)
            full_report = pd.read_csv("report.csv", index_col=[0])
            report = classifier_pipeline(
                type_of_algo="-".join(emb_type),
                X=X,
                y=y,
                seeds=[1, 2, 3, 4, 5, 6],
                objective="report",
            )
            full_report = full_report.append(report)
            full_report.to_csv("report.csv")
            y_pred = classifier_pipeline(
                type_of_algo="-".join(emb_type),
                X=X,
                y=y,
                seeds=[1, 2, 3, 4, 5, 6],
                objective="classification",
            )
        return report, y_pred'''

    data = pd.read_csv("data/data_ready.csv")
    y = list(data["label"])
    not_flamboyant = list(data["flamboyant_seed"] == 0)

    full_report = pd.read_csv("report.csv", index_col=[0])
    mean_report = pd.read_csv("mean_report.csv")

    if type_of_model != ["bayesian"]:
        print([dict_emb[model].shape for model in type_of_model if model != "bayesian"])
        X = np.concatenate([dict_emb[model] for model in type_of_model if model != "bayesian"], axis=1)
    else:
        X = None
    output = classifier_pipeline(
        type_of_algo=type_of_model,
        X=X,
        y=y,
        not_flamboyant=not_flamboyant,
        seeds=seeds,
        objective=objective,
    )

    if objective == "report":
        full_report = full_report.append(output)
        full_report.to_csv("report.csv")

        mean = {}
        for metric in ["precision", "recall", "f1"]:
            mean[metric] = "{}±{}".format(
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

    else:
        data["prediction"] = output
        data.to_csv("prediction_{}.csv".format("-".join(type_of_model)), index=False)
        nb_positives = data[data.prediction == 1].shape[0]
        nb_false_positives = data[(data.prediction == 1) & (data.label == 0)].shape[0]
        nb_false_negatives = data[(data.prediction == 0) & (data.label == 1)].shape[0]
        print("Nb of accounts predicted as galaxy members: {} ({}% of all accounts)".format(
            nb_positives, round(nb_positives * 100 / data.shape[0], 2)
        ))
        print("Nb of accounts falsely predicted as galaxy members (false positives): {} ({}% of all accounts)".format(
            nb_false_positives, round(nb_false_positives * 100 / data.shape[0], 2)
        ))
        print("Nb of missed galaxy members (false negatives): {} ({}% of all accounts)".format(
        nb_false_negatives, round(nb_false_negatives * 100 / data.shape[0], 2)
        ))



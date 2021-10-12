import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, euclidean_distances
from sklearn.svm import SVC


def triangular_kernel(X, Y):
    return 1 - abs(euclidean_distances(X, Y))


classifiers = {
    "SVM_triangular_kernel": SVC(kernel=triangular_kernel, C=3),
    "SVM_RBF_kernel": SVC(),
}


def classifier_pipeline(
    X: np.array,
    y: np.array,
    classifier_model: str = "SVM_triangular_kernel",
    seeds: list = [2, 3, 4, 5],
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

    # Chose classifier
    clf = classifiers[classifier_model]

    if objective == "report":  # Only compute the report

        display_df = pd.DataFrame()
        for seed in seeds:

            # Train Test Split and Predict
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.6, random_state=seed
            )

            # Predict
            clf.fit(X_train, y_train)
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

            display_df = display_df.append(current_results)
            display_df = display_df.reset_index(drop=True)

        return display_df

    # Only predict the values
    elif objective == "classification":
        clf.fit(X, y)
        y_pred = clf.predict(X)
        return y_pred
    else:
        print("Chose a right objective")


if __name__ == "__main__":

    # Load Data
    X = np.load("data/final_X.npy", allow_pickle=True)
    y = np.load("data/final_y.npy", allow_pickle=True)
    report = classifier_pipeline(X, y, seeds=[3, 7, 8, 9, 10, 11])
    report.to_csv("data/report.csv")

    print(report)
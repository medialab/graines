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


def evaluate(y_test: pd.DataFrame, y_pred: pd.DataFrame, seed: int) -> pd.DataFrame:
    params = pd.DataFrame()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, pos_label=1, average="binary"
    )

    params["p"] = precision
    params["r"] = recall
    params["f1"] = f1
    params["seed"] = seed
    params["datetime"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    return params


def test_params(
    labels_file: str,
    classifier_model: str = "SVM_triangular_kernel",
    seeds: list = [2, 3, 4, 5, 6],
    report_file_path: str = None,
    embedding_file="bert_fitted_on_descriptions.npy",
    objective: str = "report",
):

    # Load the embeddings (as numpy)
    X = np.load(embedding_file)

    # read Data
    data = pd.read_csv(labels_file)
    mask = data["label"].notna()
    y = data[mask]["label"].astype(int).values

    # Chose classifier
    clf = classifiers[classifier_model]

    if objective == "report":  # Only compute the report

        display_df = pd.DataFrame()
        for seed in seeds.pop(seeds):

            # Train Test Split and Predict
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=seed
            )

            # Predict
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Evaluate the results
            current_results = evaluate(y_test, y_pred, seed)
            display_df = display_df.append(current_results)
            try:
                results = pd.read_csv(report_file_path)
            except FileNotFoundError:
                results = pd.DataFrame()
            current_results = results.append(display_df, ignore_index=True)
            current_results.to_csv(report_file_path, index=False)

    # Only predic the values
    elif objective == "classification":
        clf.fit(X[mask], y)
        y_pred = clf.predict(X)
        data["predicted"] = y_pred
        data.to_csv("predicted_data.csv", index=False)
    else:
        print("Chose a right objective")

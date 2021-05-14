from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import argparse
import logging
import warnings
from create_ground_truth import LABEL_FILE_NAME
from classifiers import classifiers
from datetime import datetime
import getpass

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
report_fields = ["model", "classifier", "f1", "p", "r", "datetime", "author", "labels", "seed"]
report_file = "results_binary_classif.csv"
username = getpass.getuser()

parser = argparse.ArgumentParser()


parser.add_argument('--model',
                    required=True,
                    help="""
                    Name of the user embedding
                    """
                    )
parser.add_argument('--classifier',
                    required=False,
                    default="SVM_triangular_kernel",
                    choices=classifiers.keys(),
                    help="""
                    Name of the classifier
                    """
                    )
parser.add_argument('--report',
                    action='store_true',
                    help="""
                    Whether you want to want to save the precision, recall, f1 score in the result file
                    """

                    )
parser.add_argument('--objective',
                    required=False,
                    choices=["test", "classification"],
                    default="test",
                    help="""
                    Whether you want to test your embedding or do the final classification of users
                    """
                    )
parser.add_argument('--labels',
                    required=False,
                    default=LABEL_FILE_NAME,
                    help="""
                    Path to the csv file containing the labels of users in the "label" column
                    """
                    )


def main(args):
    test_params(**args, seeds=[628, 11, 1008, 2993, 559])


def test_params(**params):
    X = np.load(params["model"] + ".npy")
    display_df = pd.DataFrame()
    data = pd.read_csv(params["labels"])
    mask = data.label.notna()
    y = data[mask].label.astype(int).values
    clf = classifiers[params["classifier"]]
    logging.info("Start classification. This may take some time...")

    if params["objective"] == "test":
        for seed in params.pop("seeds"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average="binary")
            params["p"] = precision
            params["r"] = recall
            params["f1"] = f1
            params["seed"] = seed
            params["datetime"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            params["author"] = username
            current_results = pd.DataFrame(params, index=[0])[report_fields].round(4)
            display_df = display_df.append(current_results)
            if params["report"]:
                try:
                    results = pd.read_csv(report_file)
                except FileNotFoundError:
                    results = pd.DataFrame()
                current_results = results.append(current_results, ignore_index=True)
                current_results.to_csv(report_file, index=False)
        logging.info(
            "average F1 on {} runs: {}±{}".format(
                display_df.shape[0],
                display_df[["f1"]].mean().round(2).values[0],
                display_df[["f1"]].std().round(2).values[0])
        )
        if params["report"]:
            logging.info("Saved report to {}".format(report_file))

    elif params["objective"] == "classification":
        clf.fit(X[mask], y)
        y_pred = clf.predict(X)
        data["predicted_as_galaxy_member"] = y_pred
        data.to_csv(params["labels"], index=False)
        logging.info("{} galaxy members found".format(len(data[data["predicted_as_galaxy_member"] == 1])))
        logging.info("Predictions saved to {}".format(params["labels"]))


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)



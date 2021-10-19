import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np


def features_emb(data: pd.DataFrame, features: list) -> np.array:
    """Extracts the features & normalize them and output the embeddings

    Args:
        data (pd.DataFrame): [description]
        features (list): [description]

    Returns:
        [np.array]: array of the embeddings
    """

    data_feat = data[features]
    data_feat = data_feat.fillna(0)
    data_feat = normalize(data_feat)

    return data_feat


if __name__ == "__main__":
    data = pd.read_csv("data/data_ready.csv", index_col=[0])

    data_feat = features_emb(
        data, features=["verified", "followers", "friends", "lists","tweets"]
    )
    np.save("embeddings/features.npy", data_feat)

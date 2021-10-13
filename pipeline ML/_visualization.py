import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly


def visualize_emb(emb: np.array, data: pd.DataFrame, hover_data: list):
    """This function takes an embedding and the dataframe with the features and output a figure

    Args:
        emb (np.array): embeddings
        data (pd.DataFrame): features
        hover_data (list): information on the hover
    """

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb)

    df_emb = pd.DataFrame(emb_pca)
    df_emb = pd.concat([df_emb, data], axis=1)

    df_emb["label"] = df_emb["label"].astype("object")
    fig = px.scatter(df_emb, x=0, y=1, color="label", hover_data=hover_data)

    return fig


if __name__ == "__main__":

    data = pd.read_csv("data/data_ready.csv", index_col=[0])

    # Load the embeddings computed before
    X_tfidf = np.load("embeddings/tfidf.npy", allow_pickle=True)
    X_bert = np.load("embeddings/bert.npy", allow_pickle=True)
    X_image = np.load("embeddings/full_profile_pictures.npy", allow_pickle=True)
    X_features = np.load("embeddings/features.npy", allow_pickle=True)

    X = np.concatenate((X_bert, X_tfidf, X_image), axis=1)

    fig = visualize_emb(X_tfidf, data, hover_data=["description"])
    fig.show()
    fig.write_html("figures/file.html")

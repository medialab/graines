import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Data
data = pd.read_csv("data/data_ready.csv", index_col=[0])
data_tfidf = data[["screen_name", "description", "name",'location']]
data_tfidf = data_tfidf.fillna(" ")
data_tfidf = (
    data_tfidf["screen_name"]
    + "//"
    + data_tfidf["description"]
    + "//"
    + data_tfidf["name"]
    + "//"
    + data_tfidf["location"]
)
X = list(data_tfidf.values)


model = SentenceTransformer("distiluse-base-multilingual-cased")
embeddings = model.encode(X, show_progress_bar=True)
np.save("embeddings/bert.npy", embeddings)

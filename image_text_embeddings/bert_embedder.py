from neocortext.sentence_embedding.bert_embedding import bert_embedding
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

data = pd.read_csv("graines_et_non_graines.csv", index_col=[0])
data["text"] = data["screen_name"] + "//" + data["description"]
data["text"] = data["text"].astype(str)
texts = list(data["text"])

model = SentenceTransformer("distiluse-base-multilingual-cased")
embeddings = model.encode(texts, show_progress_bar=True)
np.save("bert_embeddings.npy", embeddings)

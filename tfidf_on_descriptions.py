from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from create_ground_truth import LABEL_FILE_NAME
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

corpus = pd.read_csv(LABEL_FILE_NAME).description.fillna("").tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
svd = TruncatedSVD(n_components=300, random_state=42)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

np.save("tfidf_on_descriptions.npy", X)

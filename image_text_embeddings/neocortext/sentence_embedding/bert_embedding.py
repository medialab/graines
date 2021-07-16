
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('distiluse-base-multilingual-cased')

def bert_embedding(corpus, save=True):
	embeddings = model.encode(corpus, show_progress_bar=True)
	if save is True:
		with open('embeddings.pickle', 'wb') as f:
			pickle.dump(embeddings, f)

	return embeddings


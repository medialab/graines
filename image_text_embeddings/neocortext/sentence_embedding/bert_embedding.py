
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('distilroberta-base-paraphrase-v1')

def bert_embedding(corpus, save=True):
	embeddings = model.encode(corpus, show_progress_bar=True)
	if save is True:
		with open('embeddings.pickle', 'wb') as f:
			pickle.dump(embeddings, f)

	return embeddings


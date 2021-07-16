from neocortext.sentence_embedding.bert_embedding import bert_embedding
import pickle
import pandas as pd

data = pd.read_csv('graines_et_non_graines.csv', index_col = [0])
data['text'] = data['screen_name'] + '//' + data['description']
data['text'] = data['text'].astype(str)
texts = list(data['text'])

embeddings = bert_embedding(texts, save=True)


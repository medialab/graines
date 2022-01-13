import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
import nltk
from config import objective
nltk.download('punkt')

data = pd.read_csv("data/followers_metadata_version_2021_10_19.csv", dtype={'id':'string'})
data_flamboyant = pd.read_csv("data/graines_metadata.csv", dtype={'id':'string'})

#!!!!!
#data['description']=data['description'].dropna()#rajouter name et screen_name
#data_flamboyant['description']=data['description'].dropna()

data['text']=data['screen_name']+' '+data['description']+' '+data['name']+' '+data['location']
data_flamboyant['text']=data_flamboyant['screen_name']+' '+data_flamboyant['description']+' '+data_flamboyant['name']+' '+data_flamboyant['location']

def lower_it(string):
    try:
        return string.lower()
    except:
        return ''


tagged_data = [TaggedDocument(words=word_tokenize(lower_it(_d)), tags=[str(i)]) for i, _d in enumerate(list(data['text'])+list(data_flamboyant['text']))]
model = Doc2Vec(#size=vec_size,
				alpha=.025,
				min_alpha=0.00025,
				vector_size=50, window=10, min_count=10, workers=4,
				dm = 1,max_vocab_size=50000)

model.build_vocab(tagged_data)
max_epochs=5
for epoch in range(max_epochs):
	print('iteration {} of {}'.format(epoch, max_epochs))
	model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
	model.alpha -= 0.0002
# fix the learning rate, no decay
model.min_alpha = model.alpha


index_id = {}
for i, id in enumerate(list(data.screen_name)+list(data_flamboyant.screen_name)):
    index_id[id] = i
data_ready = pd.read_csv("data/data_ready.csv", index_col=[0],dtype={'user_id':'string'})

doctovecs=[]
h=0

if objective == "report":
    for x in data_ready.screen_name:
        doctovecs.append(model.docvecs[index_id[x]])
    np.save("embeddings/doc2vecs.npy", doctovecs)
elif objective == "classification":
    for x in tqdm(data.screen_name):
        doctovecs.append(model.docvecs[index_id[x]])
    np.save("embeddings/doc2vecs_final_predict.npy", doctovecs)



import pandas as pd
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
import nltk
nltk.download('punkt')

data = pd.read_csv("data/followers_metadata_version_2021_10_19.csv",dtype={'id':'string'})
data_flamboyant = pd.read_csv("data/graines_metadata.csv",dtype={'id':'string'})

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

    
try:
	model = Word2Vec.load("word2vec.model")
except:
	tagged_data = [TaggedDocument(words=word_tokenize(lower_it(_d)), tags=[str(i)]) for i, _d in enumerate(list(data['text'])+list(data_flamboyant['text']))]
	model = Doc2Vec(#size=vec_size,
	                alpha=.025, 
	                min_alpha=0.00025,
	                vector_size=50, window=10, min_count=10, workers=4,
	                dm = 1,max_vocab_size=50000)
	
	model.build_vocab(tagged_data)
	max_epochs=5
	for epoch in range(max_epochs):
	    print('iteration {0}'.format(epoch))
	    model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
	    model.alpha -= 0.0002
    # fix the learning rate, no decay
	model.min_alpha = model.alpha






	model.save("word2vec.model")


print(model.wv['journaliste'])
print(model.wv.most_similar('journalist'))
print(len(model.docvecs))
print(len(data))

index_id={}
for i,id in enumerate(list(data.screen_name)+list(data_flamboyant.screen_name)):
    index_id[id]=i
data_ready = pd.read_csv("data/data_ready.csv", index_col=[0],dtype={'user_id':'string'})
index_id['Marcusjvn']

doctovecs=[]
h=0
for x in data_ready.screen_name:
    #try:
    if 1:#:
        doctovecs.append(model.docvecs[index_id[x]])
    #except:
        #doctovecs.append(model.docvecs[index_id['Marcusjvn']])#no desc vector !!!
        #h+=1
print()
np.save("embeddings/doc2vecs.npy", doctovecs)



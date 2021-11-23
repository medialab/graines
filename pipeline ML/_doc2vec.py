import pandas as pd
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re

data = pd.read_csv("data/followers_metadata_version_2021_09_21.csv",dtype={'id':'string'})

#!!!!!
data['description']=data['description'].dropna()#rajouter name et screen_name

def lower_it(string):
    try:
        return string.lower()
    except:
        return ''

    
tagged_data = [TaggedDocument(words=word_tokenize(lower_it(_d)), tags=[str(i)]) for i, _d in enumerate(data['description'])]


vec_size = 20
alpha = 0.025

model = Doc2Vec(#size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                vector_size=50, window=10, min_count=10, workers=4,
                dm =1,max_vocab_size=50000)
  
model.build_vocab(tagged_data)


max_epochs=5
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha





#np.save("embeddings/bert.npy", embeddings)
model.save("d2v.npy")
print("Model Saved")



print(model.wv['journaliste'])
print(model.wv.most_similar('journalist'))
len(model.docvecs)
len(data)

index_id={}
for i,id in enumerate(data.screen_name):
    index_id[id]=i
data_ready = pd.read_csv("data/data_ready.csv", index_col=[0],dtype={'user_id':'string'})
index_id['Marcusjvn']

doctovecs=[]
h=0
for x in data_ready.screen_name:
    try:
        doctovecs.append(model.docvecs[index_id[x]])
    except:
        doctovecs.append(model.docvecs[index_id['Marcusjvn']])#no desc vector !!!
        h+=1

np.save("embeddings/doc2vecs.npy", doctovecs)
   
    


# In[64]:


model.docvecs[index_id['Marcusjvn']]


# In[72]:


np.save("embeddings/doc2vecs.npy", np.array(doctovecs))


# In[ ]:





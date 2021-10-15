#!/usr/bin/env python
# coding: utf-8

# In[23]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
df=pd.read_csv("data/followers_graines_version_2021_09_21.csv",dtype={'follower_id':'str',"twitter_handle":'str'})
dfri=pd.read_csv("data/friends_graines.csv",dtype={'friend_id':'str',"twitter_handle":'str'})

metaf=pd.read_csv("data/followers_metadata_version_2021_09_21.csv")#,dtype={'follower_id':'str',"twitter_handle":'str'})
metag=pd.read_csv("data/graines_metadata.csv")#,dtype={'follower_id':'str',"twitter_handle":'str'})


#dgg=pd.read_csv('data/graines_metadata.csv',dtype={'id':'str'})
#dgg.head()


#raisins=set(map(lambda x: str(x),dgg['id'].values))


dfri.head()


# In[ ]:





# In[10]:


#len(df),len(dfri),len(metaf),len(metag),len(dgg)


# In[17]:


raisins=list(df.twitter_handle.unique())
len(raisins)


# In[18]:


dfri.sample(5)


# In[21]:


#raisins


# In[24]:



nb_followers_dict= dict(zip(metag.screen_name, metag.followers))
nb_friends_dict= dict(zip(metag.screen_name, metag.friends))



# In[27]:


dfri.head()


# In[28]:


net_fri={}
for x,y in zip(dfri['friend_id'],dfri['twitter_handle']):
    if y in raisins:
        net_fri.setdefault(x,[]).append(y)

len(net_fri)


# In[31]:


net_fol={}
for x,y in zip(df['follower_id'],df['twitter_handle']):
    if y in raisins:
        net_fol.setdefault(x,[]).append(y)

len(net_fol)


# In[ ]:





# In[33]:


dfri['nb_friends']=dfri['twitter_handle'].map(nb_friends_dict.get)
dfri.sample(10)


# In[34]:


df['nb_followers']=df['twitter_handle'].map(nb_followers_dict.get)
df.sample(10)



# In[38]:



dfri.head()
net_fri_norm={}
for x,y,w in zip(dfri['friend_id'],dfri['twitter_handle'],dfri['nb_friends']):
    if y in raisins:
        
        if w>0:
            net_fri_norm.setdefault(x,[]).append(1/np.log(w))
len(net_fri_norm)


# In[39]:



#dfol.head()
net_fol_norm={}
for x,y,w in zip(df['follower_id'],df['twitter_handle'],df['nb_followers']):
    if y in raisins:
        
        if w>0:
            net_fol_norm.setdefault(x,[]).append(1/np.log(w))

len(net_fol_norm)


# In[53]:


dg = pd.read_csv("data/data_ready.csv",dtype={"user_id": "str"})
#dg=dg.drop_duplicates(subset=['screen_name'])
len(dg)


# In[56]:


dg.sentiment.value_counts()


# In[57]:


topo={}
for id,fol,fri in zip(dg['user_id'],dg['followers'], dg['friends']):
    print (id,fol,fri,len(net_fol.get(str(id),[])),len(net_fri.get(str(id),[])),sum(net_fri_norm.get(str(id),[])))
    #print (np.array(float(len(net.get(str(id),[]))/(fri+1)),float(len(net_fri.get(str(id),[]))/(fol+1))))
    if 1:#fol>0:
        topo[id]={}
        topo[id]['raw number graines following me']=float(len(net_fri.get(str(id),[])))
        topo[id]['raw number of graines I follow']=float(len(net_fol.get(str(id),[])))
        

        topo[id]['proportion of graines following me']=float(len(net_fri.get(str(id),[]))/(fol+1))
        topo[id]['normalized proportion of graines following me']=float(sum(net_fri_norm.get(str(id),[]))/(fol+1))

        topo[id]['proportion of graines I follow']=float(len(net_fol.get(str(id),[]))/(fri+1))
        topo[id]['normalized proportion of graines I follow']=float(sum(net_fol_norm.get(str(id),[]))/(fri+1))
        print ('topo',id,topo[id])

# In[93]:


pd.DataFrame.from_dict(topo).transpose().to_csv('topology.csv')

vector_topo=pd.DataFrame.from_dict(topo).transpose()

dtopo=pd.DataFrame.from_dict(topo).transpose()
np.save("embeddings/topo.npy", dtopo.values)



# In[49]:


#dg.sort_values(by='followers',ascending=False)


# In[50]:


len(vector_topo)


# In[10]:


#len(meta)


# In[ ]:





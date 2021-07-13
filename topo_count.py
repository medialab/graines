#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
df=pd.read_csv("data/followers_graines.csv.gz",dtype={'follower_id':'str',"twitter_handle":'str'})
dfri=pd.read_csv("data/friends_graines.csv",dtype={'friend_id':'str',"twitter_handle":'str'})


dg=pd.read_csv('graines_et_non_graines.csv',dtype={'id':'str'})
dg.head()


raisins=set(map(lambda x: str(x),dg['id'].values))


dfri.head()
net_fri={}
for x,y in zip(dfri['friend_id'],dfri['twitter_handle']):
    if x in raisins:
        net_fri.setdefault(x,[]).append(y)

net={}
for x,y in zip(df['follower_id'],df['twitter_handle']):
    if x in raisins:
        net.setdefault(x,[]).append(y)


topo={}
for id,fol,fri in zip(dg['id'],dg['followers'], dg['friends']):
    print (id,fol,fri,len(net.get(str(id),[])),len(net_fri.get(str(id),[])))
    #print (np.array(float(len(net.get(str(id),[]))/(fri+1)),float(len(net_fri.get(str(id),[]))/(fol+1))))
    topo[id]={}
    topo[id]['proportion of graines following me']=float(len(net.get(str(id),[]))/(fri+1))
    topo[id]['proportion of graines I follow']=float(len(net_fri.get(str(id),[]))/(fol+1))


# In[93]:


pd.DataFrame.from_dict(topo).transpose().to_csv('topology.csv')

vector_topo=pd.DataFrame.from_dict(topo).transpose()

dtopo=pd.DataFrame.from_dict(topo).transpose()
np.save("topo.npy", dtopo.values)


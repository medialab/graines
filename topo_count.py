#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

df = pd.read_csv(
    "data/followers_graines.csv.gz",
    dtype={"follower_id": "str", "twitter_handle": "str"},
)
dfri = pd.read_csv(
    "data/friends_graines.csv.gz", dtype={"friend_id": "str", "twitter_handle": "str"}
)

meta = pd.read_csv(
    "data/followers_graines.csv.gz",
    dtype={"follower_id": "str", "twitter_handle": "str"},
)
dfri = pd.read_csv(
    "data/friends_graines.csv.gz", dtype={"friend_id": "str", "twitter_handle": "str"}
)


dg = pd.read_csv("graines_metadata.csv", dtype={"id": "str"})
dg.head()


raisins = set(map(lambda x: str(x), dg["id"].values))


dfri.head()
net_fri = {}
for x, y in zip(dfri["friend_id"], dfri["twitter_handle"]):
    if x in raisins:
        net_fri.setdefault(x, []).append(y)

net = {}
for x, y in zip(df["follower_id"], df["twitter_handle"]):
    if x in raisins:
        net.setdefault(x, []).append(y)


nb_followers_dict = dict(zip(dg.screen_name, dg.followers))


dfri["nb_followers"] = dfri["twitter_handle"].map(nb_followers_dict.get)
dfri.sample(10)

dfri.head()
net_fri_norm = {}
for x, y, w in zip(dfri["friend_id"], dfri["twitter_handle"], dfri["nb_followers"]):
    if x in raisins:

        if w > 0:
            net_fri_norm.setdefault(x, []).append(1 / np.log(w))
        else:
            print(x, y, w)

# matadon_ and mcefic missing in graines_metadata.csv


topo = {}
for id, fol, fri in zip(dg["id"], dg["followers"], dg["friends"]):
    print(
        id,
        fol,
        fri,
        len(net.get(str(id), [])),
        len(net_fri.get(str(id), [])),
        sum(net_fri_norm.get(str(id), [])),
    )
    # print (np.array(float(len(net.get(str(id),[]))/(fri+1)),float(len(net_fri.get(str(id),[]))/(fol+1))))
    if fol > 0:
        topo[id] = {}
        topo[id]["proportion of graines following me"] = float(
            len(net.get(str(id), [])) / (fri + 1)
        )
        topo[id]["proportion of graines I follow"] = float(
            len(net_fri.get(str(id), [])) / (fol + 1)
        )
        topo[id]["normalized proportion of graines I follow"] = float(
            sum(net_fri_norm.get(str(id), [])) / (fol + 1)
        )


pd.DataFrame.from_dict(topo).transpose().to_csv("topology.csv")

vector_topo = pd.DataFrame.from_dict(topo).transpose()

dtopo = pd.DataFrame.from_dict(topo).transpose()
np.save("topo.npy", dtopo.values)

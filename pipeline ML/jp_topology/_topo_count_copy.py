import numpy as np
import pandas as pd

df = pd.read_csv(
    "data/followers_graines_version_2021_09_21.csv",
    dtype={"follower_id": "str", "twitter_handle": "str"},
)
dfri = pd.read_csv(
    "data/friends_graines.csv.gz", dtype={"friend_id": "str", "twitter_handle": "str"}
)

metaf = pd.read_csv(
    "data/followers_metadata_version_2021_09_21.csv"
)  # ,dtype={'follower_id':'str',"twitter_handle":'str'})
metag = pd.read_csv(
    "data/graines_metadata.csv"
)  # ,dtype={'follower_id':'str',"twitter_handle":'str'})


def topo(df, dfri, metaf, metag) -> np.array:
    return


dfri.head()

raisins = list(df.twitter_handle.unique())
len(raisins)

dfri.sample(5)


nb_followers_dict = dict(zip(metag.screen_name, metag.followers))
nb_friends_dict = dict(zip(metag.screen_name, metag.friends))


dfri.head()


net_fri = {}
for x, y in zip(dfri["friend_id"], dfri["twitter_handle"]):
    if y in raisins:
        net_fri.setdefault(x, []).append(y)

len(net_fri)


net_fol = {}
for x, y in zip(df["follower_id"], df["twitter_handle"]):
    if y in raisins:
        net_fol.setdefault(x, []).append(y)

len(net_fol)


dfri["nb_friends"] = dfri["twitter_handle"].map(nb_friends_dict.get)
dfri.sample(10)


df["nb_followers"] = df["twitter_handle"].map(nb_followers_dict.get)
df.sample(10)


dfri.head()
net_fri_norm = {}
for x, y, w in zip(dfri["friend_id"], dfri["twitter_handle"], dfri["nb_friends"]):
    if y in raisins:

        if w > 0:
            net_fri_norm.setdefault(x, []).append(1 / np.log(w))

# dfol.head()
net_fol_norm = {}
for x, y, w in zip(df["follower_id"], df["twitter_handle"], df["nb_followers"]):
    if y in raisins:

        if w > 0:
            net_fol_norm.setdefault(x, []).append(1 / np.log(w))

len(net_fol_norm)


dg = pd.read_csv("data/data_ready.csv", dtype={"user_id": "str"})
# dg=dg.drop_duplicates(subset=['screen_name'])
len(dg)

dg.sentiment.value_counts()

topo = {}
for id, fol, fri in zip(dg["user_id"], dg["followers"], dg["friends"]):
    print(
        id,
        fol,
        fri,
        len(net_fol.get(str(id), [])),
        len(net_fri.get(str(id), [])),
        sum(net_fri_norm.get(str(id), [])),
    )
    # print (np.array(float(len(net.get(str(id),[]))/(fri+1)),float(len(net_fri.get(str(id),[]))/(fol+1))))
    if 1:  # fol>0:
        topo[id] = {}
        topo[id]["raw number of followers"] = fol
        topo[id]["raw number of friends"] = fri

        topo[id]["raw number graines following me"] = float(
            len(net_fri.get(str(id), []))
        )
        topo[id]["raw number of graines I follow"] = float(
            len(net_fol.get(str(id), []))
        )

        topo[id]["normalized number of graines following me"] = float(
            sum(net_fri_norm.get(str(id), []))
        )
        topo[id]["normalized number of graines I follow"] = float(
            sum(net_fol_norm.get(str(id), []))
        )

        topo[id]["proportion of graines following me"] = float(
            len(net_fri.get(str(id), [])) / (fol + 1)
        )
        topo[id]["normalized proportion of graines following me"] = float(
            sum(net_fri_norm.get(str(id), [])) / (fol + 1)
        )

        topo[id]["proportion of graines I follow"] = float(
            len(net_fol.get(str(id), [])) / (fri + 1)
        )
        topo[id]["normalized proportion of graines I follow"] = float(
            sum(net_fol_norm.get(str(id), [])) / (fri + 1)
        )
        print("topo", id, topo[id])

# pd.DataFrame.from_dict(topo).transpose().to_csv('topology.csv')

vector_topo = pd.DataFrame.from_dict(topo).transpose()

dtopo = pd.DataFrame.from_dict(topo).transpose()
np.save("embeddings/topo.npy", dtopo.values)

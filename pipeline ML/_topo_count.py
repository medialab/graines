import numpy as np
import pandas as pd
from config import objective
from glob import glob


def topo(
    objective: str,
    df: pd.DataFrame,
    dfri: pd.DataFrame,
    metag: pd.DataFrame,
    dg: pd.DataFrame,
) -> np.array:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        dfri (pd.DataFrame): [description]
        metaf (pd.DataFrame): [description]
        metag (pd.DataFrame): [description]
        dg (pd.DataFrame): [description]

    Returns:
        np.array: [description]
    """
    if objective == "report":
        raisins = list(df.twitter_handle.unique())
    elif objective == "classification":
        raisins = list(df.screen_name.unique())
    else:
        raise ValueError("objective should be either 'report' or 'classification'")
    nb_followers_dict = dict(zip(metag.screen_name, metag.followers))
    nb_friends_dict = dict(zip(metag.screen_name, metag.friends))

    net_fri = {}
    for x, y in zip(dfri["friend_id"], dfri["twitter_handle"]):
        if y in raisins:
            net_fri.setdefault(x, []).append(y)

    net_fol = {}
    for x, y in zip(df["follower_id"], df["twitter_handle"]):
        if y in raisins:
            net_fol.setdefault(x, []).append(y)

    dfri["nb_friends"] = dfri["twitter_handle"].map(nb_friends_dict.get)
    df["nb_followers"] = df["twitter_handle"].map(nb_followers_dict.get)

    net_fri_norm = {}
    for x, y, w in zip(dfri["friend_id"], dfri["twitter_handle"], dfri["nb_friends"]):
        if y in raisins:

            if w > 0:
                net_fri_norm.setdefault(x, []).append(1 / np.log(w))

    net_fol_norm = {}
    for x, y, w in zip(df["follower_id"], df["twitter_handle"], df["nb_followers"]):
        if y in raisins:

            if w > 0:
                net_fol_norm.setdefault(x, []).append(1 / np.log(w))

    topo = {}
    for id, fol, fri in zip(dg["user_id"], dg["followers"], dg["friends"]):
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

    dtopo = pd.DataFrame.from_dict(topo).transpose()

    return dtopo.values


if __name__ == "__main__":
    df = pd.read_csv(
        "data/followers_graines_version_2021_09_21.csv",
        dtype={"follower_id": "str", "twitter_handle": "str"},
    )
    dfri = pd.read_csv(
        "data/friends_graines.csv.gz",
        dtype={"friend_id": "str", "twitter_handle": "str"},
    )
    metag = pd.read_csv("data/graines_metadata.csv")
    dg = pd.read_csv("data/data_ready.csv", dtype={"user_id": "str"})

    emb = topo("report", df, dfri, metag, dg)
    np.save("embeddings/topo.npy", emb)

    if objective == "classification":
        emb = topo(objective, df, dfri, metag, dg)
        np.save("embeddings/topo_final_predict.npy")

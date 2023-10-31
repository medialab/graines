import numpy as np
import pandas as pd
from config import objective
from glob import glob
from tqdm import tqdm


def topo(
    objective: str,
    graines: set,
    metaf: pd.DataFrame,
    metag: pd.DataFrame,
    dg: pd.DataFrame,
) -> np.array:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        metaf (pd.DataFrame): [description]
        metag (pd.DataFrame): [description]
        dg (pd.DataFrame): [description]

    Returns:
        np.array: [description]
    """

    # nb_followers_dict = dict(zip(metag.screen_name, metag.followers))
    # nb_friends_dict = dict(zip(metag.screen_name, metag.friends))
    graines_nb_followers_weighting_dict = dict(zip(metag.screen_name, 1 / np.log(metag.followers)))
    graines_nb_friends_weighting_dict = dict(zip(metag.screen_name, 1 / np.log(metag.friends)))

    # net_fri = {}
    # for x, y in zip(dfri["friend_id"], dfri["twitter_handle"]):
    #     if y in raisins:
    #         net_fri.setdefault(x, []).append(y)
    #
    # net_fol = {}
    # for x, y in zip(df["follower_id"], df["twitter_handle"]):
    #     if y in raisins:
    #         net_fol.setdefault(x, []).append(y)

    # dfri["nb_friends"] = dfri["twitter_handle"].map(nb_friends_dict.get)
    # df["nb_followers"] = df["twitter_handle"].map(nb_followers_dict.get)

    net_fri_norm = {}
    for x, graines_in_friends in zip(metaf["id"], metaf["graines_in_friends"]):
        for g in graines_in_friends.split('|'):
            if g in graines:
                net_fri_norm.setdefault(x, []).append(graines_nb_followers_weighting_dict[g.strip("\n")])
        else:
            net_fri_norm.setdefault(x, [])

    net_fol_norm = {}
    for x, graines_in_followers in zip(metaf["id"], metaf["graines_in_followers"]):
        if not pd.isna(graines_in_followers):
            for g in graines_in_followers.split('|'):
                    if g in graines:
                        net_fol_norm.setdefault(x, []).append(graines_nb_friends_weighting_dict[g.strip("\n")])


    topo = {}
    user_id = "user_id"
    if objective == "classification":
        dg = metaf
        user_id = "id"
    dtopo = np.array([[
        fol, # raw number of followers
        fri, # raw number of friends
        float(count_graines_in_followers), # raw number graines following me
        float(count_graines_in_friends), # raw number of graines I follow
        float(sum(net_fri_norm.get(str(id), []))), # normalized number of graines following me
        float(sum(net_fol_norm.get(str(id), []))), # normalized number of graines I follow
        float(count_graines_in_followers / (fol + 1)), # proportion of graines following me
        float(sum(net_fri_norm.get(str(id), [])) / (fol + 1)), # normalized proportion of graines following me
        float(count_graines_in_friends / (fri + 1)), # proportion of graines I follow
        float(sum(net_fol_norm.get(str(id), [])) / (fri + 1))
    ] for id, fol, fri, count_graines_in_friends, count_graines_in_followers in tqdm(zip(
            dg[user_id],
            dg["followers"],
            dg["friends"],
            dg["count_graines_in_friends"],
            dg["count_graines_in_followers"]
    ), total=dg.shape[0])])
    # for id, fol, fri, count_graines_in_friends, count_graines_in_followers in tqdm(zip(
    #         dg[user_id],
    #         dg["followers"],
    #         dg["friends"],
    #         dg["count_graines_in_friends"],
    #         dg["count_graines_in_followers"]
    # ), total=dg.shape[0]):
    #     topo[id] = {}
    #     topo[id]["raw number of followers"] = fol
    #     topo[id]["raw number of friends"] = fri
    #
    #     # topo[id]["raw number graines following me"] = float(
    #     #     len(net_fri.get(str(id), []))
    #     # )
    #     # topo[id]["raw number of graines I follow"] = float(
    #     #     len(net_fol.get(str(id), []))
    #     # )
    #     topo[id]["raw number graines following me"] = float(count_graines_in_followers)
    #     topo[id]["raw number of graines I follow"] = float(count_graines_in_friends)
    #
    #     topo[id]["normalized number of graines following me"] = float(
    #         sum(net_fri_norm.get(str(id), []))
    #     )
    #     topo[id]["normalized number of graines I follow"] = float(
    #         sum(net_fol_norm.get(str(id), []))
    #     )
    #
    #     topo[id]["proportion of graines following me"] = float(
    #         count_graines_in_followers / (fol + 1)
    #     )
    #     topo[id]["normalized proportion of graines following me"] = float(
    #         sum(net_fri_norm.get(str(id), [])) / (fol + 1)
    #     )
    #
    #     topo[id]["proportion of graines I follow"] = float(
    #         count_graines_in_friends / (fri + 1)
    #     )
    #     topo[id]["normalized proportion of graines I follow"] = float(
    #         sum(net_fol_norm.get(str(id), [])) / (fri + 1)
    #         )
    #
    #
    # dtopo = pd.DataFrame.from_dict(topo).fillna(0.).transpose()
    # dtopo = np.array(dtopo.values)
    dtopo = np.nan_to_num(dtopo, nan=0)
    print(np.isfinite(dtopo).all())
    return dtopo


if __name__ == "__main__":
    graines = set(pd.read_csv("data/VF-Carte Raison - Corpus-final.csv").twitter_handle.unique())
    metaf = pd.read_csv(
        "data/followers_metadata_version_2021_10_19.csv",
        dtype={"id": "str", "follower_id": "str", "twitter_handle": "str"},
    )
    print(metaf.shape[0])
    metag = pd.read_csv("data/graines_metadata.csv")
    dg = pd.read_csv("data/data_ready.csv", dtype={"user_id": "str"})

    emb = topo("report", graines, metaf, metag, dg)
    np.save("embeddings/topo.npy", emb)

    if objective == "classification":
        emb = topo("classification", graines, metaf, metag, dg)
        print(emb.shape[0])
        np.save("embeddings/topo_final_predict.npy", emb)

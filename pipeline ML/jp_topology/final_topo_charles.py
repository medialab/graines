import pandas as pd
import numpy as np
import ast


def norm(x):
    return 1 / np.log(x)


# Load all the followers of the graines
foll = pd.read_csv(
    "data/followers_graines_version_2021_09_21.csv",
    index_col=[0],
    low_memory=False,
    usecols=["twitter_handle", "follower_id"],
).reset_index()

# Load of the friends of the graines
fri = pd.read_csv(
    "data/friends_graines.csv.gz",
    index_col=[0],
    low_memory=False,
    usecols=["twitter_handle", "friend_id"],
).reset_index()

# Create the dictionary of graines and their followers
group_foll = (
    foll.groupby("twitter_handle")["follower_id"]
    .count()
    .rename("count_followers")
    .reset_index()
)
dict_follo = dict(zip(group_foll["twitter_handle"], group_foll["count_followers"]))

# Create the dictionary of graines and their friends
group_fri = (
    fri.groupby("twitter_handle")["friend_id"]
    .count()
    .rename("count_friends")
    .reset_index()
)
dict_fri = dict(zip(group_fri["twitter_handle"], group_fri["count_friends"]))

# Load the Dataset
columns = [
    "user_id",
    "count_graines_in_followers",
    "graines_in_followers",
    "followers",
    "friends",
    "count_graines_in_friends",
    "graines_in_friends",
]

new_data = pd.read_csv("data/data_ready.csv", index_col=[0], usecols=columns)


new_data_follo = new_data.explode("graines_in_followers")
new_data_follo["total_grain_follo"] = new_data_follo["graines_in_followers"].apply(
    lambda x: dict_follo.get(x)
)

# get the sum of the total followers of graine follows by an indivual
new_data_follo = (
    new_data_follo.groupby("user_id")["total_grain_follo"].sum().reset_index()
)

# Normalize by the formula
new_data_follo["normalize_follo"] = new_data_follo["total_grain_follo"].apply(
    lambda x: norm(x)
)

new_data_fri = new_data.explode("graines_in_friends")
new_data_fri["total_grain_fri"] = new_data_fri["graines_in_friends"].apply(
    lambda x: dict_fri.get(x)
)

# get the sum of the total followers of graine follows by an indivual
new_data_fri = new_data_fri.groupby("user_id")["total_grain_fri"].sum().reset_index()

# Normalize by the formula
new_data_fri["normalize_friends"] = new_data_fri["total_grain_fri"].apply(
    lambda x: norm(x)
)

# Final Concat
concat_score = pd.merge(new_data_follo, new_data_fri, on="user_id")
concat_score = pd.merge(concat_score, new_data, on="user_id")
concat_score["prop_graine_friends"] = (
    concat_score["count_graines_in_friends"] / concat_score["friends"]
)
concat_score["prop_graine_followers"] = (
    concat_score["count_graines_in_followers"] / concat_score["followers"]
)

final = concat_score[
    [
        "count_graines_in_followers",
        "prop_graine_followers",
        "count_graines_in_friends",
        "prop_graine_friends",
        "normalize_friends",
        "normalize_follo",
    ]
]

final = final.fillna(final.mean())
final = final.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
final = np.array(final)
np.save("embeddings/topo.npy", final)

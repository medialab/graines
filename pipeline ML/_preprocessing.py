import pandas as pd
import numpy as np
import os

if not os.path.exists("data"):
    os.makedirs("data")

# Load the initial data and rename the id as user_id
data = pd.read_csv("data/followers_metadata_version_2021_10_19.csv")
data = data[data.id.notna()]
data["id"] = data["id"].astype(int)
data = data.rename(columns={"id": "user_id"})

# Load annotated data and concat the annotations
data_1 = pd.read_csv("data/data annotated/project-116-at-2021-10-13-06-37-e3dd8cbd.csv")
data_2 = pd.read_csv("data/data annotated/project-118-at-2021-10-13-06-38-e2172a5e.csv")
df_ann = pd.concat([data_1, data_2])
key = ["screen_name", "name", "description", "protected", "location"]
df_ann = df_ann[key + ["sentiment"]].drop_duplicates(key)

# Merge the initial data and the results of the annotations based on similar various keys
merged = pd.merge(
    data,
    df_ann,
    on=key,
    how="right"
)

# Deal with data that have not been annotated
final = merged[merged.sentiment.notna()].reset_index(drop=True)
final["flamboyant_seed"] = 0

# Load the seeds
seeds = pd.read_csv("data/VF-Carte Raison - Corpus-final.csv")
seeds = seeds[["twitter_handle"]]
seeds["sentiment"] = "graine"
seeds["flamboyant_seed"] = 1

# Merge the initial data and the seeds
seeds = pd.merge(
    seeds,
    data,
    left_on="twitter_handle",
    right_on="screen_name",
    how="left"
)[final.columns]

# convert all numerical columns to integers
seeds = seeds[seeds.screen_name.notna()]
for col, dtype in seeds.dtypes.iteritems():
    if dtype == "float64" and col != "witheld_scope" and col != "witheld_in_countries":
        seeds[col] = seeds[col].astype(int)

# Concatenate annotated data and seeds, remove duplicates, and shuffle them
final = pd.concat([final, seeds])
final = final.drop_duplicates("user_id")
final = final.sample(final.shape[0], random_state=0)

# Change the labeling into a categorical variable
map_code = {"non-graine": 0, "graine": 1}
final["label"] = final["sentiment"].map(map_code)
final.to_csv("data/data_ready.csv")

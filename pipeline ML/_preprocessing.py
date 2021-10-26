import pandas as pd
import numpy as np
import os

if not os.path.exists("data"):
    os.makedirs("data")

# Load the initial data and rename the id as user_id
data = pd.read_csv(
    "data/2000_followers_graines_version_2021_10_19.csv", index_col=[0]
)
data = data.rename(columns={"id": "user_id"})

# Load and concat the annotations
data_1 = pd.read_csv("data/data annotated/project-116-at-2021-10-13-06-37-e3dd8cbd.csv")
data_2 = pd.read_csv("data/data annotated/project-118-at-2021-10-13-06-38-e2172a5e.csv")
df_ann = pd.concat([data_1, data_2])

# Merge the initial data and the results of the annotations based on similar various keys
key = ["screen_name", "name", "description", "protected", "location"]
merged = pd.merge(data[key + ["user_id"]], df_ann, on=key, how="left")

# Deal with the common data (I just randomly keep one )
merged = merged.drop_duplicates(["user_id"], keep="first")
merged = merged[["user_id", "sentiment"]]
merged = merged.reset_index(drop=True)

# Deal with data that have not been annotated
merged_fin = merged[merged.sentiment.notna()].reset_index(drop=True)

final = pd.merge(merged_fin, data, on="user_id")

# Change the labeling into a categorical variable
map_code = {"non-graine": 0, "graine": 1}
final["label"] = final["sentiment"].map(map_code)

final.to_csv("data/data_ready.csv")

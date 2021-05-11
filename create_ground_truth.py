import pandas as pd
import numpy as np

LABEL_FILE_NAME = "graines_et_non_graines.csv"
df = pd.DataFrame()

for i, g in enumerate(["non_graines", "graines"]):
    graines = pd.read_csv("{}_metadata.csv".format(g), dtype={"id": str})
    # filter accounts that where not found using Twitter API
    graines = graines[graines.screen_name.notna()]
    # type columns containing integers
    for int_val in ["tweets", "followers", "friends", "likes", "lists", "timestamp_utc"]:
        graines[int_val] = graines[int_val].astype(int)
    graines["labels"] = i
    df = df.append(graines)

df.to_csv(LABEL_FILE_NAME, index=False)
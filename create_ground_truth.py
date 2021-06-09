import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


LABEL_FILE_NAME = "graines_et_non_graines.csv"


if __name__ == '__main__':
    df = pd.DataFrame()
    for i, g in enumerate(["non_graines", "graines"]):
        graines = pd.read_csv("{}_metadata.csv".format(g), dtype={"id": str})
        # filter accounts that where not found using Twitter API
        graines = graines[graines.screen_name.notna()]
        # type columns containing integers
        for int_val in ["tweets", "followers", "friends", "likes", "lists", "timestamp_utc"]:
            graines[int_val] = graines[int_val].astype(int)
        # add 0/1 labels
        if "graine" in graines.columns:
            graines["graine"] = graines["graine"].fillna(i).astype(str)
            graines["label"] = graines["graine"]
            graines.loc[graines.graine.str.startswith("?"), "label"] = 0
        else:
            graines["label"] = i
        df = df.append(graines)
    df = df.drop_duplicates("id", keep="last")
    df.to_csv(LABEL_FILE_NAME, index=False)
    logging.info("Saved labels to {}".format(LABEL_FILE_NAME))

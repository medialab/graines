import pandas as pd
from neocortext.extract_data.twitter_images import extract_images_from_profile
tweets = pd.read_csv('graines_et_non_graines.csv', index_col = [0])

extract_images_from_profile(tweets)

from twitter_images import extract_images_from_profile
from image_embedding import pipeline_image_embedding, image_embedding, visualize_images
import pandas as pd
import numpy as np
import glob

data = pd.read_csv(
    "/Users/charlesdedampierre/Desktop/medialab/galaxie de la raison/Annotation Task/2000_followers_graines_version_2021_09_21 (1).csv"
)

# Extract images
# extract_images_from_profile(data)

# pipeline_image_embedding(images_path="downloaded", target_directory="resized_images")

# image_embedding(dir_images="resized_images")
emb = np.load("embeddings.npy")
visualize_images(emb, dir_images="resized_images")

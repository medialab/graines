import os
import pandas as pd
import shutil


def extract_images_from_tweets(tweets: pd.DataFrame):

    """
    This function extract images from information extracted from Twitter. It isolates the urls as a csv 'images_urls.csv'
    and then use minet library (documentation: https://github.com/medialab/minet) to extract the images and store then

    parameter:
        - tweets: pd.DataFrame twitter etract with following columns:
            - media_urls: link toward twitter images

    """

    media = tweets["media_urls"].str.split("|")
    media = media.reset_index()
    media = media.explode("media_urls")
    # drop nan values
    media = media[~media.media_urls.isna()]

    searchfor = ["png", "jpg"]
    images = media[media.media_urls.str.contains("|".join(searchfor))]

    images.to_csv("images_urls.csv")

    # Get images

    # Minet documentation:
    # media_urls is the column name of the urls
    # images_urls.csv the name of the file
    # Example
    # fetch_command = 'minet fetch media_urls {}images_urls.csv --filename id > {}report.csv'.format(destination_path, destination_path)

    # id the column_name of the id name
    print("Number of images: ", len(images))

    # Activate the bash script with the os command
    fetch_command = "minet fetch media_urls images_urls.csv > report.csv"
    os.system(fetch_command)


def extract_images_from_profile(tweets: pd.DataFrame, id_column: str = "screen_name"):
    """
    This function extract profile images from information extracted from Twitter. It isolates the urls as a csv 'images_urls.csv'
    and then use minet library (documentation: https://github.com/medialab/minet) to extract the images and store then

    parameter:
        - tweets: pd.DataFrame twitter etract with following columns:
            - image: link toward twitter images
            - id: id od the corresponding tweet

    Return:
        - Create a pd.DataFrame urls file with all the pictures
        - Create a 'downloaded' directory with the pictures of individual. the name of the picture is the id_column

    """

    if not os.path.exists("image"):
        os.makedirs("image")

    media = tweets[["image", id_column]]
    media = media.reset_index(drop=True)

    searchfor = ["png", "jpg"]
    images = media[media["image"].str.contains("|".join(searchfor))]

    # Save as csv
    images.to_csv("image/images_urls.csv", index=False)
    print("Number of images: ", len(images))

    # Get images using minet
    fetch_command = "minet fetch image image/images_urls.csv --filename {} > image/report.csv".format(
        id_column
    )
    os.system(fetch_command)
    # Move the repo to the image repo
    shutil.move("downloaded", "image")


if __name__ == "__main__":

    # Extract images from Twitter
    data = pd.read_csv("data/data_ready.csv", index_col=[0])
    extract_images_from_profile(tweets=data, id_column="screen_name")

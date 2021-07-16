import pandas as pd
import os

def extract_images_from_tweets(tweets):

	media = tweets["media_urls"].str.split("|")
	media = media.reset_index()
	media = media.explode('media_urls')
	media = media[~media.media_urls.isna()]

	searchfor = ['png', 'jpg']
	images = media[media.media_urls.str.contains('|'.join(searchfor))]

	images.to_csv('images_urls.csv')

	# Get images
	#fetch_command = 'minet fetch media_urls {}images_urls.csv --filename id > {}report.csv'.format(destination_path, destination_path)
	fetch_command = 'minet fetch media_urls images_urls.csv > report.csv'

	# media_urls is the column name of the urls
	# images_urls.csv the name of the file
	# id the column_name of the id name

	print("Number of images: ", len(images))
	os.system(fetch_command)


def extract_images_from_profile(tweets):

	media = tweets["image"]
	media = media.reset_index()

	searchfor = ['png', 'jpg']
	images = media[media['image'].str.contains('|'.join(searchfor))]

	images.to_csv('images_urls.csv')

	# Get images
	fetch_command = 'minet fetch image images_urls.csv --filename id > report.csv'

	# image is the column name of the urls
	# images_urls.csv the name of the file
	# id the column_name of the id name

	print("Number of images: ", len(images))
	os.system(fetch_command)

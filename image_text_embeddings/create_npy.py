import glob
import pandas as pd
import pickle
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Make .npy', add_help=True, conflict_handler='resolve')
parser.add_argument('-i', '--images', required=True, help='path to the image file to get the indexes')
parser.add_argument('-ie', '--image_embeddings', required=True, help='path to the image embedding pickle file')
parser.add_argument('-te', '--text_embeddings', required=True, help='path to the text embedding pickle file')
parser.add_argument('-f', '--file', required=True, help='path to the original twitter file')
args = vars(parser.parse_args())

tweets_path = args['file']
text_embeddings = args['text_embeddings']
image_embeddings = args['image_embeddings']
files = args['images']


if not os.path.exists('final_embeddings'):
    os.mkdir('final_embeddings')

tweets = pd.read_csv(tweets_path, index_col = [0])
all_ids = list(tweets.index) 
files = glob.glob('{}/*'.format(files))
images_id = []
for image_path in files:
    new = image_path.split('content/')[1].split('.')[0]
    images_id.append(new)
    
# Image embeddings
with open('{}'.format(image_embeddings), 'rb') as f:
    image_embeddings = pickle.load(f)
    
df_emb = pd.DataFrame(image_embeddings)
df_emb = df_emb.add_suffix('_im')
df_emb.index = images_id
final_im = pd.merge(tweets, df_emb, left_index=True, right_index=True, how ='left')
size = np.shape(df_emb)[1]
final_im.iloc[:,-size:] = final_im.iloc[:,-size:].fillna(final_im.iloc[:,-size:].mean())
final_im['id'] = final_im.index
final_im = final_im.reset_index(drop=True)

# text embeddings
with open('{}'.format(text_embeddings), 'rb') as f:
    text_embeddings = pickle.load(f) 
text_embeddings = pd.DataFrame(text_embeddings)
text_embeddings = text_embeddings.add_suffix('_text')
tweets = tweets.reset_index()
final_text = pd.concat([tweets[['id']], text_embeddings], axis=1)
final = pd.merge(final_im, final_text, on = 'id')

# deal with Nan values
regex = '\d_text|\d_im'
mean = final.filter(regex='{}'.format(regex), axis=1).mean()
vectors_na = final.filter(regex='{}'.format(regex), axis=1).fillna(mean)
vec_columns = list(vectors_na.columns)
final.loc[:, final.columns.isin(vec_columns)] = vectors_na
final.to_csv('final_embeddings/final_embeddings_text_images.csv')


# Save embeddings
regex = '\d_text|\d_im'
embeddings_c = final.filter(regex='{}'.format(regex), axis=1)
embeddings_c = np.array(embeddings_c)
np.save('final_embeddings/embeddings_images_text.npy', embeddings_c)

regex = '\d_text'
embeddings_text = final.filter(regex='{}'.format(regex), axis=1)
embeddings_text = np.array(embeddings_text)
np.save('final_embeddings/embeddings_text.npy', embeddings_text)

regex = '\d_im'
embeddings_im = final.filter(regex='{}'.format(regex), axis=1)
embeddings_im = np.array(embeddings_im)
np.save('final_embeddings/embeddings_image.npy', embeddings_im)




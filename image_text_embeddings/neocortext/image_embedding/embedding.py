import glob
from PIL import Image
import pickle
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageOps
import pandas as pd
from .utils import getImage

def image_embedding(path_images):

	#Load CLIP model
	model = SentenceTransformer('clip-ViT-B-32')

	pbar = tqdm(total = len(path_images))
	#Encode an image:
	embeddings=[]
	for image in path_images:
	    img_emb = model.encode(Image.open(image))
	    embeddings.append(img_emb)
	    pbar.update(1)

	with open('image_embedding.pickle', 'wb') as f:
		pickle.dump(embeddings, f)

	return embeddings
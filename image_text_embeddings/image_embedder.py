import os 
import pandas as pd
from neocortext.image_embedding.embedding import image_embedding
from neocortext.image_embedding.visualization import visualise
import glob 

# Resize images
path_images = glob.glob('content/*')

# embeddings
embeddings = image_embedding(path_images)

# Create the visualization
visualise(embeddings, path_images, dpi = 100, figsize= (70,50))



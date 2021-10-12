This is a pipeline to execute different actions:
- Twitter Image Extraction
- Embeddings (Bert, tf-idf, Images)
- Classification

Description of the files:
- _bert_embedder.py: takes the dataset, filter the features, concat them and output a np.array
embedding with bert SentenceTransformer
- _tfidf.py: takes the dataset, filter the features, concat them and output a np.array
embedding with bert tidf
- twitter_images.py: extract the images of profiles in twitter and store them in image/downloaded
- image_embedding.py: takes the downloaded images, resize them and ouput a np.array embedding using CLIP
- feature_extract.py: extract the features of intereste, noramlize them an and output a np.array
- creating_x_y.ipynb: concat the embedding, gather the files and clean them and output X & y
- triangular_classifier: takes X and Y and carries out the classification task.

Directories:
- data: original data and the the annotated data
- embeddings: results of the different embeddings (as np.array)
- image: get the downloaded and resized images

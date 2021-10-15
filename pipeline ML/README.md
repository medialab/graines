This is a pipeline to execute different actions:
- Twitter Image Extraction
- Embeddings (Bert, tf-idf, Images)
- Classification

Description of the files:
- _preprocessing.py: merging annotated tasks and features to prepare for embeddings and create the dataset data/data_ready.csv
- _bert_embedder.py: takes the dataset, filter the features, concat them and output a np.array
embedding with bert SentenceTransformer -> (embeddings/bert.npy)
- _tfidf.py: takes the dataset, filter the features, concat them and output a np.array
embedding with bert tidf -> (embeddings/tfidf.npy)
- twitter_images.py: extract the images of profiles in twitter and store them in image/downloaded
- image_embedding.py: takes the downloaded images, resize them and ouput a np.array embedding using CLIP -> (embeddings/full_rpofile_pictures.npy)
- feature_extract.py: extract the features of intereste, noramlize them an and output a np.array -> (embeddings/features.npy)
- triangular_classifier: takes embeddings & label and carries out the classification task. -> output a report

Directories:
- data: original data and the the annotated data
- embeddings: results of the different embeddings (as np.array)
- image: get the downloaded and resized images

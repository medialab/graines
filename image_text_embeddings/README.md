# Image Embeddings
* run `python image_extract.py` to extract images from twitter profiles, from the file: 'graines_et_non_graines.csv'et store them into a 'content/' directory
* run `python image_embedder.py` to read images from the 'content/' directory and output an 'image_embedding.pickle' file, a image.png for visualization of the content, an images_urls.csv file and a report.csv file (the exraction is made thank to the [Minet Library](https://medialab.sciencespo.fr/en/tools/minet/).

# Text Embeddings
* run `python text_embedder.py` to embed text from the 'graines_et_non_graines.csv' file from 'screen_name' and 'text' columns concatenated and output an 'embeddings.pickle' file

# Output .npy files
* run `python create_npy.py -i content -ie 'image_embedding.pickle' -te 'embeddings.pickle' -f graines_et_non_graines.csv` to create 4 files in a 'final_embeddings/' directory:
    - final_embeddings_text_images.csv: the original twitter file with embeddings of text and images
    - embeddings_images_text.npy: vectors of concatenated images and text embeddings
    - embeddings_text.npy: vectors for text embeddings
    - embeddings_images.npy: vectors for image embeddings

# Neocortext
The Neocortext directory contains packages to carry out operations



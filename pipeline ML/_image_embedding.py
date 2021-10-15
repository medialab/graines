import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def resize(
    images_path: str = "image/downloaded",
    size: tuple = (200, 200),
    target_directory: str = "image/images_preprocessed",
):
    """
    This function resizes images save them in a new directory called 'images_preprocessed'
    parameters:
        - images_path: directory that containes the images
        - size: resizing target of the images
        - target_directory: name of the newly created directory

    Output:
        - A new directory that contains resized images
    """
    # Create a new directory

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Resize images
    path_images = glob.glob(images_path + "/*")
    for image in path_images:
        im = Image.open(image)
        imResize = im.resize(size, Image.ANTIALIAS)
        image_name = image.split(images_path)[1]
        imResize.save(target_directory + "/" + image_name)


def image_embedding(dir_images: str) -> np.array:

    """
    This function uses SentenceTransformer and CLIP model (documentation: https://www.sbert.net/examples/applications/image-search/README.html)
    to embed images.

    parameters:
        - dir_images: path of the directory that contains resized images

    Output:
        - image_embedding.npy that contains the embeddings
    """

    # Load CLIP model
    model = SentenceTransformer("clip-ViT-B-32")
    path_images = glob.glob(dir_images + "/*")

    # Encode images:
    pbar = tqdm(total=len(path_images))
    embeddings = []
    for image in path_images:
        img_emb = model.encode(Image.open(image))
        embeddings.append(img_emb)
        pbar.update(1)

    return embeddings


def getImage(path):
    return OffsetImage(plt.imread(path))


def visualize_images(
    embeddings: np.array,
    dir_images: str,
    dpi: int = 150,
    figsize: tuple = (160, 100),
    image_name: str = "images",
):
    """
    This function takes embeddings from images as an impute, carries out 2D dimentionality reduction and output a plot with the images
    on it

    parameters:
        - embeddings: list of image embeddings
        - dir_images: path of the directory that contains the resized images
        - dpi: quality of the rendered image
        - figsize: size of the rendered image
        - image_name: name of the rendered image

    outputs:
        - a .png image

    """
    # 2D reduction
    print("Reduction Algorithm...")
    tsne_data = TSNE(n_components=2).fit_transform(embeddings)
    df = pd.DataFrame(tsne_data)
    df.columns = ["x", "y"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df.x, df.y)

    images_path = glob.glob(dir_images + "/*")
    # Add the images on the plot
    for x0, y0, path in zip(df.x, df.y, images_path):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)

    plt.savefig(image_name + ".png", dpi=dpi)


def pipeline_image_embedding(
    images_path: str, target_directory: str, size: tuple = (200, 200)
):

    """

    This function executes the following pipeline:
            - extract pictures from the coluln containing url of a tweet extract
            - resize pictures
            - embed the resized pictures
            - visualize the results

    parameters:
        - images_path: file of original images
        - target_directory: name of the new directory with resized images
        - size: size of the resized images
    """

    resize(images_path=images_path, size=size, target_directory=target_directory)
    embeddings = image_embedding(target_directory)
    visualize_images(embeddings, path_images=target_directory)


def get_full_image_emb(
    data: pd.DataFrame, emb: np.array, list_ids: list, column_id: str
) -> np.array:

    """
    This function takes the embeddings from images and complet it with the profile
    that did not have any images. It fills the NaN values with the mean of every column

    Args:
        data (pd.DataFrame): data initial
        column_id (str): the name of the column with the ids (also the name on the list_ids)
        emb (np.array): the embeddings of images
        list_ids (list): the list of all the ids whose image has been embedded

    Returns:
        [np.array]: the final embedding of all individuals
    """

    """list_ids = glob.glob("image/images_preprocessed/*")
    list_ids = [
        x.split("image/images_preprocessed/")[1].split(".")[0] for x in list_ids
    ]"""

    data_image = data[[column_id]]
    embedding_data = pd.DataFrame(emb)
    embedding_data[column_id] = list_ids

    fin = pd.merge(data_image, embedding_data, on=column_id, how="left")
    fi_emb = fin.drop(column_id, axis=1)

    # Fill with mean
    fi_emb = fi_emb.fillna(fi_emb.mean())

    return fi_emb


if __name__ == "__main__":

    resize(images_path="image/downloaded")
    emb = image_embedding("image/images_preprocessed")
    np.save("embeddings/profile_pictures.npy", emb)

    # Get the full embeddings
    data = pd.read_csv("data/data_ready.csv")
    list_ids = glob.glob("image/images_preprocessed/*")
    list_ids = [
        x.split("image/images_preprocessed/")[1].split(".")[0] for x in list_ids
    ]

    emb = np.load("embeddings/profile_pictures.npy")
    fi_emb = get_full_image_emb(data, emb, list_ids, column_id="screen_name")
    np.save("embeddings/full_profile_pictures.npy", fi_emb)

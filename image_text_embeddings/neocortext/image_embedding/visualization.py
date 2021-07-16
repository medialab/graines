
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def getImage(path):
    return OffsetImage(plt.imread(path))

def visualise(embeddings, path_images, dpi = 300, figsize=(160,100)):
	'''from embedding to images'''

	print('Reduction Algorithm...')
	tsne_data = TSNE(n_components=2).fit_transform(embeddings)
	df = pd.DataFrame(tsne_data)
	df.columns = ['x', 'y']

	fig, ax = plt.subplots(figsize = figsize)
	ax.scatter(df.x, df.y) 

	for x0, y0, path in zip(df.x, df.y, path_images):
	    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
	    ax.add_artist(ab)

	plt.savefig('image.png', dpi=dpi)

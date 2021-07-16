import glob
import os
from PIL import Image

def resize(images_path = 'content/',size = (200,200)):
	'''Go to the directory names content and resize all images
				Put them in a directory called images_preprocessed'''

	os.makedirs('images_preprocessed')

	# resize images
	path_images = glob.glob(images_path + '*')
	for image in path_images:
	    im = Image.open(image)
	    imResize = im.resize(size, Image.ANTIALIAS)
	    image_name = image.split(images_path)[1]
	    imResize.save('images_preprocessed/'+ image_name)

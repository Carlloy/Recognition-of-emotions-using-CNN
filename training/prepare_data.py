import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

fer_path = 'Recognition-of-emotions-using-CNN/training/dataset/fer2013.csv'
width, height = 48, 48
data_generator = ImageDataGenerator(
	featurewise_center=False,
	featurewise_std_normalization=False,
	rotation_range=10,
	width_shift_range=0.1,
	height_shift_range=0.1,
	zoom_range=.1,
	horizontal_flip=True)


def load_data():
	data = pd.read_csv(fer_path)
	pixels = data['pixels'].tolist()
	images = []
	emotions_labels = pd.get_dummies(data['emotion']).as_matrix()
	for img in pixels:
		image = [int(pixel) for pixel in img.split(' ')]
		image = np.asarray(image).reshape(width, height)
		images.append(image.astype('float32'))
	images = np.asarray(images) / 255.0
	images = np.expand_dims(images, -1)
	return images, emotions_labels

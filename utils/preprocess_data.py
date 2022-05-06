import pandas as pd
import pickle
import json
import pdb
import warnings
import os
import gzip
import numpy as np

from pathlib import Path
from utils.feature_tools import FeatureTools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

## Using tensorflow gpu
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


warnings.filterwarnings("ignore")


def load_new_training_data(path):
	data = []
	with open(path, "r") as f:
		for line in f:
			data.append(json.loads(line))
	return pd.DataFrame(data)

def load_data(path_dict):
	"""Load MNIST data from `path`"""

	images_path = path_dict.get('images')
	labels_path = path_dict.get('labels')

	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
								offset=8)

	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype=np.uint8,
								offset=16).reshape(len(labels), 784)

	return images, labels


def build_train(train_path_dict, results_path, IMAGE_SHAPE, dataprocessor_id=0, PATH_2=None):

	X_train, y_train = load_data(train_path_dict)
	if PATH_2:
		df_tmp = load_new_training_data(PATH_2)
		X_train = np.append(X_train, df_tmp.image)
		y_train = np.append(y_train, df_tmp.label)
	
	X_train, X_validate, y_train, y_validate = train_test_split(X_train,y_train,test_size = 0.2,random_state = 12345)
	X_train = X_train.reshape(X_train.shape[0],*IMAGE_SHAPE)
	X_validate = X_validate.reshape(X_validate.shape[0],*IMAGE_SHAPE)

	processed_data = [X_train, y_train, X_validate, y_validate]

	dataprocessor_fname = 'dataprocessor_{}_.p'.format(dataprocessor_id)
	pickle.dump(processed_data, open(results_path/dataprocessor_fname, "wb"))

	return processed_data



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
	# target = 'income_label'
	# # read initial DataFrame
	# df = pd.read_csv(train_path)
	# if PATH_2:
	# 	df_tmp = load_new_training_data(PATH_2)
	# 	# Let's make sure columns are in the same order
	# 	df_tmp = df_tmp[df.columns]
	# 	# append new DataFrame
	# 	df = pd.concat([df, df_tmp], ignore_index=True)
	# 	# Save it to disk
	# 	df.to_csv(train_path, index=False)

	# df[target] = (df['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
	# df.drop('income_bracket', axis=1, inplace=True)

	# categorical_columns = list(df.select_dtypes(include=['object']).columns)
	# numerical_columns = [c for c in df.columns if c not in categorical_columns+[target]]
	# crossed_columns = (['education', 'occupation'], ['native_country', 'occupation'])

	# preprocessor = FeatureTools()
	# dataprocessor = preprocessor.fit(
	# 	df,
	# 	target,
	# 	numerical_columns,
	# 	categorical_columns,
	# 	crossed_columns,
	# 	sc=MinMaxScaler()
	# 	)

	X_train, y_train = load_data(train_path_dict, kind='train')
	# X_test, y_test = load_data('fashion_mnist/data/fashion', kind='t10k')
	X_train, X_validate, y_train, y_validate = train_test_split(X_train,y_train,test_size = 0.2,random_state = 12345)
	X_train = X_train.reshape(X_train.shape[0],*IMAGE_SHAPE)
	X_test = X_test.reshape(X_test.shape[0],*IMAGE_SHAPE)
	X_validate = X_validate.reshape(X_validate.shape[0],*IMAGE_SHAPE)

	processed_data = [X_train, y_train, X_validate, y_validate]

	dataprocessor_fname = 'dataprocessor_{}_.p'.format(dataprocessor_id)
	pickle.dump(processed_data, open(results_path/dataprocessor_fname, "wb"))
	# if dataprocessor_id==0:
	# 	pickle.dump(df.columns.tolist()[:-1], open(results_path/'column_order.p', "wb"))

	return processed_data


# if __name__ == '__main__':

# 	PATH = Path('data/')
# 	TRAIN_PATH = PATH/'train'
# 	DATAPROCESSORS_PATH = PATH/'dataprocessors'

# 	dataprocessor = build_train(TRAIN_PATH/'train.csv', DATAPROCESSORS_PATH)

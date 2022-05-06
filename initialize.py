import numpy as np
import pandas as pd
import pickle
import warnings
import argparse

from pathlib import Path
from utils.preprocess_data import build_train

from train.train_cnn_model import CNNModel


PATH = Path('data/')
DATA_PATH = '{}/fashion'.format(PATH)
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'

IMAGE_ROWS = 28
IMAGE_COLS = 28
BATCH_SIZE = 4096
IMAGE_SHAPE = (IMAGE_ROWS, IMAGE_COLS, 1) 
TRAIN_EPOCHS = 25

CLASS_LABELS = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def create_folders():
	print("creating directory structure...")
	(PATH).mkdir(exist_ok=True)
	(MODELS_PATH).mkdir(exist_ok=True)
	(DATAPROCESSORS_PATH).mkdir(exist_ok=True)
	(MESSAGES_PATH).mkdir(exist_ok=True)

def create_data_processor():
	print("creating preprocessor...")
	path_dict = dict()
	path_dict['images'] = '{}/train-images-idx3-ubyte.gz'.format(DATA_PATH)
	path_dict['labels'] = '{}/train-labels-idx1-ubyte.gz'.format(DATA_PATH)
	dataprocessor = build_train(path_dict, DATAPROCESSORS_PATH, IMAGE_SHAPE)


def create_model(hyper):
	print("creating model...")
	init_dataprocessor = 'dataprocessor_0_.p'
	dtrain = pickle.load(open(DATAPROCESSORS_PATH/init_dataprocessor, 'rb'))
	cnn_model = CNNModel(dtrain, MODELS_PATH, IMAGE_SHAPE, BATCH_SIZE, TRAIN_EPOCHS)
	cnn_model.fit()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--hyper", type=str, default="hyperopt")
	args = parser.parse_args()
	create_folders()
	create_data_processor()
	create_model(args.hyper)
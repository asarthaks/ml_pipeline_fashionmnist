import pdb
import json
import pandas as pd
import pickle
import argparse

from pathlib import Path
from kafka import KafkaConsumer

from utils.messages_utils import publish_traininig_completed
from utils.preprocess_data import build_train
from initialize import IMAGE_SHAPE, BATCH_SIZE, TRAIN_EPOCHS
from train.train_cnn_model import CNNModel


KAFKA_HOST = 'localhost:9092'
RETRAIN_TOPIC = 'retrain_topic'
PATH = Path('data/')
TRAIN_DATA = PATH/'train/train.csv'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'


def train(model_id, messages, hyper):
	print("RETRAINING STARTED (model id: {})".format(model_id))
	dtrain = build_train(TRAIN_DATA, DATAPROCESSORS_PATH, model_id, messages)
	cnn_model = CNNModel(dtrain, MODELS_PATH, IMAGE_SHAPE, BATCH_SIZE, TRAIN_EPOCHS)
	cnn_model.fit()
	print("RETRAINING COMPLETED (model id: {})".format(model_id))


def start(hyper):
	consumer = KafkaConsumer(RETRAIN_TOPIC, bootstrap_servers=KAFKA_HOST)

	for msg in consumer:
		message = json.loads(msg.value)
		if 'retrain' in message and message['retrain']:
			model_id = message['model_id']
			batch_id = message['batch_id']
			message_fname = 'messages_{}_.txt'.format(batch_id)
			messages = MESSAGES_PATH/message_fname

			train(model_id, messages, hyper)
			publish_traininig_completed(model_id)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--hyper", type=str, default="hyperopt")
	args = parser.parse_args()

	start(args.hyper)
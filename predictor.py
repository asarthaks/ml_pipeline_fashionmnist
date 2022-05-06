import pdb
import json
import pandas as pd
import pickle
import numpy as np

from pathlib import Path
from kafka import KafkaConsumer
from utils.messages_utils import append_message, read_messages_count, send_retrain_message, publish_prediction
from tensorflow.keras.models import load_model
import base64

KAFKA_HOST = 'localhost:9092'
TOPICS = ['app_messages', 'retrain_topic']
PATH = Path('data/')
MODELS_PATH = PATH/'models'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MESSAGES_PATH = PATH/'messages'
RETRAIN_EVERY = 25
EXTRA_MODELS_TO_KEEP = 1

from initialize import IMAGE_SHAPE, IMAGE_COLS, IMAGE_ROWS

# column_order = pickle.load(open(DATAPROCESSORS_PATH/'column_order.p', 'rb'))
dataprocessor = None
consumer = None
model = None


def reload_model(path):
	print('reloading model..')
	return load_model(path)


def is_retraining_message(msg):
	message = json.loads(msg.value)
	return msg.topic == 'retrain_topic' and 'training_completed' in message and message['training_completed']


def is_application_message(msg):
	message = json.loads(msg.value)
	return msg.topic == 'app_messages' and 'prediction' not in message


def predict(message, IMAGE_SHAPE):
	image_data = np.array(message['data']['image'])
	# row = pd.DataFrame(message, index=[0])
	# sanity check
	assert image_data.shape == (IMAGE_COLS*IMAGE_ROWS,)
	# In the real world we would not have the target (here 'income_bracket').
	# In this example we keep it and we will retrain the model as it reads
	# RETRAIN_EVERY number of messages. In the real world, after RETRAIN_EVERY
	# number of messages have been collected, one would have to wait until we
	# can collect RETRAIN_EVERY targets AND THEN retrain
	# row.drop('income_bracket', axis=1, inplace=True)
	# trow = dataprocessor.transform(row)
	trow = image_data.reshape(*IMAGE_SHAPE)
	return model.predict(trow)[0]


def start(model_id, messages_count, batch_id):
	print('here')
	consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST, auto_offset_reset='earliest', group_id=None)
	consumer.subscribe(TOPICS)
	print(consumer)
	for msg in consumer:
		print(msg)
		message = json.loads(msg.value)

		if is_retraining_message(msg):
			model_fname = 'model_{}_.h5'.format(model_id)
			model = reload_model(MODELS_PATH/model_fname)
			print("NEW MODEL RELOADED {}".format(model_id))

		elif is_application_message(msg):
			request_id = message['request_id']
			pred = predict(message['data'], IMAGE_SHAPE)
			publish_prediction(pred, request_id)

			append_message(message['data'], MESSAGES_PATH, batch_id)
			messages_count += 1
			if messages_count % RETRAIN_EVERY == 0:
				model_id = (model_id + 1) % (EXTRA_MODELS_TO_KEEP + 1)
				send_retrain_message(model_id, batch_id)
				batch_id += 1


if __name__ == '__main__':
	dataprocessor_id = 0
	dataprocessor_fname = 'dataprocessor_{}_.p'.format(dataprocessor_id)
	dataprocessor = pickle.load(open(DATAPROCESSORS_PATH/dataprocessor_fname, 'rb'))

	messages_count = read_messages_count(MESSAGES_PATH, RETRAIN_EVERY)
	batch_id = messages_count % RETRAIN_EVERY

	model_id = batch_id % (EXTRA_MODELS_TO_KEEP + 1)
	model_fname = 'model_{}_.h5'.format(model_id)
	model = reload_model(MODELS_PATH/model_fname)

	# consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
	# consumer.subscribe(TOPICS)

	start(model_id, messages_count, batch_id)

import pandas as pd
import json
import threading
import uuid

from pathlib import Path
from kafka import KafkaProducer, KafkaConsumer
from time import sleep

from utils.preprocess_data import load_data
from initialize import DATA_PATH
import base64

PATH = Path('data/')
KAFKA_HOST = 'localhost:9092'

test_data_path_dict = dict()
test_data_path_dict['images'] = '{}/t10k-images-idx3-ubyte.gz'.format(DATA_PATH)
test_data_path_dict['labels'] = '{}/t10k-labels-idx1-ubyte.gz'.format(DATA_PATH)
X_test, y_test = load_data(test_data_path_dict)
# In the real world, the messages would not come with the target/outcome of
# our actions. Here we will keep it and assume that at some point in the
# future we can collect the outcome and monitor how our algorithm is doing
# df_test['json'] = df_test.apply(lambda x: x.to_json(), axis=1)
# messages = df_test.json.tolist()


def start_producing():
	producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)
	for i in range(200):
		message_id = str(uuid.uuid4())
		message = {'request_id': message_id, 'data': {'image' : [int(item) for item in X_test[i].tolist()], 'label' :int(y_test[i])}}
		# print(message)
		_ = producer.send('app_messages', json.dumps(message).encode('utf-8'))
		print(_)
		producer.flush()

		print("\033[1;31;40m -- PRODUCER: Sent message with id {}".format(message_id))
		sleep(2)


def start_consuming():
	consumer = KafkaConsumer('app_messages', bootstrap_servers=KAFKA_HOST)

	for msg in consumer:
		message = json.loads(msg.value)
		if 'prediction' in message:
			request_id = message['request_id']
			print("\033[1;32;40m ** CONSUMER: Received prediction {} for request id {}".format(message['prediction'], request_id))


threads = []
t = threading.Thread(target=start_producing)
t2 = threading.Thread(target=start_consuming)
threads.append(t)
threads.append(t2)
t.start()
t2.start()

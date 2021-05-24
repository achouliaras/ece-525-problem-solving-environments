import json
import threading
import uuid
from time import sleep

import matplotlib.pyplot as plt
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from sklearn.metrics import accuracy_score

from utils.config import *

df_test = pd.read_csv(PATH / 'streaming_data.csv')
df_test['json'] = df_test.apply(lambda x: x.to_json(), axis=1)

y_test = df_test["label"]


def start_producing():
	producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)
	print("Sending a batch of {0} new rows".format(RETRAIN_EVERY))

	for index, row in df_test.iterrows():
		message_id = str(uuid.uuid4())
		message = {'request_id': message_id, 'data': json.loads(row['json'])}

		producer.send('app_messages', json.dumps(message).encode('utf-8'))
		producer.flush()
		if index==1500:
			print("a")
		if index == 1950:
			print("b")
		#print("\033[1;31;40m ---- PRODUCER: Requesting prediction for '{0}'".format(row['url']))
		if (index + 1) % RETRAIN_EVERY == 0:
			print("Done. Waiting for 5 secs.")
			sleep(5)
			print("Sending a batch of {0} new rows".format(RETRAIN_EVERY))
	return


def start_consuming():
	consumer = KafkaConsumer('app_messages', bootstrap_servers=KAFKA_HOST)
	y_pred = []
	for msg in consumer:
		message = json.loads(msg.value)
		if 'prediction' in message:
			#print("\033[1;32;40m **** CONSUMER: Received prediction {0} for url {1}".format(message['prediction'], message['url']))
			y_pred.append(message['prediction'])
	acc = []
	for i in range(20, len(y_pred), 1):
		acc.append(accuracy_score(y_test[:i], y_pred[:i]) * 100)
	print("plot!!")
	plt.plot(range(len(acc)), acc, 'g-')
	plt.xlabel('time (s)')
	plt.ylabel('Accuracy (%)')
	plt.title('About as simple as it gets, folks')
	plt.grid(True)
	plt.show()

	KafkaConsumer.close()
	return


threads = []
t = threading.Thread(target=start_producing)
t2 = threading.Thread(target=start_consuming)
t.daemon = True
t2.daemon = True
threads.append(t)
threads.append(t2)
t.start()
t2.start()

while True:
	sleep(1)

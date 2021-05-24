import json

import numpy as np
import lightgbm as lgb
from gensim.models import FastText
from kafka import KafkaConsumer
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.messages_utils import append_message, is_application_message, is_retraining_message, publish_prediction, read_messages_count, send_retrain_message
from utils.config import *
from utils.word_embeddings import get_sentence_embedding

model = None
model_fast_text = None

#overall_acc = []


def calculate_metrics(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	ac = accuracy_score(y_true, y_pred)
	return cm, ac


def predict(message, model):
	global model_fast_text
	url = message['url']

	turl = get_sentence_embedding(url, model_fast_text)

	turl = turl.reshape(1, turl.shape[0])

	pred = model.predict(turl)
	pred = np.around(pred)
	return int(pred[0])


def start(messages_count, batch_id):
	global model
	batch = 0
	last_batch_preds = []
	last_batch_true_y = []
	for msg in consumer:
		message = json.loads(msg.value)

		if is_retraining_message(msg):
			model = lgb.Booster(params, model_file=str(MODELS_PATH / MODEL_NAME_LGB))
			cm, ac = calculate_metrics(last_batch_true_y, last_batch_preds)
			print("Batch {0} confusion_matrix".format(batch))
			print(cm)
			print("Batch {0} accuracy:{1}".format(batch, ac * 100))

			#with open('acc.txt','a') as f:
			#	f.write(str(ac)+",")
			batch = batch + 1
			last_batch_preds = []
			last_batch_true_y = []
			print("New model reloaded!")

		elif is_application_message(msg):
			request_id = message['request_id']
			pred = predict(message['data'], model)

			last_batch_preds.append(pred)
			last_batch_true_y.append(message['data']['label'])

			publish_prediction(pred, message['data']['url'], request_id)

			append_message(message['data'], MESSAGES_PATH, batch_id)
			messages_count += 1
			if messages_count % RETRAIN_EVERY == 0:
				send_retrain_message(batch_id)
				batch_id += 1


if __name__ == '__main__':

	print("Getting ready...")

	messages_count = read_messages_count(MESSAGES_PATH, RETRAIN_EVERY)
	batch_id = messages_count % RETRAIN_EVERY

	model = lgb.Booster(params, model_file=str(MODELS_PATH / MODEL_NAME_LGB))
	model_fast_text = FastText.load(str(MODELS_PATH / MODEL_NAME_FAST_TEXT))

	consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
	consumer.subscribe(TOPICS)

	print("Done!")

	start(messages_count, batch_id)

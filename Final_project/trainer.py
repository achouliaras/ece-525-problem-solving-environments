import json

import pandas as pd
import lightgbm as lgb
from gensim.models import FastText
from kafka import KafkaConsumer

from utils.messages_utils import publish_training_completed, is_application_message
from utils.config import *
from utils.word_embeddings import transform_text_to_embedding

model_fast_text = None


def load_new_training_data(path):
	data = []
	with open(path, "r") as f:
		for line in f:
			data.append(json.loads(line))
	return pd.DataFrame(data)


def train(messages):
	global model_fast_text
	print("RETRAINING STARTED")

	df_tmp = load_new_training_data(messages)  # append received messages to the initial DataFrame
	y = df_tmp["label"]
	url_list = df_tmp["url"]

	X = transform_text_to_embedding(url_list, model_fast_text)

	lgb_train = lgb.Dataset(X, y)

	model = lgb.train(params, lgb_train, num_boost_round=num_iterations//2, keep_training_booster=True)
	model.save_model(str(MODELS_PATH / MODEL_NAME_LGB))

	print("RETRAINING COMPLETED")


def start():
	consumer = KafkaConsumer(RETRAIN_TOPIC, bootstrap_servers=KAFKA_HOST)

	for msg in consumer:
		message = json.loads(msg.value)
		if 'retrain' in message and message['retrain']:
			batch_id = message['batch_id']
			message_fname = 'messages_{}_.txt'.format(batch_id)
			messages = MESSAGES_PATH / message_fname

			train(messages)
			publish_training_completed()

if __name__ == '__main__':
	print("Getting ready...")
	model_fast_text = FastText.load(str(MODELS_PATH / MODEL_NAME_FAST_TEXT))
	print("Done!")
	start()

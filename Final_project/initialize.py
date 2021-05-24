import shutil

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import FastText

from utils.word_embeddings import Tokenizer, transform_text_to_embedding
from utils.config import *


def create_folders(data, delete=False):
	print("Creating directory structure...")

	shutil.rmtree(TRAIN_PATH, ignore_errors=True)
	shutil.rmtree(MESSAGES_PATH, ignore_errors=True)

	if delete:
		shutil.rmtree(MODELS_PATH, ignore_errors=True)

	TRAIN_PATH.mkdir(exist_ok=True)
	MODELS_PATH.mkdir(exist_ok=True)
	MESSAGES_PATH.mkdir(exist_ok=True)

	data.to_csv(TRAIN_PATH / "train.csv", index=False)


def create_word_embeddings(data):
	print("Generating initial word embeddings...")

	url_txt = data['url'].values

	model_fast_text = FastText(size=vector_size, workers=8)
	model_fast_text.build_vocab(sentences=Tokenizer(url_txt))
	model_fast_text.train(sentences=Tokenizer(url_txt), total_examples=model_fast_text.corpus_count, total_words=model_fast_text.corpus_total_words, epochs=epochs)

	model_fast_text.save(str(MODELS_PATH / MODEL_NAME_FAST_TEXT))

	print("Word embeddings generated")


def create_model(data):
	print("Training initial model...")
	# statistics
	urls = data.count().values[0]
	mal_url = data[data.label == 1].count().values[0]
	safe_url = data[data.label == 0].count().values[0]

	print('Total number of Urls: {0}'.format(urls))
	print('Number of Malicious Urls: {0}'.format(mal_url))
	print('Number of Safe Urls: {0}'.format(safe_url))
	print('Percentage of Malicious Urls: {0:.2f}'.format(100 * mal_url / urls))
	print('Percentage of Safe Urls: {0:.2f}'.format(100 * safe_url / urls))

	y = data["label"]
	url_list = data["url"]

	model_fast_text = FastText.load(str(MODELS_PATH / MODEL_NAME_FAST_TEXT))

	X = transform_text_to_embedding(url_list, model_fast_text)

	X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.05)

	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

	model = lgb.train(params, lgb_train, num_boost_round=num_iterations, valid_sets=lgb_eval, keep_training_booster=True)

	model.save_model(str(MODELS_PATH / MODEL_NAME_LGB))
	model.save_model(str(MODELS_PATH / ('initial_'+MODEL_NAME_LGB)))

	print("Initial model trained")


if __name__ == '__main__':


	reinitialize = False
	data = pd.read_csv(PATH / "initial_data.csv")

	if Path(MODELS_PATH / "initialized").is_file():
		while True:
			choice = input("Reinitialize?(y/n)").lower()
			if choice == "y":
				reinitialize = True
				break
			elif choice == "n":
				reinitialize = False
				break

		if reinitialize:
			create_folders(data, True)
			create_word_embeddings(data)
			create_model(data)
		else:
			create_folders(data, False)
	else:
		create_folders(data, True)
		create_word_embeddings(data)
		create_model(data)
		open((MODELS_PATH / "initialized"), 'w').close()

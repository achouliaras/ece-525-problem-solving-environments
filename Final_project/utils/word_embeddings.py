from gensim.utils import tokenize
import numpy as np

from utils.config import *


class Tokenizer:
	def __init__(self, data):
		self.data = data

	def __iter__(self):
		for line in self.data:
			yield list(tokenize(line))


def get_sentence_embedding(sentence, model):
	words = list(tokenize(sentence))
	if words:
		vectors = model.wv[words]
		return np.mean(vectors / np.sqrt(np.einsum('...i,...i', vectors, vectors))[:, np.newaxis], axis=0)
	else:
		return np.zeros(vector_size)


def transform_text_to_embedding(sentences, model):
	result = []

	for sentence in sentences:
		result.append(get_sentence_embedding(sentence, model))

	return np.array(result)

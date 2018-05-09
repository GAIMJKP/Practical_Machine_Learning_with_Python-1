import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  # Elimina ing ed para evitar run running para este caso no nos interesa el significado
import numpy as np
import random
import pickle
from collections import Counter


'''
lexicon ->[chair, table, spoon, television]
Sentence -> I pull the chair up to the table
np.zeros(len(lexicon))
-> array to the sentence [1 1 0 0]
'''

lemmatizer = WordNetLemmatizer()
# MemoryError
hm_lines = 100000


def create_lexicon(pos, neg):
	lexicon = []
	for fi in [pos, neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)


	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	# w_counts = {'the':52521, 'and':25153} Será un diccionario de esta forma
	l2 = []
	# Queremos quitar palabras muy comunes como the y and ni las muy raras que no ocurren
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)
	print(len(l2))
	return l2


def sample_handling(sample, lexicon, classification):
	featureset = []
	''' Estructura del featuresset
	[
	[[0 1 0 1 1 0], [0 1]] --> lexicon y pos y neg [1 0] pos [0 1] neg

	]
	'''
	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append([features, classification])
	return featureset


def create_features_set_and_lables(pos, neg, test_size=0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling(pos, lexicon, [1, 0])
	features += sample_handling(neg, lexicon, [0, 1])
	random.shuffle(features)

	features = np.array(features)

	testing_size = int(test_size * len(features))


	train_x = list(features[:, 0][:-testing_size])
	# take all the 0 elements features = [features,labels] --> [[0 1 0 1 1 0], [0 1]] -> 0 serian todas las features hasta ultimo 10%
	# [[5,8],
	# [7,9]]
# ---> [5,7]

	train_y = list(features[:, 1][:-testing_size])

	test_x = list(features[:, 0][-testing_size:])  # El ultimo 10%, desde -testing size hasta el final
	test_y = list(features[:, 1][-testing_size:])

	return train_x, train_y, test_x, test_y


if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_features_set_and_lables('./data/pos.txt', './data/neg.txt')
	with open('./data/sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)






















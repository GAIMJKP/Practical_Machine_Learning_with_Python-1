from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd



lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

# Procesamos los archivos csv y creamos otros con lo que necesitamos
# Dataset utlizado http://help.sentiment140.com/for-students/

def init_process(fin, fout):
	n_line = 1

	outfile = open(fout, 'a')
	with open(fin, buffering=200000, encoding='latin-1', errors="replace") as f:
		try:
			for line in  f:
				print(n_line)
				line = line.replace('"', '')
				initial_polarity = line.split(',')[0]
				# Nos quedamos unicamente con los positivos y negativos para que sea igual que el ejemplo anterior
				if initial_polarity == '0':
					initial_polarity = [1, 0, 0] # Pos
				elif initial_polarity == '4':
					initial_polarity = [0, 1, 0] # Neg
				elif initial_polarity == '2':
					initial_polarity = [0, 0, 1] # Neutral
				# Nos quedamos con la polaridad y el tweet, fechas no nos importan
				tweet = line.split(',')[-1]
				outline = str(initial_polarity) + ':::' + tweet
				outfile.write(outline)
				n_line += 1
		except Exception as e:
			print(str(e)+str(n_line))


	outfile.close()



# Esto solo es necesario realizarlo la primera vez
init_process('./data/training.1600000.processed.noemoticon.csv', './data/train_set.csv')
init_process('./data/testdata.manual.2009.06.14.csv', './data/test_set.csv')


# Crea una lista con las palabras utiles
def create_lexicon(fin):
	lexicon = []
	with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
		try:
			counter = 1
			content = ''
			for line in f:
				counter += 1
				# Cogemos una muestra aleatoria cada 2500 elementos para crear el lexicon
				if (counter / 2500.0).is_integer():
					tweet = line.split(':::')[1]
					content += ' ' + tweet
					words = word_tokenize(content)
					words = [lemmatizer.lemmatize(i) for i in words]
					lexicon = list(set(lexicon + words))
					print(counter, len(lexicon))

		except Exception as e:
			print(str(e))

	with open('./data/lexicon.pickle', 'wb') as f:
		pickle.dump(lexicon, f)


# Unicamente es necesario ejecutarlo una vez y si necesitamos el lexicon creado
create_lexicon('./data/train_set.csv')


# Inviable si tenemos un dataset suficientemente grande 20gb+
def convert_to_vec(fin, fout, lexicon_pickle):
	with open(lexicon_pickle, 'rb') as f:
		lexicon = pickle.load(f)
	outfile = open(fout, 'a')
	with open(fin, buffering=20000, encoding='latin-1') as f:
		counter = 0
		for line in f:
			counter += 1
			label = line.split(':::')[0]
			tweet = line.split(':::')[1]
			current_words = word_tokenize(tweet.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]

			features = np.zeros(len(lexicon))

			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					# OR DO +=1, test both
					features[index_value] += 1

			features = list(features)
			outline = str(features) + '::' + str(label) + '\n'
			outfile.write(outline)
		print(counter)
convert_to_vec('./data/test_set.csv', './data/processed-test-set.csv', './data/lexicon.pickle')


def shuffle_data(fin):
	df = pd.read_csv(fin, error_bad_lines=False)
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv('./data/train_set_shuffled.csv', index=False)


shuffle_data('./data/train_set.csv')


# Guarda en picke la array con los valores para el test
def create_test_data_pickle(fin):
	feature_sets = []
	labels = []
	counter = 0
	with open(fin, buffering=20000) as f:
		for line in f:
			try:
				features = list(eval(line.split('::')[0]))
				label = list(eval(line.split('::')[1]))

				feature_sets.append(features)
				labels.append(label)
				counter += 1
			except:
				pass
	print(counter)
	feature_sets = np.array(feature_sets)
	labels = np.array(labels)


create_test_data_pickle('./data/processed-test-set.csv')

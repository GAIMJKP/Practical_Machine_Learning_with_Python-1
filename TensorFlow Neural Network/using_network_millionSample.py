
import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 3
#hm_data = 2000000

batch_size = 32
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')


current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([729, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

    return output

#saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./data/model.ckpt.meta')

def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('./data/lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess, "./data/model.ckpt")
        except Exception as e:
            print(str(e))
        # saver.restore(sess,"./data/model.ckpt")
        
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0,0] , argmax: 0
        # neg: [0,1,0] , argmax: 1
        # neutral: [0,0,1], argmax 2?
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        
        print(sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))

        if result[0] == 0:
            print('Positive:',input_data, result)
        elif result[0] == 1:
            print('Negative:',input_data, result)

use_neural_network("He's an idiot and a jerk.")
use_neural_network("This was the best store i've ever seen.")
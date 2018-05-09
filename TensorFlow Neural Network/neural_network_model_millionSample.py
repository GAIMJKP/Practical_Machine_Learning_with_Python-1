import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 3
load_model = True

# Numero de 'lineas' que va a tener cada batch
batch_size = 32

#Esto no sería así, una vez procesado y quitado el neutro se nos quedan 600k~~
total_batches = int(1600000 / batch_size)
hm_epochs = 10

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# La entrada sera una array con 32 valores podemos comprobarlo en el lexicon
hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([729, n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes]))}


def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
    return output


saver = tf.train.Saver()
tf_log = 'tf.log'


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            if load_model == True:
                epoch = int(open(tf_log, 'r').read().split('\n')[-2]) + 1
                print('STARTING:', epoch)
            else:
                epoch = 1
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "./data/model.ckpt")
            epoch_loss = 1
            with open('./data/lexicon.pickle', 'rb') as f:
                lexicon = pickle.load(f)
            with open('./data/train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
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
                    #print(len(lexicon))  # 32 Palabras en el lexicon
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    #Va a ir iterando linea a linea y almacenando datos en el batch
                    #Cuando el batch sea mayor que el tamaño lo ejecuta y vacia el batch
                    if len(batch_x) >= batch_size:
                        #print(np.array(batch_x).shape)
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x), y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch, '| Batch Loss:', c,)

            saver.save(sess, "./data/model.ckpt")
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            with open(tf_log, 'a') as f:
                f.write(str(epoch) + '\n')
            epoch += 1


train_neural_network(x)


def test_neural_network(x):
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess, "./data/model.ckpt")
            except Exception as e:
                print(str(e))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('./data/processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested', counter, 'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


test_neural_network(x)

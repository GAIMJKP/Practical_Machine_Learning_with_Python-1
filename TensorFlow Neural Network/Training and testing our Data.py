import tensorflow as tf
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #to avoid warning in windows
from Preprocesing_Data_Sentiment_Featureset import create_features_set_and_lables
import numpy as np

#Mismo codigo que en la parte 6-2 pero adaptado

##Arreglar esto

train_x,train_y,test_x,test_y = create_features_set_and_lables('./data/pos.txt', './data/neg.txt')


#Numero de nodos para este ejemplo
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
#Numero de ejemplos por bloque que van a pasar por la red (memory)
batch_size = 100
# cycles feed forward + backprop
hm_epochs = 10

x = tf.placeholder('float', [None, len(train_x[0])]) 
y = tf.placeholder('float')

def neural_network_model(data):

    #(input data * weights) + biasses
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    #l1 = tf.nn.dropout(l1, keep_prob) Podriamos añadir dropout

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    #One_hot Array
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # compara la salida que predice con la que debería ser
    # Queremos minimizar el coste
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # deault learning_Rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #Traninng the nerwork
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end=i+batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])         

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y}) 
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        
        #Devuelve el index del valor maximo en las arrays y esperamos que sean iguales
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
    

train_neural_network(x)
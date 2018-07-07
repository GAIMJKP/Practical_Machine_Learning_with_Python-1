import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #to avoid warning in windows
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


# 10 classes, 0-9

'''
One_hot encoder do
0= [1,0,0,0,0,0,0,0,0,0]
1= [0,1,0,0,0,0,0,0,0,0]
.
.
en lugar de de enteros 1,2,3,4,5



input > weigth > hidden layer 1 (act function) > weigth > hidden l 2 (act func) > weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost(AdamOptimizer.........SGD, AdaGrad)

backpropagation
feed foward + backprop = epoch
'''


#Numero de nodos para este ejemplo
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
#Numero de ejemplos por bloque que van a pasar por la red (memory)
batch_size = 100

# height x width, lo convierte en una array de una sola dimension de 784 elementos
x = tf.placeholder('float', [None, 784]) #28*28 pixels, [None, 784] No es necesario pero si no lo ponemos y metemos otras dimensiones no nos dara error
y = tf.placeholder('float')

def neural_network_model(data):

    #(input data * weights) + biasses
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
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
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    # deault learning_Rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

    # cycles feed forward + backprop
    hm_epochs = 10
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        #Traninng the nerwork
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): #how many times we need to cycle to cover all data
                #Separa el dataset en batchs
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}) #tensorflow automaticamente actualiza los weights y bias
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        
        #Devuelve el index del valor maximo en las arrays y esperamos que sean iguales
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    

train_neural_network(x)
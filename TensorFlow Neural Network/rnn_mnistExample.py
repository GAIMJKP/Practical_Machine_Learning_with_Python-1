import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


n_classes = 10
hm_epochs = 3
batch_size = 128
chuck_size = 28 # Mnist Images 28x28
n_chucks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chucks,chuck_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    #Explicacion al final del codigo
    # Convierte cada fila en una array
    x = tf.transpose(x, [1, 0, 2])
    # Transforma para que cada valor de sea del nº de chunks [[1,2...n chuck], [1,2..nº]..]
    x = tf.reshape(x, [-1, chuck_size])
    # Separa x los diferentes chunks
    x = tf.split(x, n_chucks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chucks,chuck_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1,n_chucks,chuck_size)), y:mnist.test.labels}))

train_neural_network(x)




'''
say , u have a 5*5 image and u have 1 such image then it is :

x = np.ones((1,5,5))

so u have , 

x  =  array([[[ 1.,  1.,  1.,  1.,  1.],
                     [ 1.,  1.,  1.,  1.,  1.],
                     [ 1.,  1.,  1.,  1.,  1.],
                     [ 1.,  1.,  1.,  1.,  1.],
                     [ 1.,  1.,  1.,  1.,  1.]]])

now for the rnn u need to convert each row of pixel into a single chunk.
so , u would have 5 chunks of 5 values each
so, u need to convert each row to an array

x = np.transpose(x,(1,0,2))

this swaps the 0th dim with the 1st dim . so, u get shape of x as (5,1,5)
which is 5 arrays of 1 chunk each of 5 elements 

x = array([[[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]]])

now , u need to remove 1 pair of extra braces . so flatten by one dimension

x = np.reshape(x,(-1,chunk_size))

so, u will have :

x = array([[ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.]])

and finally u will need to split the entire thing into 5 chunks(5 arrays)
x = np.split(x,n_chunks,0)

so, finally u have :

x = [array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]])]
'''
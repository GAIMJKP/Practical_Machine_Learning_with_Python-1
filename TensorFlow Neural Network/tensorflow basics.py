import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Avoid warnings in windows

x1 = tf.constant(5)
x2 = tf.constant(6)

#Eficient way
result= tf.multiply(x1,x2) 


print(result)


with tf.Session() as sess:
	output=sess.run(result)
	print(output) 


#Example for running some computation on specific GPU
with tf.Session() as sess:
  with tf.device("/cpu:0"):  #change cpu for gpu
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    output2 = sess.run(product)
    
print(output2)

"""
sess = tf.Session()
print(sess.run(result))
sess.close()
"""








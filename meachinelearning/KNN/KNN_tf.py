

from __future__ import print_function
import numpy as np
import tensorflow as tf

from tensorflow.example.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot = True)

# In this example we limit mnist data
Xtr,Ytr = mnist.train.next_batch(5000)
Xte,Yte = mnist.test.next_batch(200)

# tf Graph Input
xtr = tf.placeholder("float",[None,784])
xte = tf.placeholder("float",[784])


# Nearest Neighbor calculation using L1 Distance
distance = tf.reduce_sum(tf.abs(xtr,tf.negative(xte)),reduction_indices=1)


# prediction:Get 
pred = tf.arg_min(distance,0)

accuracy = 0

init = tf.global_variables_initializer()

with tf.session() as sess:
	sess.run(init)
	# loop over test data
	for i in range(len(xte)):
		nn_index = sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
		# get nearest neighbor class label and compare it to its true label
		print("Test",i,"Prediction:",np.argmax(Ytr[nn_index]),\
			"True class:",np.argmax(Yte[i]))
		# calculate accuracy
		if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
			accuracy += 1./len(Xte)
	print("Done!")
	print("Accuracy:",accuracy)




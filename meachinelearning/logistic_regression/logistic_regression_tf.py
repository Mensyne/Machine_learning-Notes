
from __future__ import print_function

import tensorflow as tf

from tensorflow.example.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

#Parameters
learning_rate =0.01
training_epochs = 25
batch_size=100
display_step=1


#tf Graph Input
x = tf.placeholder(tf.float32,[None,784]) #mnist data image of shape 28*28=784
y =tf.placeholder(tf.float32,[None,10]) #0-9 digits recognition => 10 classes

# set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(X,W)+b)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),redution_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer()

init = tf.global_variables_initializer()

with tf.session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		avg_cost =0.
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_xs,batch_ys = mnist.train.next_batch(batch_size)
			_,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
			avg_cost += c/total_batch
			if (epoch+1)%display_step ==0:
				print("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(avg_cost))
	print("Optimization Finished!")
	correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	# calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print('Accuracy :',accuracy.eval({x:mnist.test.image,y:mnist.test.labels}))
	






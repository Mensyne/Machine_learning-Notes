
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

# parameters
learning_rate = 0.01
training_epochs =1000
display_step = 50

# Training Data

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0]

# tf.graph weihts
W = tf.variable(rng.randn(),name ='weight')
b = tf.Variable(rng.randn(),name = 'bias')

# construct a linear model
pred = tf.add(tf.multiply(X,W),b)

# Mean square error
coss = tf.reduce_sum(tf.pow(pred-Y,2)/(2*n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables
init =tf.global_variables_initializer()

# start training
with  tf.session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):
		if (epoch+1)%display_step ==0:
			c =sess.run(cost,feed_dict = {X:train_X,Y:train_Y})
			print("Epoch:",'%04d' %(epoch+1),"cost=","{:.9f}".format(c),\
				"W=",sess.run(W),"b=",sess.run(b))
	print('optimizer Finished!')
	training_cost = sess.run(cost,feed_dict={X:train_x,Y:train_Y})
	print("Training cost=",training_cost,'W=',sess.run(W),'b=',sess.run(b),'\n')

	# Graphic display
	plt.plot(train_X,train_Y,'ro',label='Original data')
	plt.plot(train_x,sess.run(W)*train_X+sess.run(b),label ='Fitted line')
	plt.legend()
	plt.show()

	# Testing example as requested(Issue #2)
	test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    print('Testing ....(Mean square loss Comparison)')
    testing_cost = sess.run(
    	tf.reduce_sum(tf.pow(pred-Y,2))/(2*test_X.shape[0]),
    	feed_dict ={X:test_X,Y:test_Y})
    print("Tesing cost=",testing_cost)
    print('Absolute mean square loss difference',abs(training_cost-testing_cost))

    plt.plot(test_X,test_Y,'bo',label='Testing data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()


    


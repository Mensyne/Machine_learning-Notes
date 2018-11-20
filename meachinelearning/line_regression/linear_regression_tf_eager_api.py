
from  __future__ import absolute_import,division,print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# set eager Api
tf.enable_eager_execution()
tfe = tf.contrib.eager


# Training Data
train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]

n_samples = len(train_X)

# Parameters
learning_rate = 0.01
display_step =100
num_steps = 1000

# weight and bias
W = tfe.Variables(np.random.randn())
b = tfe.Variables(np.random.randn())

# Linear regression(Wx+b)
def linear_regression(inputs):
	return inputs*W+b

# Mean square error
def mean_square_fn(model_fn,inoputs,labels):
	return tf.reduce_sum(tf.pow(model_fn(inputs)-labels,2))/(2*n_samples)

# SGD Optimizer
Optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

# Compute gradients
grad  = tfe.implicit_gradients(mean_square_fn)

# Initial cost before optimizing
print('Initial cost ={:.9f}'.format(
		mean_square_fn(linear_regression,train_X,train_Y)),
		"W=",W.numpy(),"b=",b.numpy())

# Training
for step in range(num_steps):
	Optimizer.apply_gradient(grad(linear_regression,train_X,train_Y))

	if(step+1)%display_step ==0  or step ==0:
		print("Epoch:",'%4d'%(step+1),"cost=","{:.9f}".format(mean_square_fn(linear_regression,train_X,train_Y)),
			"W=",W.numpy(),"b=",b.numpy())

plt.plot(train_X,train_Y,'ro',label= 'original data')		
plt.plot(train_X,np.array(W*train_X+b),label='Fitted line')
plt.legend()
plt.show()







from __future__ import absolute_import,division,print_function

import tensorflow as tf

# set eager api
tf.enable_eager_execution()
tfe = tf.contrib.eager


# Import mnist data
from tensorflow .examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=False)

# parameters
learning_rate = 0.1
batch_size = 128
num_steps = 1000
display_step= 100

dataset = tf.data.Dataset.from_tensor_slices(
	(mnist.train.images,mnist.train.labels)).batch(batch_size)

dataset_iter = tfe.Iterator(dataset)

#variables
W = tfe.Variable(tf.zeros([784,10]),name='weights')
b = tfe.Variable(tf.zeros([10]),name='bias')

# Logistic regression(Wx+b)
def logistic_regression(inputs):
	return tf.matmul(inputs,w)+b


# Cross-Entory loss function
def loss_fn(inference_fn,inputs,labels):
	# using sparse_softmax cross entropy
	return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=inference_fn(inputs),labels =labels))

# calculate accuracy
def accuracy_fn(inference_fn,inputs,labels):
	prediction = tf.nn.softmax(inference_fn(inputs))
	correct_pred = tf.equal(tf.argmax(prediction,1),labels)
	return tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# SGD Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Compute gradients
grad = tfe.implicit_gradients(loss_fn)

# Training
average_loss = 0.
average_acc = 0.
for step in range(num_steps):
	try:
		d = dataset_iter.next()
	except StopIteration:
		# Refill queue
		dataset_iter = tfe.Iterator(dataset)
		d  = dataset_iter.next()

	# Images
	x_batch = d[0]
	# labels
	y_batch = tf.cast(d[1],dtype =tf.int64)
	# compute the batch loss
	batch_loss = loss_fn(logistic_regression,x_batch,y_batch)
	average_loss += batch_loss
	batch_accuracy = accuracy_fn(logistic_regression,x_batch,y_batch)
	average_acc += batch_accuracy
	if step == 0:
		print("Initial loss={.9f}".format(average_loss))

	# Update the variables following gradients info
	optimizer.apply_gradients(grad(logistic_regression,x_batch,y_batch))
	# Display info
	if (step+1) % display_step ==0 or step ==0:
		if step >0:
			average_loss /= display_step
			average_acc /= display_step
		print("step:",'%04d'%(step+1),'loss=',
			"{:.9f}".format(average_loss),"accuracy=","{:.4f}".format(average_acc))
		average_loss = 0.
		average_acc = 0.

	testX = mnist.test.images
	testY = mnist.test.labels

	test_acc= accuracy_fn(logistic_regression,testX,testY)
	print('Testset Accuracy:{:.4f}'.format(test_acc))
	








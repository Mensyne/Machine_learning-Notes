

import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",validation_size=0)
# Training Parameters
learning_rate =0.01
num_steps = 3000
batch_size = 256
display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 =256
num_hidden_2  = 128
num_input = 784

# tf Graph input(only pictures)
X = tf.placeholder("float",[None,num_input])

weights = {

    "encoder_h1":tf.Variable(tf.random_normal([num_input,num_hidden_1])),
    "encoder_h2":tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
    "decoder_h1":tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1])),
    "decoder_h2":tf.Variable(tf.random_normal([num_hidden_1,num_input]))
}
biases = {

    "encoder_h1": tf.Variable(tf.random_normal([num_hidden_1])),
    "encoder_h2": tf.Variable(tf.random_normal([num_hidden_2])),
    "decoder_h1": tf.Variable(tf.random_normal([num_hidden_1])),
    "decoder_h2": tf.Variable(tf.random_normal([num_input]))

}

# Building the encoder
def encoder(f):
    layer_1 = tf.nn.sigmoid(tf.matmul(x,weights['encoder_h1'])),
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1,weights['encoder_h2']),
                            biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true =X

loss = tf.reduce_mean(tr.pow(y_true-))
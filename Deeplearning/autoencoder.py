

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



# 2016 Graphcore
# carlo@graphcore.ai
# ===============================================
"""Convolutional Network for MNIST classification."""

import tensorflow as tf
import numpy as np

def weight_variable(name, shape):
  initial = tf.random_normal_initializer(mean=0.0, stddev=0.1)
  return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initial)

def bias_variable(name, shape):
  initial = tf.constant_initializer(0.1)
  return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def infer(images, keep_prob):
  x_image = tf.reshape(images, [2,28,28,1])

  # Convolutional layer 1
  W_conv1 = weight_variable("wc1", [5,5,1,32])
  b_conv1 = bias_variable("bc1", [32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # Convolutional layer 2
  W_conv2 = weight_variable("wc2", [5, 5, 32, 64])
  b_conv2 = bias_variable("bc2", [64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer
  W_fc1 = weight_variable("wfc1", [7 * 7 * 64, 1024])
  b_fc1 = bias_variable("bfc1", [1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Redout layer
  W_fc2 = weight_variable("wfc2", [1024, 10])
  b_fc2 = bias_variable("bfc2", [10])

  return tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Inputs
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
kp = tf.placeholder(tf.float32)

with tf.variable_scope("vars", use_resource=True):
  # Inference
  logits = infer(x, kp)

# Training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

training_data = np.zeros([2, 784]);
training_labels = np.zeros([2, 10]);

sess.run(train_step, feed_dict={x: training_data, y_: training_labels, kp: 0.5})

sess.close()

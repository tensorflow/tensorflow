# This script is forked from a sample of Tensorflow

# Copyright 2016 Google. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from net import Net

class FourLayeredFFCNN(Net):

  def __init__(self, num_classes=10):
    super(FourLayeredFFCNN, self).__init__(num_classes)

    # Dropout placeholder.
    self._keep_prob = tf.placeholder(tf.float32)

  
  def inference(self, images):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name='weights')

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name='biases')

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    # reshape input
    x_image = tf.reshape(images, [-1, self._IMAGE_SIZE, self._IMAGE_SIZE, 1])

    # 1st Convolutional layer
    with tf.name_scope('conv1') as scope:
      weights = weight_variable([5, 5, 1, 32])
      biases = bias_variable([32])
      conv = tf.nn.relu(conv2d(x_image, weights) + biases)
      pool1 = max_pool_2x2(conv)

    # 2nd Convolutional layer
    with tf.name_scope('conv2') as scope:
      weights = weight_variable([5, 5, 32, 64])
      biases = bias_variable([64])
      conv = tf.nn.relu(conv2d(pool1, weights) + biases)
      pool2 = max_pool_2x2(conv)

    # Densely connected layer
    with tf.name_scope('fc1') as scope:
      weights = weight_variable([int(self._IMAGE_SIZE/4 * self._IMAGE_SIZE/4) * 64, 1024])
      biases = bias_variable([1024])
      pool2_flat = tf.reshape(pool2, [-1, int(self._IMAGE_SIZE/4 * self._IMAGE_SIZE/4)*64])
      fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

    # Dropout
    h_fc1_drop = tf.nn.dropout(fc1, self._keep_prob)

    # Readout layer (softmax)
    with tf.name_scope('softmax') as scope:
      weights = weight_variable([1024, self._NUM_CLASSES])
      biases = bias_variable([self._NUM_CLASSES])
      y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, weights) + biases)

    return y_conv

  def loss(self, logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size, NUM_CLASSES].

    Returns:
      loss: Loss tensor of type float.
    """
    cross_entropy = -tf.reduce_sum(tf.cast(labels, tf.float32) * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy

  def training(self, loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Create the Adam optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


  def evaluation(self, logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size, NUM_CLASSES] (one hot).

    Returns:
      A scalar float32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary("accuracy", accuracy)
    return accuracy

  def run(self, ops, feed_dict):
    return self._sess.run(ops, feed_dict=feed_dict)

  def train(self, feed_dict):
    feed_dict[self._keep_prob] = 0.5
    _, loss_value = self._sess.run([self._train_op, self._loss_op],
                                   feed_dict=feed_dict)
    return loss_value

  def eval(self, feed_dict):
    feed_dict[self._keep_prob] = 1.0
    true_count = self._sess.run(self._eval_op,
                                feed_dict=feed_dict)
    return true_count

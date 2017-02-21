# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""MNIST training in imperative mode TensorFlow."""

# pylint: disable=redefined-outer-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.contrib.imperative as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10
BATCH_SIZE = 100
NUM_EPOCHS = 2
LEARNING_RATE = 0.1


class Model(object):
  """Fully connected model for MNIST."""

  def __init__(self, hidden1_units, hidden2_units):
    """Create the model parameters."""
    self.params = []
    # Hidden 1
    with tf.name_scope('hidden1'):
      self.weights1 = tf.Variable(
          np.random.normal(scale=1.0 / np.sqrt(float(IMAGE_PIXELS)),
                           size=[IMAGE_PIXELS, hidden1_units]),
          dtype=tf.float32,
          name='weights')
      self.biases1 = tf.Variable(
          np.zeros([hidden1_units]),
          dtype=tf.float32,
          name='biases')
    # Hidden 2
    with tf.name_scope('hidden2'):
      self.weights2 = tf.Variable(
          np.random.normal(scale=1.0 / np.sqrt(float(hidden1_units)),
                           size=[hidden1_units, hidden2_units]),
          dtype=tf.float32,
          name='weights')
    self.biases2 = tf.Variable(
        np.zeros([hidden2_units]),
        dtype=tf.float32,
        name='biases')
    # Linear
    with tf.name_scope('softmax_linear'):
      self.sm_w = tf.Variable(
          np.random.normal(scale=1.0 / np.sqrt(float(hidden2_units)),
                           size=[hidden2_units, NUM_CLASSES]),
          dtype=tf.float32,
          name='weights')
      self.sm_b = tf.Variable(
          np.zeros([NUM_CLASSES]),
          dtype=tf.float32,
          name='biases')
    self.params = [self.weights1, self.biases1,
                   self.weights2, self.biases2,
                   self.sm_w, self.sm_b]

  def __call__(self, images):
    """Run the model's forward prop on `images`."""
    hidden1 = tf.nn.relu(tf.matmul(images, self.weights1) + self.biases1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, self.weights2) + self.biases2)
    logits = tf.matmul(hidden2, self.sm_w) + self.sm_b
    return logits


model = Model(128, 32)
data = read_data_sets('/tmp/mnist_train')


def get_test_accuracy():
  """Gets the model's classification accuracy on test data."""
  num_examples = data.test.num_examples
  test_images = np.split(data.test.images, num_examples/BATCH_SIZE)
  test_labels = np.split(data.test.labels.astype(np.int32),
                         num_examples/BATCH_SIZE)
  num_correct = 0
  for _, (images, labels) in enumerate(zip(test_images, test_labels)):
    with tf.new_step():
      logits = model(images)
      predictions = tf.argmax(tf.nn.softmax(logits), axis=1)
      num_correct += np.sum(predictions.value == labels)
  return float(num_correct) / float(num_examples)

num_examples = data.train.num_examples
train_images = np.split(data.train.images, num_examples/BATCH_SIZE)
train_labels = np.split(data.train.labels.astype(np.int32),
                        num_examples/BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
  for i, (images, labels) in enumerate(zip(train_images, train_labels)):
    with tf.new_step() as step:
      logits = model(images)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name='xentropy')
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
      gradients = tf.gradients(loss, model.params)
      step.run([v.assign_sub(LEARNING_RATE * g)
                for g, v in zip(gradients, model.params)])
      if i % 10 == 0:
        print('Loss after {} steps = {}'.format(i, loss))
      if i % 100 == 0:
        print('Test accuracy after {} steps = {}'
              .format(i, get_test_accuracy()))

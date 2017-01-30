#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""This showcases how simple it is to build image classification networks.

It follows description from this TensorFlow tutorial:
    https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
import tensorflow as tf

layers = tf.contrib.layers
learn = tf.contrib.learn


def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(
      tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(feature, target, mode):
  """2-layer convolution model."""
  # Convert the target to a one-hot tensor of shape (batch_size, 10) and
  # with a on-value of 1 for each one-hot vector of length 10.
  target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

  # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
  # image width and height final dimension being the number of color channels.
  feature = tf.reshape(feature, [-1, 28, 28, 1])

  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = layers.convolution(
        feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool1 = max_pool_2x2(h_conv1)

  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = layers.convolution(
        h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  # Densely connected layer with 1024 neurons.
  h_fc1 = layers.dropout(
      layers.fully_connected(
          h_pool2_flat, 1024, activation_fn=tf.nn.relu),
      keep_prob=0.5,
      is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

  # Create a tensor for training op.
  train_op = layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='SGD',
      learning_rate=0.001)

  return tf.argmax(logits, 1), loss, train_op


def main(unused_args):
  ### Download and load MNIST dataset.
  mnist = learn.datasets.load_dataset('mnist')

  ### Linear classifier.
  feature_columns = learn.infer_real_valued_columns_from_input(
      mnist.train.images)
  classifier = learn.LinearClassifier(
      feature_columns=feature_columns, n_classes=10)
  classifier.fit(mnist.train.images,
                 mnist.train.labels.astype(np.int32),
                 batch_size=100,
                 steps=1000)
  score = metrics.accuracy_score(mnist.test.labels,
                                 list(classifier.predict(mnist.test.images)))
  print('Accuracy: {0:f}'.format(score))

  ### Convolutional network
  classifier = learn.Estimator(model_fn=conv_model)
  classifier.fit(mnist.train.images,
                 mnist.train.labels,
                 batch_size=100,
                 steps=20000)
  score = metrics.accuracy_score(mnist.test.labels,
                                 list(classifier.predict(mnist.test.images)))
  print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
  tf.app.run()

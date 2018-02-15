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
import tensorflow as tf


N_DIGITS = 10  # Number of digits.
X_FEATURE = 'x'  # Name of the input feature.


def conv_model(features, labels, mode):
  """2-layer convolution model."""
  # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
  # image width and height final dimension being the number of color channels.
  feature = tf.reshape(features[X_FEATURE], [-1, 28, 28, 1])

  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = tf.layers.conv2d(
        feature,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    h_pool1 = tf.layers.max_pooling2d(
        h_conv1, pool_size=2, strides=2, padding='same')

  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = tf.layers.conv2d(
        h_pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    h_pool2 = tf.layers.max_pooling2d(
        h_conv2, pool_size=2, strides=2, padding='same')
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  # Densely connected layer with 1024 neurons.
  h_fc1 = tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)
  if mode == tf.estimator.ModeKeys.TRAIN:
    h_fc1 = tf.layers.dropout(h_fc1, rate=0.5)

  # Compute logits (1 per class) and compute loss.
  logits = tf.layers.dense(h_fc1, N_DIGITS, activation=None)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class': predicted_classes,
        'prob': tf.nn.softmax(logits)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute loss.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Create training op.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  # Compute evaluation metrics.
  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_args):
  tf.logging.set_verbosity(tf.logging.INFO)

  ### Download and load MNIST dataset.
  mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: mnist.train.images},
      y=mnist.train.labels.astype(np.int32),
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: mnist.train.images},
      y=mnist.train.labels.astype(np.int32),
      num_epochs=1,
      shuffle=False)

  ### Linear classifier.
  feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=mnist.train.images.shape[1:])]

  classifier = tf.estimator.LinearClassifier(
      feature_columns=feature_columns, n_classes=N_DIGITS)
  classifier.train(input_fn=train_input_fn, steps=200)
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (LinearClassifier): {0:f}'.format(scores['accuracy']))

  ### Convolutional network
  classifier = tf.estimator.Estimator(model_fn=conv_model)
  classifier.train(input_fn=train_input_fn, steps=200)
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (conv_model): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.app.run()

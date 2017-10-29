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
"""This example builds deep residual network for mnist data.

Reference Paper: http://arxiv.org/pdf/1512.03385.pdf

Note that this is still a work-in-progress. Feel free to submit a PR
to make this better.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from math import sqrt

import numpy as np
import tensorflow as tf


N_DIGITS = 10  # Number of digits.
X_FEATURE = 'x'  # Name of the input feature.


def res_net_model(features, labels, mode):
  """Builds a residual network."""

  # Configurations for each bottleneck group.
  BottleneckGroup = namedtuple('BottleneckGroup',
                               ['num_blocks', 'num_filters', 'bottleneck_size'])
  groups = [
      BottleneckGroup(3, 128, 32), BottleneckGroup(3, 256, 64),
      BottleneckGroup(3, 512, 128), BottleneckGroup(3, 1024, 256)
  ]

  x = features[X_FEATURE]
  input_shape = x.get_shape().as_list()

  # Reshape the input into the right shape if it's 2D tensor
  if len(input_shape) == 2:
    ndim = int(sqrt(input_shape[1]))
    x = tf.reshape(x, [-1, ndim, ndim, 1])

  # First convolution expands to 64 channels
  with tf.variable_scope('conv_layer1'):
    net = tf.layers.conv2d(
        x,
        filters=64,
        kernel_size=7,
        activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net)

  # Max pool
  net = tf.layers.max_pooling2d(
      net, pool_size=3, strides=2, padding='same')

  # First chain of resnets
  with tf.variable_scope('conv_layer2'):
    net = tf.layers.conv2d(
        net,
        filters=groups[0].num_filters,
        kernel_size=1,
        padding='valid')

  # Create the bottleneck groups, each of which contains `num_blocks`
  # bottleneck groups.
  for group_i, group in enumerate(groups):
    for block_i in range(group.num_blocks):
      name = 'group_%d/block_%d' % (group_i, block_i)

      # 1x1 convolution responsible for reducing dimension
      with tf.variable_scope(name + '/conv_in'):
        conv = tf.layers.conv2d(
            net,
            filters=group.num_filters,
            kernel_size=1,
            padding='valid',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv)

      with tf.variable_scope(name + '/conv_bottleneck'):
        conv = tf.layers.conv2d(
            conv,
            filters=group.bottleneck_size,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv)

      # 1x1 convolution responsible for restoring dimension
      with tf.variable_scope(name + '/conv_out'):
        input_dim = net.get_shape()[-1].value
        conv = tf.layers.conv2d(
            conv,
            filters=input_dim,
            kernel_size=1,
            padding='valid',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv)

      # shortcut connections that turn the network into its counterpart
      # residual function (identity shortcut)
      net = conv + net

    try:
      # upscale to the next group size
      next_group = groups[group_i + 1]
      with tf.variable_scope('block_%d/conv_upscale' % group_i):
        net = tf.layers.conv2d(
            net,
            filters=next_group.num_filters,
            kernel_size=1,
            padding='same',
            activation=None,
            bias_initializer=None)
    except IndexError:
      pass

  net_shape = net.get_shape().as_list()
  net = tf.nn.avg_pool(
      net,
      ksize=[1, net_shape[1], net_shape[2], 1],
      strides=[1, 1, 1, 1],
      padding='VALID')

  net_shape = net.get_shape().as_list()
  net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

  # Compute logits (1 per class) and compute loss.
  logits = tf.layers.dense(net, N_DIGITS, activation=None)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class': predicted_classes,
        'prob': tf.nn.softmax(logits)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute loss.
  onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), N_DIGITS, 1, 0)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Create training op.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
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
  # Download and load MNIST data.
  mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')

  # Create a new resnet classifier.
  classifier = tf.estimator.Estimator(model_fn=res_net_model)

  tf.logging.set_verbosity(tf.logging.INFO)  # Show training logs.

  # Train model and save summaries into logdir.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: mnist.train.images},
      y=mnist.train.labels.astype(np.int32),
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=100)

  # Calculate accuracy.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: mnist.test.images},
      y=mnist.test.labels.astype(np.int32),
      num_epochs=1,
      shuffle=False)
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy: {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.app.run()

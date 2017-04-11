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
import os

import tensorflow as tf

batch_norm = tf.contrib.layers.batch_norm
convolution2d = tf.contrib.layers.convolution2d


def res_net(x, y, activation=tf.nn.relu):
  """Builds a residual network.

  Note that if the input tensor is 2D, it must be square in order to be
  converted to a 4D tensor.

  Borrowed structure from:
  github.com/pkmital/tensorflow_tutorials/blob/master/10_residual_network.py

  Args:
    x: Input of the network
    y: Output of the network
    activation: Activation function to apply after each convolution

  Returns:
    Predictions and loss tensors.
  """

  # Configurations for each bottleneck group.
  BottleneckGroup = namedtuple('BottleneckGroup',
                               ['num_blocks', 'num_filters', 'bottleneck_size'])
  groups = [
      BottleneckGroup(3, 128, 32), BottleneckGroup(3, 256, 64),
      BottleneckGroup(3, 512, 128), BottleneckGroup(3, 1024, 256)
  ]

  input_shape = x.get_shape().as_list()

  # Reshape the input into the right shape if it's 2D tensor
  if len(input_shape) == 2:
    ndim = int(sqrt(input_shape[1]))
    x = tf.reshape(x, [-1, ndim, ndim, 1])

  # First convolution expands to 64 channels
  with tf.variable_scope('conv_layer1'):
    net = convolution2d(
        x, 64, 7, normalizer_fn=batch_norm, activation_fn=activation)

  # Max pool
  net = tf.nn.max_pool(net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # First chain of resnets
  with tf.variable_scope('conv_layer2'):
    net = convolution2d(net, groups[0].num_filters, 1, padding='VALID')

  # Create the bottleneck groups, each of which contains `num_blocks`
  # bottleneck groups.
  for group_i, group in enumerate(groups):
    for block_i in range(group.num_blocks):
      name = 'group_%d/block_%d' % (group_i, block_i)

      # 1x1 convolution responsible for reducing dimension
      with tf.variable_scope(name + '/conv_in'):
        conv = convolution2d(
            net,
            group.bottleneck_size,
            1,
            padding='VALID',
            activation_fn=activation,
            normalizer_fn=batch_norm)

      with tf.variable_scope(name + '/conv_bottleneck'):
        conv = convolution2d(
            conv,
            group.bottleneck_size,
            3,
            padding='SAME',
            activation_fn=activation,
            normalizer_fn=batch_norm)

      # 1x1 convolution responsible for restoring dimension
      with tf.variable_scope(name + '/conv_out'):
        input_dim = net.get_shape()[-1].value
        conv = convolution2d(
            conv,
            input_dim,
            1,
            padding='VALID',
            activation_fn=activation,
            normalizer_fn=batch_norm)

      # shortcut connections that turn the network into its counterpart
      # residual function (identity shortcut)
      net = conv + net

    try:
      # upscale to the next group size
      next_group = groups[group_i + 1]
      with tf.variable_scope('block_%d/conv_upscale' % group_i):
        net = convolution2d(
            net,
            next_group.num_filters,
            1,
            activation_fn=None,
            biases_initializer=None,
            padding='SAME')
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

  target = tf.one_hot(y, depth=10, dtype=tf.float32)
  logits = tf.contrib.layers.fully_connected(net, 10, activation_fn=None)
  loss = tf.losses.softmax_cross_entropy(target, logits)
  return tf.softmax(logits), loss


def res_net_model(x, y):
  prediction, loss = res_net(x, y)
  predicted = tf.argmax(prediction, 1)
  accuracy = tf.equal(predicted, tf.cast(y, tf.int64))
  predictions = {'prob': prediction, 'class': predicted, 'accuracy': accuracy}
  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='Adagrad',
      learning_rate=0.001)
  return predictions, loss, train_op


# Download and load MNIST data.
mnist = tf.contrib.learn.datasets.load_dataset('mnist')

# Create a new resnet classifier.
classifier = tf.contrib.learn.Estimator(model_fn=res_net_model)

tf.logging.set_verbosity(tf.logging.INFO)  # Show training logs. (avoid silence)

# Train model and save summaries into logdir.
classifier.fit(mnist.train.images,
               mnist.train.labels,
               batch_size=100,
               steps=1000)

# Calculate accuracy.
result = classifier.evaluate(
    x=mnist.test.images,
    y=mnist.test.labels,
    metrics={
        'accuracy':
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key='accuracy'),
    })
score = result['accuracy']
print('Accuracy: {0:f}'.format(score))

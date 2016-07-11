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

from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import input_data


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
  BottleneckGroup = namedtuple(
      'BottleneckGroup', ['num_blocks', 'num_filters', 'bottleneck_size'])
  groups = [BottleneckGroup(3, 128, 32),
            BottleneckGroup(3, 256, 64),
            BottleneckGroup(3, 512, 128),
            BottleneckGroup(3, 1024, 256)]

  input_shape = x.get_shape().as_list()

  # Reshape the input into the right shape if it's 2D tensor
  if len(input_shape) == 2:
    ndim = int(sqrt(input_shape[1]))
    x = tf.reshape(x, [-1, ndim, ndim, 1])

  # First convolution expands to 64 channels
  with tf.variable_scope('conv_layer1'):
    net = learn.ops.conv2d(x, 64, [7, 7], batch_norm=True,
                           activation=activation, bias=False)

  # Max pool
  net = tf.nn.max_pool(
      net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # First chain of resnets
  with tf.variable_scope('conv_layer2'):
    net = learn.ops.conv2d(net, groups[0].num_filters,
                           [1, 1], [1, 1, 1, 1],
                           padding='VALID', bias=True)

  # Create the bottleneck groups, each of which contains `num_blocks`
  # bottleneck groups.
  for group_i, group in enumerate(groups):
    for block_i in range(group.num_blocks):
      name = 'group_%d/block_%d' % (group_i, block_i)

      # 1x1 convolution responsible for reducing dimension
      with tf.variable_scope(name + '/conv_in'):
        conv = learn.ops.conv2d(net, group.bottleneck_size,
                                [1, 1], [1, 1, 1, 1],
                                padding='VALID',
                                activation=activation,
                                batch_norm=True,
                                bias=False)

      with tf.variable_scope(name + '/conv_bottleneck'):
        conv = learn.ops.conv2d(conv, group.bottleneck_size,
                                [3, 3], [1, 1, 1, 1],
                                padding='SAME',
                                activation=activation,
                                batch_norm=True,
                                bias=False)

      # 1x1 convolution responsible for restoring dimension
      with tf.variable_scope(name + '/conv_out'):
        input_dim = net.get_shape()[-1].value
        conv = learn.ops.conv2d(conv, input_dim,
                                [1, 1], [1, 1, 1, 1],
                                padding='VALID',
                                activation=activation,
                                batch_norm=True,
                                bias=False)

      # shortcut connections that turn the network into its counterpart
      # residual function (identity shortcut)
      net = conv + net

    try:
      # upscale to the next group size
      next_group = groups[group_i + 1]
      with tf.variable_scope('block_%d/conv_upscale' % group_i):
        net = learn.ops.conv2d(net, next_group.num_filters,
                               [1, 1], [1, 1, 1, 1],
                               bias=False,
                               padding='SAME')
    except IndexError:
      pass

  net_shape = net.get_shape().as_list()
  net = tf.nn.avg_pool(net,
                       ksize=[1, net_shape[1], net_shape[2], 1],
                       strides=[1, 1, 1, 1], padding='VALID')

  net_shape = net.get_shape().as_list()
  net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

  return learn.models.logistic_regression(net, y)

# Download and load MNIST data.
mnist = input_data.read_data_sets('MNIST_data')

# Restore model if graph is saved into a folder.
if os.path.exists('models/resnet/graph.pbtxt'):
  classifier = learn.TensorFlowEstimator.restore('models/resnet/')

while True:
  # Train model and save summaries into logdir.
  classifier.fit(
      mnist.train.images, mnist.train.labels, logdir='models/resnet/')

  # Calculate accuracy.
  score = metrics.accuracy_score(
      mnist.test.labels, classifier.predict(mnist.test.images, batch_size=64))
  print('Accuracy: {0:f}'.format(score))

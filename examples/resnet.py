#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
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

"""
This example builds deep residual network for mnist data. 
Reference Paper: http://arxiv.org/pdf/1512.03385.pdf
"""

import os
import random
from sklearn import metrics

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import skflow

from collections import namedtuple
from math import sqrt


def res_net(x, y, activation=tf.nn.relu):
    """Builds a residual network.

    Borrowed structure from here: https://github.com/pkmital/tensorflow_tutorials/blob/master/10_residual_network.py

    Args:
        x: Input of the network
        y: Output of the network
        activation: Activation function to apply after each convolution
    Raises:
        ValueError
            If a 2D tensor is not square, it cannot be converted to a 
            4D tensor.
    """
    LayerBlock = namedtuple(
        'LayerBlock', ['num_layers', 'num_filters', 'bottleneck_size'])
    blocks = [LayerBlock(3, 128, 32),
              LayerBlock(3, 256, 64),
              LayerBlock(3, 512, 128),
              LayerBlock(3, 1024, 256)]

    # Input check
    input_shape = x.get_shape().as_list()
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        if ndim * ndim != input_shape[1]:
            raise ValueError('input_shape should be square')
        x = tf.reshape(x, [-1, ndim, ndim, 1])

    # First convolution expands to 64 channels
    with tf.variable_scope('conv_layer1'):
        net = skflow.ops.conv2d(x, 64, [7, 7], batch_norm=True,
                                activation=activation, bias=False)

    # Max pool
    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # First chain of resnets
    with tf.variable_scope('conv_layer2'):
        net = skflow.ops.conv2d(net, blocks[0].num_filters,
                               [1, 1], [1, 1, 1, 1],
                               padding='VALID', bias=True)

    # Create resnets for each residual block
    for block_i, block in enumerate(blocks):
        for layer_i in range(block.num_layers):

            name = 'block_%d/layer_%d' % (block_i, layer_i)
            with tf.variable_scope(name + '/conv_in'):
                conv = skflow.ops.conv2d(net, block.num_filters,
                                         [1, 1], [1, 1, 1, 1],
                                         padding='VALID',
                                         activation=activation,
                                         batch_norm=True,
                                         bias=False)

            with tf.variable_scope(name + '/conv_bottleneck'):
                conv = skflow.ops.conv2d(conv, block.bottleneck_size,
                                         [3, 3], [1, 1, 1, 1],
                                         padding='SAME',
                                         activation=activation,
                                         batch_norm=True,
                                         bias=False)

            with tf.variable_scope(name + '/conv_out'):
                conv = skflow.ops.conv2d(conv, block.num_filters,
                                         [1, 1], [1, 1, 1, 1],
                                         padding='VALID',
                                         activation=activation,
                                         batch_norm=True,
                                         bias=False)

            net = conv + net

        try:
            # upscale to the next block size
            next_block = blocks[block_i + 1]
            with tf.variable_scope('block_%d/conv_upscale' % block_i):
                net = skflow.ops.conv2d(net, next_block.num_filters,
                                        [1, 1], [1, 1, 1, 1],
                                        bias=False,
                                        padding='SAME')
        except IndexError:
            pass


    net = tf.nn.avg_pool(net,
                         ksize=[1, net.get_shape().as_list()[1],
                                net.get_shape().as_list()[2], 1],
                         strides=[1, 1, 1, 1], padding='VALID')
    net = tf.reshape(
        net,
        [-1, net.get_shape().as_list()[1] *
         net.get_shape().as_list()[2] *
         net.get_shape().as_list()[3]])

    return skflow.models.logistic_regression(net, y)


# Download and load MNIST data.
mnist = input_data.read_data_sets('MNIST_data')

# Restore model if graph is saved into a folder.
if os.path.exists("models/resnet/graph.pbtxt"):
    classifier = skflow.TensorFlowEstimator.restore("models/resnet/")
else:
    # Create a new resnet classifier.
    classifier = skflow.TensorFlowEstimator(
        model_fn=res_net, n_classes=10, batch_size=100, steps=100,
        learning_rate=0.001)

while True:
    # Train model and save summaries into logdir.
    classifier.fit(mnist.train.images, mnist.train.labels, logdir="models/resnet/")

    # Calculate accuracy.
    score = metrics.accuracy_score(
        mnist.test.labels, classifier.predict(mnist.test.images, batch_size=64))
    print('Accuracy: {0:f}'.format(score))

    # Save model graph and checkpoints.
    classifier.save("models/resnet/")


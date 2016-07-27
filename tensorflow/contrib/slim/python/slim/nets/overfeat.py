# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the model definition for the OverFeat network.

The definition for the network was obtained from:
  OverFeat: Integrated Recognition, Localization and Detection using
  Convolutional Networks
  Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus and
  Yann LeCun, 2014
  http://arxiv.org/abs/1312.6229

Usage:
  with slim.arg_scope(overfeat.overfeat_arg_scope()):
    outputs, end_points = overfeat.overfeat(inputs)

@@overfeat
"""

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def overfeat_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def overfeat(inputs,
             num_classes=1000,
             dropout_keep_prob=0.5,
             is_training=True,
             spatial_squeeze=True,
             scope='overfeat'):
  """Contains the model definition for the OverFeat network.

  The definition for the network was obtained from:
    OverFeat: Integrated Recognition, Localization and Detection using
    Convolutional Networks
    Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus and
    Yann LeCun, 2014
    http://arxiv.org/abs/1312.6229

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 231x231. To use in fully
        convolutional mode, set spatial_squeeze to false.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    is_training: whether or not the model is being trained.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.

  """
  with tf.variable_op_scope([inputs], scope, 'overfeat') as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.conv2d(net, 256, [5, 5], padding='VALID', scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.conv2d(net, 512, [3, 3], scope='conv3')
      net = slim.conv2d(net, 1024, [3, 3], scope='conv4')
      net = slim.conv2d(net, 1024, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 3072, [6, 6], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer,
                          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = dict(tf.get_collection(end_points_collection))
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points

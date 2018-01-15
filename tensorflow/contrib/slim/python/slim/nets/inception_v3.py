# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def inception_v3_base(inputs,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
  """Inception model from http://arxiv.org/abs/1512.00567.

  Constructs an Inception v3 network from inputs to the given final endpoint.
  This method can construct the network up to the final inception block
  Mixed_7c.

  Note that the names of the layers in the paper do not correspond to the names
  of the endpoints registered by this function although they build the same
  network.

  Here is a mapping from the old_names to the new names:
  Old name          | New name
  =======================================
  conv0             | Conv2d_1a_3x3
  conv1             | Conv2d_2a_3x3
  conv2             | Conv2d_2b_3x3
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  conv4             | Conv2d_4a_3x3
  pool2             | MaxPool_5a_3x3
  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d
  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e
  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with variable_scope.variable_scope(scope, 'InceptionV3', [inputs]):
    with arg_scope(
        [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
        stride=1,
        padding='VALID'):
      # 299 x 299 x 3
      end_point = 'Conv2d_1a_3x3'
      net = layers.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 149 x 149 x 32
      end_point = 'Conv2d_2a_3x3'
      net = layers.conv2d(net, depth(32), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 147 x 147 x 32
      end_point = 'Conv2d_2b_3x3'
      net = layers.conv2d(
          net, depth(64), [3, 3], padding='SAME', scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 147 x 147 x 64
      end_point = 'MaxPool_3a_3x3'
      net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 73 x 73 x 64
      end_point = 'Conv2d_3b_1x1'
      net = layers.conv2d(net, depth(80), [1, 1], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 73 x 73 x 80.
      end_point = 'Conv2d_4a_3x3'
      net = layers.conv2d(net, depth(192), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 71 x 71 x 192.
      end_point = 'MaxPool_5a_3x3'
      net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 35 x 35 x 192.

      # Inception blocks
    with arg_scope(
        [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
        stride=1,
        padding='SAME'):
      # mixed: 35 x 35 x 256.
      end_point = 'Mixed_5b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_1: 35 x 35 x 288.
      end_point = 'Mixed_5c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_2: 35 x 35 x 288.
      end_point = 'Mixed_5d'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_3: 17 x 17 x 768.
      end_point = 'Mixed_6a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net,
              depth(384), [3, 3],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = layers.conv2d(
              branch_1,
              depth(96), [3, 3],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_1x1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.max_pool2d(
              net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        net = array_ops.concat([branch_0, branch_1, branch_2], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed4: 17 x 17 x 768.
      end_point = 'Mixed_6b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = layers.conv2d(
              branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = layers.conv2d(
              branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_5: 17 x 17 x 768.
      end_point = 'Mixed_6c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = layers.conv2d(
              branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = layers.conv2d(
              branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # mixed_6: 17 x 17 x 768.
      end_point = 'Mixed_6d'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = layers.conv2d(
              branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = layers.conv2d(
              branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_7: 17 x 17 x 768.
      end_point = 'Mixed_6e'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = layers.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_8: 8 x 8 x 1280.
      end_point = 'Mixed_7a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = layers.conv2d(
              branch_0,
              depth(320), [3, 3],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = layers.conv2d(
              branch_1,
              depth(192), [3, 3],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.max_pool2d(
              net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        net = array_ops.concat([branch_0, branch_1, branch_2], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # mixed_9: 8 x 8 x 2048.
      end_point = 'Mixed_7b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = array_ops.concat(
              [
                  layers.conv2d(
                      branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                  layers.conv2d(
                      branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')
              ],
              3)
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = array_ops.concat(
              [
                  layers.conv2d(
                      branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                  layers.conv2d(
                      branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
              ],
              3)
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_10: 8 x 8 x 2048.
      end_point = 'Mixed_7c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = array_ops.concat(
              [
                  layers.conv2d(
                      branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                  layers.conv2d(
                      branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')
              ],
              3)
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers.conv2d(
              net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = layers.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = array_ops.concat(
              [
                  layers.conv2d(
                      branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                  layers.conv2d(
                      branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
              ],
              3)
        with variable_scope.variable_scope('Branch_3'):
          branch_3 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = layers.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=layers_lib.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
  """Inception model from http://arxiv.org/abs/1512.00567.

  "Rethinking the Inception Architecture for Computer Vision"

  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
  Zbigniew Wojna.

  With the default arguments this method constructs the exact model defined in
  the paper. However, one can experiment with variations of the inception_v3
  network by changing arguments dropout_keep_prob, min_depth and
  depth_multiplier.

  The default image size used to train this network is 299x299.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
      of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      To use this parameter, the input images must be smaller
      than 300x300 pixels, in which case the output logit layer
      does not contain spatial information and can be removed.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if 'depth_multiplier' is less than or equal to zero.
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with variable_scope.variable_scope(
      scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
    with arg_scope(
        [layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
      net, end_points = inception_v3_base(
          inputs,
          scope=scope,
          min_depth=min_depth,
          depth_multiplier=depth_multiplier)

      # Auxiliary Head logits
      with arg_scope(
          [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
          stride=1,
          padding='SAME'):
        aux_logits = end_points['Mixed_6e']
        with variable_scope.variable_scope('AuxLogits'):
          aux_logits = layers_lib.avg_pool2d(
              aux_logits, [5, 5],
              stride=3,
              padding='VALID',
              scope='AvgPool_1a_5x5')
          aux_logits = layers.conv2d(
              aux_logits, depth(128), [1, 1], scope='Conv2d_1b_1x1')

          # Shape of feature map before the final layer.
          kernel_size = _reduced_kernel_size_for_small_input(aux_logits, [5, 5])
          aux_logits = layers.conv2d(
              aux_logits,
              depth(768),
              kernel_size,
              weights_initializer=trunc_normal(0.01),
              padding='VALID',
              scope='Conv2d_2a_{}x{}'.format(*kernel_size))
          aux_logits = layers.conv2d(
              aux_logits,
              num_classes, [1, 1],
              activation_fn=None,
              normalizer_fn=None,
              weights_initializer=trunc_normal(0.001),
              scope='Conv2d_2b_1x1')
          if spatial_squeeze:
            aux_logits = array_ops.squeeze(
                aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits

      # Final pooling and prediction
      with variable_scope.variable_scope('Logits'):
        kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
        net = layers_lib.avg_pool2d(
            net,
            kernel_size,
            padding='VALID',
            scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        # 1 x 1 x 2048
        net = layers_lib.dropout(
            net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        logits = layers.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


inception_v3.default_image_size = 299


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.tf.contrib.slim.ops._two_element_tuple
  cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [
        min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
    ]
  return kernel_size_out


def inception_v3_arg_scope(weight_decay=0.00004,
                           batch_norm_var_collection='moving_vars',
                           use_fused_batchnorm=True):
  """Defines the default InceptionV3 arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_var_collection: The name of the collection for the batch norm
      variables.
    use_fused_batchnorm: Enable fused batchnorm.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      # Use fused batch norm if possible.
      'fused': use_fused_batchnorm,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }

  # Set weight_decay for weights in Conv and FC layers.
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope(
        [layers.conv2d],
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers_lib.batch_norm,
        normalizer_params=batch_norm_params) as sc:
      return sc

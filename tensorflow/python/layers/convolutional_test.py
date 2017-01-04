# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.layers.core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ConvTest(test.TestCase):

  def testInvalidDataFormat(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'data_format'):
      conv_layers.conv2d(images, 32, 3, data_format='invalid')

  def testInvalidStrides(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'strides'):
      conv_layers.conv2d(images, 32, 3, strides=(1, 2, 3))

    with self.assertRaisesRegexp(ValueError, 'strides'):
      conv_layers.conv2d(images, 32, 3, strides=None)

  def testInvalidKernelSize(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'kernel_size'):
      conv_layers.conv2d(images, 32, (1, 2, 3))

    with self.assertRaisesRegexp(ValueError, 'kernel_size'):
      conv_layers.conv2d(images, 32, None)

  def testCreateConv2D(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.Conv2D(32, [3, 3], activation=nn_ops.relu)
    output = layer.apply(images)
    self.assertEqual(output.op.name, 'conv2d/Relu')
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height - 2, width - 2, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateConv2DIntegerKernelSize(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.Conv2D(32, 3)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height - 2, width - 2, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateConv2DChannelsFirst(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, 4, height, width))
    layer = conv_layers.Conv2D(32, [3, 3], data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, 32, height - 2, width - 2])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testConv2DPaddingSame(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 32), seed=1)
    layer = conv_layers.Conv2D(64, images.get_shape()[1:3], padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, height, width, 64])

  def testCreateConvWithStrides(self):
    height, width = 6, 8
    # Test strides tuple
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    layer = conv_layers.Conv2D(32, [3, 3], strides=(2, 2), padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width / 2, 32])

    # Test strides integer
    layer = conv_layers.Conv2D(32, [3, 3], strides=2, padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width / 2, 32])

    # Test unequal strides
    layer = conv_layers.Conv2D(32, [3, 3], strides=(2, 1), padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width, 32])

  def testCreateConv1D(self):
    width = 7
    data = random_ops.random_uniform((5, width, 4))
    layer = conv_layers.Conv1D(32, 3, activation=nn_ops.relu)
    output = layer.apply(data)
    self.assertEqual(output.op.name, 'conv1d/Relu')
    self.assertListEqual(output.get_shape().as_list(), [5, width - 2, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateConv1DChannelsFirst(self):
    width = 7
    data = random_ops.random_uniform((5, 4, width))
    layer = conv_layers.Conv1D(32, 3, data_format='channels_first')
    output = layer.apply(data)
    self.assertListEqual(output.get_shape().as_list(), [5, 32, width - 2])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateConv3D(self):
    depth, height, width = 6, 7, 9
    volumes = random_ops.random_uniform((5, depth, height, width, 4))
    layer = conv_layers.Conv3D(32, [3, 3, 3], activation=nn_ops.relu)
    output = layer.apply(volumes)
    self.assertEqual(output.op.name, 'conv3d/Relu')
    self.assertListEqual(output.get_shape().as_list(),
                         [5, depth - 2, height - 2, width - 2, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 3, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testConv2DKernelRegularizer(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    layer = conv_layers.Conv2D(32, [3, 3], kernel_regularizer=reg)
    layer.apply(images)
    loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(layer.losses, loss_keys)

  def testConv2DBiasRegularizer(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    layer = conv_layers.Conv2D(32, [3, 3], bias_regularizer=reg)
    layer.apply(images)
    loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(layer.losses, loss_keys)

  def testConv2DNoBias(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.Conv2D(
        32, [3, 3], activation=nn_ops.relu, use_bias=False)
    output = layer.apply(images)
    self.assertEqual(output.op.name, 'conv2d/Relu')
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height - 2, width - 2, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
    self.assertEqual(layer.bias, None)

  def testDilatedConv2D(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.Conv2D(32, [3, 3], dilation_rate=3)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, 1, 3, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

    # Test tuple dilation rate
    layer = conv_layers.Conv2D(32, [3, 3], dilation_rate=(1, 3))
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, height - 2, 3, 32])

  def testFunctionalConv2DReuse(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    conv_layers.conv2d(images, 32, [3, 3], name='conv1')
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 2)
    conv_layers.conv2d(images, 32, [3, 3], name='conv1', reuse=True)
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 2)

  def testFunctionalConv2DNoReuse(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    conv_layers.conv2d(images, 32, [3, 3])
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 2)
    conv_layers.conv2d(images, 32, [3, 3])
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 4)


class SeparableConv2DTest(test.TestCase):

  def testInvalidDataFormat(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'data_format'):
      conv_layers.separable_conv2d(images, 32, 3, data_format='invalid')

  def testInvalidStrides(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'strides'):
      conv_layers.separable_conv2d(images, 32, 3, strides=(1, 2, 3))

    with self.assertRaisesRegexp(ValueError, 'strides'):
      conv_layers.separable_conv2d(images, 32, 3, strides=None)

  def testInvalidKernelSize(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'kernel_size'):
      conv_layers.separable_conv2d(images, 32, (1, 2, 3))

    with self.assertRaisesRegexp(ValueError, 'kernel_size'):
      conv_layers.separable_conv2d(images, 32, None)

  def testCreateSeparableConv2D(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.SeparableConv2D(32, [3, 3], activation=nn_ops.relu)
    output = layer.apply(images)
    self.assertEqual(output.op.name, 'separable_conv2d/Relu')
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height - 2, width - 2, 32])
    self.assertListEqual(layer.depthwise_kernel.get_shape().as_list(),
                         [3, 3, 4, 1])
    self.assertListEqual(layer.pointwise_kernel.get_shape().as_list(),
                         [1, 1, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateSeparableConv2DDepthMultiplier(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.SeparableConv2D(32, [3, 3], depth_multiplier=2)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height - 2, width - 2, 32])
    self.assertListEqual(layer.depthwise_kernel.get_shape().as_list(),
                         [3, 3, 4, 2])
    self.assertListEqual(layer.pointwise_kernel.get_shape().as_list(),
                         [1, 1, 8, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateSeparableConv2DIntegerKernelSize(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.SeparableConv2D(32, 3)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height - 2, width - 2, 32])
    self.assertListEqual(layer.depthwise_kernel.get_shape().as_list(),
                         [3, 3, 4, 1])
    self.assertListEqual(layer.pointwise_kernel.get_shape().as_list(),
                         [1, 1, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateSeparableConv2DChannelsFirst(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, 4, height, width))
    layer = conv_layers.SeparableConv2D(
        32, [3, 3], data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, 32, height - 2, width - 2])
    self.assertListEqual(layer.depthwise_kernel.get_shape().as_list(),
                         [3, 3, 4, 1])
    self.assertListEqual(layer.pointwise_kernel.get_shape().as_list(),
                         [1, 1, 4, 32])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testSeparableConv2DPaddingSame(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 32), seed=1)
    layer = conv_layers.SeparableConv2D(
        64, images.get_shape()[1:3], padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, height, width, 64])

  def testCreateSeparableConvWithStrides(self):
    height, width = 6, 8
    # Test strides tuple
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    layer = conv_layers.SeparableConv2D(
        32, [3, 3], strides=(2, 2), padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width / 2, 32])

    # Test strides integer
    layer = conv_layers.SeparableConv2D(32, [3, 3], strides=2, padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width / 2, 32])

    # Test unequal strides
    layer = conv_layers.SeparableConv2D(
        32, [3, 3], strides=(2, 1), padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height / 2, width, 32])

  def testFunctionalConv2DReuse(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    conv_layers.separable_conv2d(images, 32, [3, 3], name='sepconv1')
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 3)
    conv_layers.separable_conv2d(
        images, 32, [3, 3], name='sepconv1', reuse=True)
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 3)

  def testFunctionalConv2DNoReuse(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    conv_layers.separable_conv2d(images, 32, [3, 3])
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 3)
    conv_layers.separable_conv2d(images, 32, [3, 3])
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 6)

  def testSeparableConv2DDepthwiseRegularizer(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    layer = conv_layers.SeparableConv2D(32, [3, 3], depthwise_regularizer=reg)
    layer.apply(images)
    loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(layer.losses, loss_keys)

  def testSeparableConv2DPointwiseRegularizer(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    layer = conv_layers.SeparableConv2D(32, [3, 3], pointwise_regularizer=reg)
    layer.apply(images)
    loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(layer.losses, loss_keys)

  def testSeparableConv2DBiasRegularizer(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    layer = conv_layers.SeparableConv2D(32, [3, 3], bias_regularizer=reg)
    layer.apply(images)
    loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(layer.losses, loss_keys)

  def testSeparableConv2DNoBias(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.SeparableConv2D(
        32, [3, 3], activation=nn_ops.relu, use_bias=False)
    output = layer.apply(images)
    self.assertEqual(output.op.name, 'separable_conv2d/Relu')
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height - 2, width - 2, 32])
    self.assertListEqual(layer.depthwise_kernel.get_shape().as_list(),
                         [3, 3, 4, 1])
    self.assertListEqual(layer.pointwise_kernel.get_shape().as_list(),
                         [1, 1, 4, 32])
    self.assertEqual(layer.bias, None)


class Conv2DTransposeTest(test.TestCase):

  def testInvalidDataFormat(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'data_format'):
      conv_layers.conv2d_transpose(images, 32, 3, data_format='invalid')

  def testInvalidStrides(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'strides'):
      conv_layers.conv2d_transpose(images, 32, 3, strides=(1, 2, 3))

    with self.assertRaisesRegexp(ValueError, 'strides'):
      conv_layers.conv2d_transpose(images, 32, 3, strides=None)

  def testInvalidKernelSize(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    with self.assertRaisesRegexp(ValueError, 'kernel_size'):
      conv_layers.conv2d_transpose(images, 32, (1, 2, 3))

    with self.assertRaisesRegexp(ValueError, 'kernel_size'):
      conv_layers.conv2d_transpose(images, 32, None)

  def testCreateConv2DTranspose(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.Conv2DTranspose(32, [3, 3], activation=nn_ops.relu)
    output = layer.apply(images)
    self.assertEqual(output.op.name, 'conv2d_transpose/Relu')
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height + 2, width + 2, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 32, 4])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateConv2DTransposeIntegerKernelSize(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.Conv2DTranspose(32, 3)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height + 2, width + 2, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 32, 4])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testCreateConv2DTransposeChannelsFirst(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, 4, height, width))
    layer = conv_layers.Conv2DTranspose(
        32, [3, 3], data_format='channels_first')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, 32, height + 2, width + 2])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 32, 4])
    self.assertListEqual(layer.bias.get_shape().as_list(), [32])

  def testConv2DTransposePaddingSame(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 32), seed=1)
    layer = conv_layers.Conv2DTranspose(
        64, images.get_shape()[1:3], padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [5, height, width, 64])

  def testCreateConv2DTransposeWithStrides(self):
    height, width = 6, 8
    # Test strides tuple
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    layer = conv_layers.Conv2DTranspose(
        32, [3, 3], strides=(2, 2), padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height * 2, width * 2, 32])

    # Test strides integer
    layer = conv_layers.Conv2DTranspose(32, [3, 3], strides=2, padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height * 2, width * 2, 32])

    # Test unequal strides
    layer = conv_layers.Conv2DTranspose(
        32, [3, 3], strides=(2, 1), padding='same')
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height * 2, width, 32])

  def testConv2DTransposeKernelRegularizer(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    layer = conv_layers.Conv2DTranspose(32, [3, 3], kernel_regularizer=reg)
    layer.apply(images)
    loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(layer.losses, loss_keys)

  def testConv2DTransposeBiasRegularizer(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    layer = conv_layers.Conv2DTranspose(32, [3, 3], bias_regularizer=reg)
    layer.apply(images)
    loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(layer.losses, loss_keys)

  def testConv2DTransposeNoBias(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 4))
    layer = conv_layers.Conv2DTranspose(
        32, [3, 3], activation=nn_ops.relu, use_bias=False)
    output = layer.apply(images)
    self.assertEqual(output.op.name, 'conv2d_transpose/Relu')
    self.assertListEqual(output.get_shape().as_list(),
                         [5, height + 2, width + 2, 32])
    self.assertListEqual(layer.kernel.get_shape().as_list(), [3, 3, 32, 4])
    self.assertEqual(layer.bias, None)

  def testFunctionalConv2DTransposeReuse(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    conv_layers.conv2d_transpose(images, 32, [3, 3], name='deconv1')
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 2)
    conv_layers.conv2d_transpose(images, 32, [3, 3], name='deconv1', reuse=True)
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 2)

  def testFunctionalConv2DTransposeNoReuse(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    conv_layers.conv2d_transpose(images, 32, [3, 3])
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 2)
    conv_layers.conv2d_transpose(images, 32, [3, 3])
    self.assertEqual(
        len(ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)), 4)


if __name__ == '__main__':
  test.main()

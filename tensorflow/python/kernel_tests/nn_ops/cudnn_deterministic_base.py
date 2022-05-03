# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for deterministic cuDNN functionality."""

import collections

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

# Notes:
#
# TensorFlow makes cuDNN run deterministically when op determinism is enabled
# via tf.config.experimental.enable_op_determinism(). Additionally, setting the
# environmental variable TF_CUDNN_DETERMINISTIC to 'true' or '1' makes cuDNN run
# deterministically, although this environemtnal variable is deprecated and will
# be removed in a future TensorFlow version. Unlike the enable_op_determinism()
# function, the environmental variable only makes ops using cuDNN deterministic,
# not all TensorFlow ops.
#
# Where both deterministic and non-deterministic cuDNN algorithms are available,
# selecting determinitic operation will lead to only the deterministic
# algorithms being chosen. Additionally, selecting deterministic operation will
# result in a deterministic, or reproducible, selection of algorithms (for any
# given layer configuration) for each of the forward and the two backward paths.
#
# These tests intend to confirm that deterministic algorithms are chosen (for
# the back-prop paths) when desterministic operation is selected. The tested
# configurations were first confirmed to produce non-deterministic results when
# the above-mentioned environment variables are not set.
#
# Even though selecting determinitic operation should ensure that the same
# algorithms, for a given layer configuration, are always used (i.e. that
# algorithm selection is deterministic / reproducible), this is not tested.

# TODO(duncanriach): Add test for deterministic cuDNN max-pooling

LayerShapeNHWC = collections.namedtuple('LayerShapeNHWC',
                                        'batch, height, width, channels')
FilterShape2D = collections.namedtuple(
    'FilterShape2D', 'height, width, in_channels, out_channels')
FilterShape2DTranspose = collections.namedtuple(
    'FilterShape2DTranspose', 'height, width, out_channels, in_channels')

LayerShapeNCDHW = collections.namedtuple(
    'LayerShapeNCDHW', 'batch, channels, depth, height, width')
FilterShape3D = collections.namedtuple(
    'FilterShape3D', 'depth, height, width, in_channels, out_channels')


class ConvolutionTest(test.TestCase):
  """Tests for deterministic cuDNN functionality."""

  def _random_data_op(self, shape):
    # np.random.random_sample can properly interpret either tf.TensorShape or
    # namedtuple as a list.
    return constant_op.constant(
        2 * np.random.random_sample(shape) - 1, dtype=dtypes.float32)

  def _random_out_op(self, in_shape, filter_shape, strides, padding, dilations):
    # Choosing not to use array_op.zeros() to prevent possible removal by
    # optimization
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    # Use the forward op's shape-inference
    conv_op = nn_ops.conv2d(
        in_op, filter_op, strides=strides, padding=padding, dilations=dilations)
    out_shape = conv_op.get_shape()
    out_op = self._random_data_op(out_shape)
    return out_op

  def _assert_reproducible(self, operation):
    with test_util.force_gpu():
      result_1 = operation()
      result_2 = operation()
    self.assertAllEqual(result_1, result_2)

  # The default forward algorithm choice, when using cuDNN 7, does not support
  # the following layer configuration. This test case intends to confirm that
  # an alternative algorithm is selected. Note that, in cuDNN 7, all forward
  # algorithms are determnistic.
  @test_util.run_cuda_only
  def testConvForwardDefaultAlgorithmChoice(self):
    in_shape = LayerShapeNCDHW(batch=2, channels=3, depth=5, height=7, width=6)
    filter_shape = FilterShape3D(
        depth=3, height=3, width=3, in_channels=3, out_channels=2)
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    self._assert_reproducible(lambda: nn_ops.conv3d(
        in_op, filter_op, strides=[1, 1, 1, 1, 1], padding='VALID',
        data_format='NCDHW', dilations=[1, 1, 2, 2, 2]))

  # This test is primarily testing XLA since cuDNN forward convolutions are
  # always deterministic, even when determinism is not enabled. The convolution
  # configuration tested is nondeterministic with XLA when determinism is not
  # enabled.
  @test_util.run_cuda_only
  def testConvForwardXLA(self):
    in_shape = LayerShapeNCDHW(
        batch=2, channels=8, depth=5, height=12, width=15)
    filter_shape = FilterShape3D(
        depth=3, height=3, width=3, in_channels=8, out_channels=1)
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    self._assert_reproducible(lambda: nn_ops.conv3d(
        in_op, filter_op, strides=[1, 1, 1, 1, 1], padding='VALID',
        data_format='NCDHW', dilations=[1, 1, 2, 2, 2]))

  @test_util.run_cuda_only
  def testConvBackwardFilterGradient(self, rate=1):
    in_shape = LayerShapeNHWC(batch=8, height=64, width=64, channels=8)
    filter_shape = FilterShape2D(
        height=3, width=3, in_channels=8, out_channels=8)
    in_op = self._random_data_op(in_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    dilations = [1, rate, rate, 1]
    out_op = self._random_out_op(
        in_shape, filter_shape, strides, padding, dilations)
    self._assert_reproducible(lambda: nn_ops.conv2d_backprop_filter(
        in_op, filter_shape, out_op, strides=strides, padding=padding,
        dilations=dilations))

  # A configuration for this test could not be found that exercises
  # nondeterminism when using XLA with determinism not enabled.
  @test_util.run_cuda_only
  def testConvBackwardFilterGradientWithDilations(self):
    self.testConvBackwardFilterGradient(rate=2)

  @test_util.run_cuda_only
  def testConvBackwardInputGradient(self, rate=1):
    in_shape = LayerShapeNHWC(batch=1, height=16, width=16, channels=1)
    filter_shape = FilterShape2D(
        height=7, width=7, in_channels=1, out_channels=3)
    filter_op = self._random_data_op(filter_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    dilations = [1, rate, rate, 1]
    out_op = self._random_out_op(
        in_shape, filter_shape, strides, padding, dilations)
    self._assert_reproducible(lambda: nn_ops.conv2d_backprop_input(
        in_shape, filter_op, out_op, strides=strides, padding=padding,
        dilations=dilations))

  # A configuration for this test could not be found that exercises
  # nondeterminism when using XLA with determinism not enabled.
  @test_util.run_cuda_only
  def testConvBackwardInputGradientWithDilations(self):
    self.testConvBackwardInputGradient(rate=2)

  @test_util.run_cuda_only
  def testConvTransposeForward(self, rate=1):
    in_channels = 3; out_channels = 1
    in_shape = LayerShapeNHWC(
        batch=1, height=16, width=16, channels=in_channels)
    filter_shape = FilterShape2DTranspose(
        height=7, width=7, out_channels=out_channels, in_channels=in_channels)
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    out_shape = LayerShapeNHWC(
        batch=in_shape.batch, height=in_shape.height, width=in_shape.width,
        channels=out_channels)
    self._assert_reproducible(lambda: nn_ops.conv2d_transpose_v2(
        in_op, filter_op, out_shape, strides=1, padding='SAME',
        data_format='NHWC', dilations=[1, rate, rate, 1]))

  # A configuration for this test could not be found that exercises
  # nondeterminism when using XLA with determinism not enabled.
  @test_util.run_cuda_only
  def testConvTransposeForwardWithDilations(self):
    self.testConvTransposeForward(rate=2)

  @test_util.run_cuda_only
  def testConvTransposeBackwardFilterGradient(self, rate=1):
    in_channels = 8; out_channels = 8
    in_shape = LayerShapeNHWC(
        batch=8, height=64, width=64, channels=in_channels)
    filter_shape = FilterShape2DTranspose(
        height=3, width=3, out_channels=out_channels, in_channels=in_channels)
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    out_shape = LayerShapeNHWC(
        batch=in_shape.batch, height=in_shape.height, width=in_shape.width,
        channels=out_channels)
    upstream_gradients = self._random_data_op(out_shape)

    def gradient():
      with backprop.GradientTape() as tape:
        tape.watch(filter_op)
        op_output = nn_ops.conv2d_transpose_v2(
            in_op, filter_op, out_shape, strides=1, padding='SAME',
            data_format='NHWC', dilations=[1, rate, rate, 1])
        gradient_injector_output = op_output * upstream_gradients
      return tape.gradient(gradient_injector_output, [filter_op])[0]

    self._assert_reproducible(gradient)

  # A configuration for this test could not be found that exercises
  # nondeterminism when using XLA with determinism not enabled.
  @test_util.run_cuda_only
  def testConvTransposeBackwardFilterGradientWithDilations(self):
    self.testConvTransposeBackwardFilterGradient(rate=2)

  # A configuration for this test could not be found that exercises
  # nondeterminism when determinism is not enabled (for either XLA or non-XLA).
  @test_util.run_cuda_only
  def testConvTransposeBackwardInputGradient(self, rate=1):
    in_channels = 1; out_channels = 3
    in_shape = LayerShapeNHWC(
        batch=1, height=16, width=16, channels=in_channels)
    filter_shape = FilterShape2DTranspose(
        height=7, width=7, out_channels=out_channels, in_channels=in_channels)
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    out_shape = LayerShapeNHWC(
        batch=in_shape.batch, height=in_shape.height, width=in_shape.width,
        channels=out_channels)
    upstream_gradients = self._random_data_op(out_shape)

    def gradient():
      with backprop.GradientTape() as tape:
        tape.watch(in_op)
        op_output = nn_ops.conv2d_transpose_v2(
            in_op, filter_op, out_shape, strides=1, padding='SAME',
            data_format='NHWC', dilations=[1, rate, rate, 1])
        gradient_injector_output = op_output * upstream_gradients
      return tape.gradient(gradient_injector_output, [in_op])[0]

    self._assert_reproducible(gradient)

  # A configuration for this test could not be found that exercises
  # nondeterminism when determinism is not enabled (for either XLA or non-XLA).
  @test_util.run_cuda_only
  def testConvTransposeBackwardInputGradientWithDilations(self):
    self.testConvTransposeBackwardInputGradient(rate=2)

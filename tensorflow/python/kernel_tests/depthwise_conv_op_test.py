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
"""Functional tests for depthwise convolutional operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def ConfigsToTest():
  """Iterator for different convolution shapes, strides and paddings.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the depthwise
    convolution parameters.
  """
  input_sizes = [[4, 5, 5, 48], [4, 8, 8, 84], [4, 17, 17, 48], [4, 35, 35, 2],
                 [4, 147, 147, 2], [3, 299, 299, 3]]
  filter_sizes = [[1, 1, 48, 2], [1, 3, 84, 1], [3, 1, 48, 4], [5, 5, 2, 1],
                  [3, 3, 2, 8], [2, 2, 3, 8]]
  out_sizes = [[4, 5, 5, 96], [4, 8, 8, 84], [4, 17, 17, 192], [4, 35, 35, 2],
               [4, 49, 49, 16], [3, 150, 150, 24]]
  strides = [1, 1, 1, 1, 3, 2]
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [SAME, SAME, SAME, SAME, VALID, SAME, SAME]
  for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                           paddings):
    yield i, f, o, s, p


def CheckGradConfigsToTest():
  """Iterator for different convolution shapes, strides and paddings.

  compute_gradient_error() is very expensive. So the configs should be
  relatively small.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the depthwise
    convolution parameters.
  """
  input_sizes = [[2, 5, 8, 1], [4, 5, 5, 1], [2, 4, 4, 2], [1, 15, 15, 2],
                 [2, 15, 16, 1]]
  filter_sizes = [[4, 4, 1, 2], [2, 2, 1, 2], [3, 1, 2, 2], [1, 3, 2, 1],
                  [3, 3, 1, 2]]
  out_sizes = [[2, 5, 8, 2], [4, 2, 2, 2], [2, 4, 4, 4], [1, 15, 15, 2],
               [2, 5, 5, 2]]
  strides = [1, 2, 1, 1, 3]
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [SAME, VALID, SAME, SAME, VALID]
  for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                           paddings):
    yield i, f, o, s, p


class DepthwiseConv2DTest(tf.test.TestCase):

  # This is testing against the output of the implementation using the
  # combination of conv_2d and slicing ops.
  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    use_gpu):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [filter_rows, filter_cols, input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      use_gpu: Whether to use GPU.
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.test_session(use_gpu=use_gpu) as sess:
      t1 = tf.constant(x1, shape=tensor_in_sizes)
      t1.set_shape(tensor_in_sizes)
      t2 = tf.constant(x2, shape=filter_in_sizes)
      conv_native = tf.nn.depthwise_conv2d_native(
          t1,
          t2,
          strides=[1, stride, stride, 1],
          padding=padding)

      conv_gold = tf.nn.depthwise_conv2d(t1,
                                         t2,
                                         strides=[1, stride, stride, 1],
                                         padding=padding)
      native_result = sess.run(conv_native)
      gold_result = sess.run(conv_gold)

    print("diff matrix:",
          np.amax(np.ravel(native_result) - np.ravel(gold_result)))
    self.assertArrayNear(np.ravel(native_result), np.ravel(gold_result), 1e-5)
    self.assertShapeEqual(native_result, conv_native)
    self.assertShapeEqual(native_result, conv_gold)

  def testDepthwiseConv2D(self):
    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Processing ", index, "th config.")
      if index == 2:
        self._VerifyValues(input_size,
                           filter_size,
                           stride,
                           padding,
                           use_gpu=True)
      self._VerifyValues(input_size,
                         filter_size,
                         stride,
                         padding,
                         use_gpu=False)

# This is testing against hand calculated results.
  def _VerifyHandValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                        expected, use_gpu):
    """Verifies the output values of the depthwise convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [filter_rows, filter_cols, input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether to use GPU.
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.test_session(use_gpu=use_gpu) as sess:
      t1 = tf.constant(x1, shape=tensor_in_sizes)
      t1.set_shape(tensor_in_sizes)
      t2 = tf.constant(x2, shape=filter_in_sizes)
      conv = tf.nn.depthwise_conv2d_native(t1,
                                           t2,
                                           strides=[1, stride, stride, 1],
                                           padding=padding)
      value = sess.run(conv)
    print("value = ", value)
    self.assertArrayNear(expected, np.ravel(value), 1e-5)
    self.assertShapeEqual(value, conv)

  def testConv2D2x2Filter(self):
    # The inputs look like this (it's a 3 x 2 matrix, each of depth 2):
    #
    # [ (1.0, 2.0), (3.0,  4.0), ( 5.0,  6.0) ]
    # [ (7.0, 8.0), (9.0, 10.0), (11.0, 12.0) ]
    #  We can view this as two inputs
    #
    #  input depth 0:
    #
    #  [ 1.0,  3.0,  5.0 ]
    #  [ 7.0,  9.0, 11.0 ]
    #
    #  input depth 1:
    #
    #  [ 2.0,  4.0,  6.0 ]
    #  [ 8.0, 10.0, 12.0 ]
    #
    # The filter looks like this (it has two 2 x 2 patches, each generating 2
    # depths):
    #
    #  filter #0:
    #
    #  [ (1.0,  3.0), ( 5.0,  7.0)]
    #  [ (9.0, 11.0), (13.0, 15.0)]
    #
    #  filter #1:
    #
    #  [ ( 2.0,  4.0), ( 6.0,  8.0)]
    #  [ (10.0, 12.0), (14.0, 16.0)]
    #
    # So the outputs are:
    #
    # (position 0, 0: in_depth 0, output_depth 0 -- using filter #0)
    #  1.0 * 1.0 + 7.0 * 9.0 + 3.0 * 5.0 + 9.0 * 13.0 = 196
    # (position 0, 0: in_depth 0, output_depth 1 -- using filter #1)
    #  1.0 * 2.0 + 7.0 * 10.0 + 3.0 * 6.0 + 9.0 * 14.0 = 216
    # (position 0, 0: in_depth 1, output_depth 2 -- using filter #0)
    #  2.0 * 3.0 + 8.0 * 11.0 + 4.0 * 7.0 + 10.0 * 15.0 = 272
    # (position 0, 0: in_depth 1, output_depth 3 -- using filter #1)
    #  2.0 * 4.0 + 8.0 * 12.0 + 4.0 * 8.0 + 10.0 * 16.0 = 296
    #
    # (position 1, 0: in_depth 0, output_depth 0 -- using filter #0)
    #  3.0 * 1.0 + 9.0 * 9.0 + 5.0 * 5.0 + 11.0 * 13.0 = 252
    # (position 1, 0: in_depth 0, output_depth 1 -- using filter #1)
    #  3.0 * 2.0 + 9.0 * 10.0 + 5.0 * 6.0 + 11.0 * 14.0 = 280
    # (position 1, 0: in_depth 1, output_depth 2 -- using filter #0)
    #  4.0 * 3.0 + 10.0 * 11.0 + 6.0 * 7.0 + 12.0 * 15.0 = 344
    # (position 1, 0: in_depth 1, output_depth 3 -- using filter #1)
    #  4.0 * 4.0 + 10.0 * 12.0 + 6.0 * 8.0 + 12.0 * 16.0 = 376
    expected_output = [196, 216, 272, 296, 252, 280, 344, 376]
    self._VerifyHandValues(tensor_in_sizes=[1, 2, 3, 2],
                           filter_in_sizes=[2, 2, 2, 2],
                           stride=1,
                           padding="VALID",
                           expected=expected_output,
                           use_gpu=False)

    self._VerifyHandValues(tensor_in_sizes=[1, 2, 3, 2],
                           filter_in_sizes=[2, 2, 2, 2],
                           stride=1,
                           padding="VALID",
                           expected=expected_output,
                           use_gpu=True)

  # Gradient checkers.This tests depthwise gradient computations for both
  # BackpropFilter and BackpropInput by comparing gradients computed by the
  # depthwise gradient ops with the gradients computed numerically (details can
  # be found in the compute_gradient_error().
  # Note this check is very expensive so the input should not be too big.
  def _ConstructAndTestGradient(self, input_shape, filter_shape, output_shape,
                                stride, padding, data_type, test_input,
                                use_gpu):
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    with self.test_session(use_gpu=use_gpu):
      if data_type == tf.float32:
        tolerance = 0.002
      else:
        self.assertEqual(data_type, tf.float64)
        tolerance = 1e-8

      input_tensor = tf.constant(input_data,
                                 shape=input_shape,
                                 dtype=data_type,
                                 name="input")
      filter_tensor = tf.constant(filter_data,
                                  shape=filter_shape,
                                  dtype=data_type,
                                  name="filter")
      depthwise_conv2d = tf.nn.depthwise_conv2d_native(input_tensor,
                                                       filter_tensor,
                                                       [1, stride, stride, 1],
                                                       padding,
                                                       name="depthwise_conv2d")
      self.assertEqual(output_shape, depthwise_conv2d.get_shape())
      if test_input:
        err = tf.test.compute_gradient_error(input_tensor, input_shape,
                                             depthwise_conv2d, output_shape)
      else:
        err = tf.test.compute_gradient_error(filter_tensor, filter_shape,
                                             depthwise_conv2d, output_shape)
      print("depthwise conv_2d gradient error = ", err)
      self.assertLess(err, tolerance)

  def testDepthwiseConv2DInputGrad(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(CheckGradConfigsToTest()):
      print("Processing ", index, "th config.")
      for use_gpu in [True, False]:
        self._ConstructAndTestGradient(input_size,
                                       filter_size,
                                       output_size,
                                       stride,
                                       padding,
                                       tf.float32,
                                       test_input=True,
                                       use_gpu=use_gpu)

  def testDepthwiseConv2DFilterGrad(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(CheckGradConfigsToTest()):
      print("Processing ", index, "th config.")
      for use_gpu in [True, False]:
        self._ConstructAndTestGradient(input_size,
                                       filter_size,
                                       output_size,
                                       stride,
                                       padding,
                                       tf.float32,
                                       test_input=False,
                                       use_gpu=use_gpu)

  def _CompareBackpropInputFloat(self, input_sizes, filter_sizes, output_sizes,
                                 stride, padding):
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_gpu):
      with self.test_session(use_gpu=use_gpu):
        t0 = tf.constant(input_sizes, shape=[len(input_sizes)])
        t1 = tf.constant(x1, shape=filter_sizes)
        t2 = tf.constant(x2, shape=output_sizes)
        backprop = tf.nn.depthwise_conv2d_native_backprop_input(
            t0,
            t1,
            t2,
            strides=[1, stride, stride, 1],
            padding=padding)
        ret = backprop.eval()
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_gpu=True)
    cpu_value = _GetVal(use_gpu=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def _CompareBackpropInputDouble(self, input_sizes, filter_sizes, output_sizes,
                                  stride, padding):
    x1 = np.random.rand(*filter_sizes).astype(np.float64)
    x2 = np.random.rand(*output_sizes).astype(np.float64)

    def _GetVal(use_gpu):
      with self.test_session(use_gpu=use_gpu):
        t0 = tf.constant(input_sizes, shape=[len(input_sizes)])
        t1 = tf.constant(x1, shape=filter_sizes)
        t2 = tf.constant(x2, shape=output_sizes)
        backprop = tf.nn.depthwise_conv2d_native_backprop_input(
            t0,
            t1,
            t2,
            strides=[1, stride, stride, 1],
            padding=padding)
        ret = backprop.eval()
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_gpu=True)
    cpu_value = _GetVal(use_gpu=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def testDepthwiseConv2DInputGradCompare(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Processing ", index, "th config.")
      self._CompareBackpropInputFloat(input_size, filter_size, output_size,
                                      stride, padding)
      self._CompareBackpropInputDouble(input_size, filter_size, output_size,
                                       stride, padding)

  def _CompareBackpropFilterFloat(self, input_sizes, filter_sizes, output_sizes,
                                  stride, padding):
    x0 = np.random.rand(*input_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_gpu):
      with self.test_session(use_gpu=use_gpu):
        t0 = tf.constant(x0, shape=input_sizes)
        t1 = tf.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = tf.constant(x2, shape=output_sizes)
        backprop = tf.nn.depthwise_conv2d_native_backprop_filter(
            t0,
            t1,
            t2,
            strides=[1, stride, stride, 1],
            padding=padding)
        ret = backprop.eval()
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_gpu=True)
    cpu_value = _GetVal(use_gpu=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def _CompareBackpropFilterDouble(self, input_sizes, filter_sizes,
                                   output_sizes, stride, padding):
    x0 = np.random.rand(*input_sizes).astype(np.float64)
    x2 = np.random.rand(*output_sizes).astype(np.float64)

    def _GetVal(use_gpu):
      with self.test_session(use_gpu=use_gpu):
        t0 = tf.constant(x0, shape=input_sizes)
        t1 = tf.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = tf.constant(x2, shape=output_sizes)
        backprop = tf.nn.depthwise_conv2d_native_backprop_filter(
            t0,
            t1,
            t2,
            strides=[1, stride, stride, 1],
            padding=padding)
        ret = backprop.eval()
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_gpu=True)
    cpu_value = _GetVal(use_gpu=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def testDepthwiseConv2DFilterGradCompare(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Processing ", index, "th config.")
      self._CompareBackpropFilterFloat(input_size, filter_size, output_size,
                                       stride, padding)
      self._CompareBackpropFilterDouble(input_size, filter_size, output_size,
                                        stride, padding)


if __name__ == "__main__":
  tf.test.main()

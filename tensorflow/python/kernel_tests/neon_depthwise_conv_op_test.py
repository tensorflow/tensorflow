# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for neon kernel for depthwise convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def ConfigsToTest():
  """Iterator for different convolution shapes, strides and paddings.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the depthwise
    convolution parameters.
  """
  input_sizes = [[4, 5, 5, 48], [4, 8, 8, 84], [4, 17, 17, 48], [4, 35, 35, 2],
                 [4, 147, 147, 2], [3, 299, 299, 3], [5, 183, 183, 1]]
  filter_sizes = [[1, 1, 48, 2], [1, 3, 84, 1], [3, 1, 48, 4], [5, 5, 2, 1],
                  [3, 3, 2, 8], [2, 2, 3, 8], [5, 5, 1, 2]]
  out_sizes = [[4, 5, 5, 96], [4, 8, 8, 84], [4, 17, 17, 192], [4, 35, 35, 2],
               [4, 49, 49, 16], [3, 150, 150, 24], [5, 92, 92, 2]]
  strides = [1, 1, 1, 1, 3, 2, 2]
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [SAME, SAME, SAME, SAME, VALID, SAME, SAME, SAME]
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


class DepthwiseConv2DTest(test.TestCase):

  # This is testing that depthwise_conv2d and depthwise_conv2d_native
  # produce the same results.  It also tests that NCHW and NHWC
  # formats agree, by comparing the depthwise_conv2d_native with
  # 'NCHW' format (with transposition) matches the 'NHWC' format using
  # the higher level interface.
  def _VerifyValues(self,
                    tensor_in_sizes,
                    filter_in_sizes,
                    stride,
                    padding,
                    use_gpu,
                    data_format="NHWC"):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [filter_rows, filter_cols, input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      use_gpu: Whether to use GPU.
      data_format: The data_format of the input.  "NHWC" or "NCHW".
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input and filter tensor with numbers incrementing from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.test_session(use_gpu=use_gpu) as sess:
      with sess.graph._kernel_label_map({"DepthwiseConv2dNative": "neon"}):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t1.set_shape(tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)

      native_t1 = t1
      strides = [1, stride, stride, 1]
      if data_format == "NCHW":
        # Transpose from NHWC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        native_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
        strides = [1, 1, stride, stride]

      conv_native = nn_ops.depthwise_conv2d_native(
          native_t1,
          t2,
          strides=strides,
          data_format=data_format,
          padding=padding)

      if data_format == "NCHW":
        # Transpose back from NCHW to NHWC
        conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])

      conv_interface = nn_impl.depthwise_conv2d(
          t1, t2, strides=[1, stride, stride, 1], padding=padding)

      native_result = sess.run(conv_native)
      interface_result = sess.run(conv_interface)

    print("depthwise conv_2d: ", tensor_in_sizes, "*", filter_in_sizes,
          ", stride:", stride, ", padding: ", padding, ", max diff: ",
          np.amax(np.absolute(native_result - interface_result)))
    self.assertAllClose(
        np.ravel(native_result), np.ravel(interface_result), 1e-5)
    self.assertShapeEqual(native_result, conv_native)
    self.assertShapeEqual(native_result, conv_interface)

  def testDepthwiseConv2D(self):
    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Processing ", index, "th config.")
      if index == 2:
        self._VerifyValues(
            input_size, filter_size, stride, padding, use_gpu=True)
      self._VerifyValues(
          input_size, filter_size, stride, padding, use_gpu=False)

  def testDepthwiseConv2DFormat(self):
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Processing ", index, "th config.")
      self._VerifyValues(
          input_size,
          filter_size,
          stride,
          padding,
          use_gpu=True,
          data_format="NCHW")

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
      with sess.graph._kernel_label_map({"DepthwiseConv2dNative": "neon"}):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t1.set_shape(tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        conv = nn_ops.depthwise_conv2d_native(
            t1, t2, strides=[1, stride, stride, 1], padding=padding)
        value = sess.run(conv)
    print("value = ", value)
    self.assertAllClose(expected, np.ravel(value), 1e-5)
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
    self._VerifyHandValues(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        stride=1,
        padding="VALID",
        expected=expected_output,
        use_gpu=False)

    self._VerifyHandValues(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        stride=1,
        padding="VALID",
        expected=expected_output,
        use_gpu=True)


if __name__ == "__main__":
  test.main()

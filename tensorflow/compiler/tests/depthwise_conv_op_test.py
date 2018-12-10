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
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


# Reference implementation of depthwise_conv2d
def ReferenceDepthwiseConv2D(input_tensor, filter_tensor, strides, padding,
                             data_format=None):
  # Reference implementation of depthwise convolution that uses regular
  # convolution.
  convs = []
  in_channels = filter_tensor.shape[2]
  # Use a custom implementation of depthwise conv2d using slicing.
  for channel in xrange(in_channels):
    # Slice the input along channel
    if data_format == "NCHW":
      input_slice = input_tensor[:, channel:channel+1, :, :]
    else:
      input_slice = input_tensor[:, :, :, channel:channel+1]

    # Slice the filters.  Filters are  H, W, InC, DepthMultiplier
    filter_slice = filter_tensor[:, :, channel:channel+1, :]
    # Do conv
    convs.append(nn_ops.conv2d(input_slice, filter_slice,
                               strides, padding,
                               data_format=data_format,
                               name="depthwise_slice_%d" % channel))

  # Concat along dimension.
  if data_format == "NCHW":
    return array_ops.concat(convs, 1)
  else:
    return array_ops.concat(convs, 3)


def ConfigsToTest():
  """Iterator for different convolution shapes, strides and paddings.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the depthwise
    convolution parameters.
  """
  input_sizes = [[4, 5, 5, 48], [4, 8, 8, 84], [4, 17, 17, 48], [4, 9, 27, 8],
                 [4, 31, 31, 7], [4, 35, 35, 2], [4, 147, 147, 2],
                 [3, 299, 299, 3], [5, 183, 183, 1]]
  filter_sizes = [[1, 1, 48, 2], [1, 3, 84, 1], [3, 1, 48, 4], [3, 3, 8, 1],
                  [3, 3, 7, 1], [5, 5, 2, 1], [3, 3, 2, 8], [2, 2, 3,
                                                             8], [5, 5, 1, 2]]
  out_sizes = [[4, 5, 5, 96], [4, 8, 8, 84], [4, 17, 17, 192], [4, 9, 27, 8],
               [4, 31, 31, 7], [4, 35, 35, 2], [4, 49, 49, 16],
               [3, 150, 150, 24], [5, 92, 92, 2]]
  strides = [1, 1, 1, 1, 1, 1, 3, 2, 2]
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [SAME, SAME, SAME, SAME, SAME, SAME, VALID, SAME, SAME, SAME]
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


class DepthwiseConv2DTest(xla_test.XLATestCase):

  # This is testing that depthwise_conv2d and depthwise_conv2d_native
  # produce the same results.  It also tests that NCHW and NWHC
  # formats agree, by comparing the depthwise_conv2d_native with
  # 'NCHW' format (with transposition) matches the 'NHWC' format using
  # the higher level interface.
  def _VerifyValues(self,
                    tensor_in_sizes,
                    filter_in_sizes,
                    stride,
                    padding,
                    data_type,
                    data_format="NHWC"):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [filter_rows, filter_cols, input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      data_type: The data type to use.
      data_format: The data_format of the input. "NHWC" or "NCHW".
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input and filter tensor with numbers incrementing from 1.
    x1 = np.array([f * 1.0 for f in range(1, total_size_1 + 1)],
                  dtype=data_type).reshape(tensor_in_sizes)
    x2 = np.array([f * 1.0 for f in range(1, total_size_2 + 1)],
                  dtype=data_type).reshape(filter_in_sizes)
    with self.cached_session() as sess:
      if data_type == np.float32:
        tolerance = 1e-4
      else:
        self.assertEqual(data_type, np.float64)
        tolerance = 1e-8

      t1 = array_ops.placeholder(shape=tensor_in_sizes, dtype=data_type)
      t2 = array_ops.placeholder(shape=filter_in_sizes, dtype=data_type)

      native_t1 = t1
      strides = [1, stride, stride, 1]
      if data_format == "NCHW":
        # Transpose from NWHC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        native_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
        strides = [1, 1, stride, stride]

      with self.test_scope():
        conv_native = nn_ops.depthwise_conv2d_native(
            native_t1,
            t2,
            strides=strides,
            data_format=data_format,
            padding=padding)

      if data_format == "NCHW":
        # Transpose back from NCHW to NHWC
        conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])

      with ops.device("CPU"):
        conv_interface = ReferenceDepthwiseConv2D(
            t1, t2, strides=[1, stride, stride, 1], padding=padding)

      native_result = sess.run(conv_native, {t1: x1, t2: x2})
      interface_result = sess.run(conv_interface, {t1: x1, t2: x2})

    print("data_type:", data_type, "max diff = ",
          np.amax(np.absolute(native_result - interface_result)))
    self.assertAllClose(
        np.ravel(native_result), np.ravel(interface_result), rtol=tolerance)

  def testDepthwiseConv2D(self):
    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2D,", index, "th config:", input_size, "*",
            filter_size, "stride:", stride, "padding:", padding)
      for data_type in self.float_types:
        # TODO(phawkins): the reference implementation only supports float32.
        if data_type == np.float32:
          self._VerifyValues(
              input_size, filter_size, stride, padding, data_type)

  def testDepthwiseConv2DFormat(self):
    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2DFormat,", index, "th config:", input_size,
            "*", filter_size, "stride:", stride, "padding:", padding)
      for data_type in self.float_types:
        # TODO(phawkins): the reference implementation only supports float32.
        if data_type == np.float32:
          self._VerifyValues(
              input_size,
              filter_size,
              stride,
              padding,
              data_type,
              data_format="NCHW")

# This is testing against hand calculated results.

  def _VerifyHandValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                        expected):
    """Verifies the output values of the depthwise convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [filter_rows, filter_cols, input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = np.array([f * 1.0 for f in range(1, total_size_1 + 1)],
                  dtype=np.float32).reshape(tensor_in_sizes)
    x2 = np.array([f * 1.0 for f in range(1, total_size_2 + 1)],
                  dtype=np.float32).reshape(filter_in_sizes)
    with self.cached_session() as sess:
      t1 = array_ops.placeholder(shape=tensor_in_sizes, dtype=np.float32)
      t2 = array_ops.placeholder(shape=filter_in_sizes, dtype=np.float32)
      with self.test_scope():
        conv = nn_ops.depthwise_conv2d_native(
            t1, t2, strides=[1, stride, stride, 1], padding=padding)
      value = sess.run(conv, {t1: x1, t2: x2})
    print("value = ", value)
    self.assertArrayNear(expected, np.ravel(value), 1e-4)
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
        expected=expected_output)

  def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes,
                            stride, padding):
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_xla):
      with self.cached_session():
        t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
        t1 = array_ops.placeholder(np.float32, shape=filter_sizes)
        t2 = array_ops.placeholder(np.float32, shape=output_sizes)
        if use_xla:
          with self.test_scope():
            backprop = nn_ops.depthwise_conv2d_native_backprop_input(
                t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        else:
          backprop = nn_ops.depthwise_conv2d_native_backprop_input(
              t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)

        ret = backprop.eval({t1: x1, t2: x2})
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_xla=True)
    cpu_value = _GetVal(use_xla=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-3, atol=1e-3)

  def testDepthwiseConv2DInputGradCompare(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2DInputGradCompare,", index, "th config:",
            input_size, "*", filter_size, "stride:", stride, "padding:",
            padding)
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding)

  def _CompareBackpropFilter(self, input_sizes, filter_sizes, output_sizes,
                             stride, padding):
    x0 = np.random.rand(*input_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_xla):
      with self.cached_session():
        t0 = array_ops.placeholder(np.float32, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = array_ops.placeholder(np.float32, shape=output_sizes)
        if use_xla:
          with self.test_scope():
            backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
                t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        else:
          backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
              t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        ret = backprop.eval({t0: x0, t2: x2})
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_xla=True)
    cpu_value = _GetVal(use_xla=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def testDepthwiseConv2DFilterGradCompare(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      print("Testing DepthwiseConv2DFilterGradCompare,", index, "th config:",
            input_size, "*", filter_size, "producing output", output_size,
            "stride:", stride, "padding:", padding)
      self._CompareBackpropFilter(input_size, filter_size, output_size,
                                  stride, padding)


if __name__ == "__main__":
  test.main()

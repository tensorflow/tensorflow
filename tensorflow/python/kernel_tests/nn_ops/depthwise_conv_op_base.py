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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _DepthwiseConv2dNumpyBasic(x1, x2, strides):
  """Compute depthwise_conv2d using Numpy.

  This allows use to test TensorFlow's depthwise_conv2d by comparing to the
  Numpy version.

  Args:
    x1: The input Numpy array, in NHWC format.
    x2: The filter Numpy array.
    strides: A Python list of 4 elements representing the strides.

  Returns:
    The depthwise conv2d output as a Numpy array.
  """
  n, h, w, c = x1.shape
  fh, fw, c2, o = x2.shape
  assert c == c2
  _, sh, sw, _ = strides
  out_rows = (h - fh + sh) // sh
  out_cols = (w - fw + sw) // sw
  out = np.zeros([n, out_rows, out_cols, c * o])
  for i in range(out_rows):
    for j in range(out_cols):
      for k in range(c):
        start_height = i * sh
        end_height = start_height + fh
        start_width = j * sw
        end_width = start_width + fw
        # multiplied_slice.shape: (b, fh, fw, o)
        multiplied_slice = (
            x1[:, start_height:end_height, start_width:end_width, k, np.newaxis]
            * x2[:, :, k, :])
        # Set a slice of b * o elements of 'out'.
        out[:, i, j, k * o:(k + 1) * o] = np.sum(multiplied_slice, axis=(1, 2))
  return out


def _DepthwiseConv2dNumpy(x1, x2, strides, padding, data_format, dilations):
  """Compute depthwise_conv2d using Numpy.

  This allows use to test TensorFlow's depthwise_conv2d by comparing to the
  Numpy version.

  Unlike `_DepthwiseConv2dNumpyBasic`, this supports more advanced features
  like padding.

  Args:
    x1: The input Numpy array.
    x2: The filter Numpy array.
    strides: A Python list of 4 elements representing the strides.
    padding: The padding. "SAME", "VALID", or a list of explicit paddings.
    data_format: "NHWC" or "NCHW".
    dilations: A list of 2 elements, representing the dilations.

  Returns:
    The depthwise conv2d as a Numpy array.
  """
  if data_format == "NCHW":
    # Transpose arguments to NHWC format.
    x1 = np.transpose(x1, (0, 3, 1, 2))
    strides = [strides[0], strides[3], strides[1], strides[2]]
    if dilations:
      dilations = [dilations[0], dilations[3], dilations[1], dilations[2]]

  if dilations:
    # Dilate the filter so _DepthwiseConv2dNumpyBasic doesn't have to deal with
    # dilations.
    fh, fw, c, o = x2.shape
    new_fh = (fh - 1) * dilations[0] + 1
    new_fw = (fw - 1) * dilations[1] + 1
    new_x2 = np.zeros((new_fh, new_fw, c, o))
    for i in range(fh):
      for j in range(fw):
        new_x2[i * dilations[0], j * dilations[1], ::] = x2[i, j, :, :]
    x2 = new_x2

  # Pad input so _DepthwiseConv2dNumpyBasic doesn't have to deal with padding.
  if padding == "SAME":

    def PaddingsForDim(input_dim, filter_dim, stride):
      """Computes paddings for a single dimension."""
      if input_dim % stride == 0:
        total_padding = max(filter_dim - stride, 0)
      else:
        total_padding = max(filter_dim - (input_dim % stride), 0)
      pad_before = total_padding // 2
      pad_after = total_padding - pad_before
      return pad_before, pad_after

    padding = [(0, 0),
               PaddingsForDim(x1.shape[1], x2.shape[0], strides[1]),
               PaddingsForDim(x1.shape[2], x2.shape[1], strides[2]), (0, 0)]
  elif padding == "VALID":
    padding = [(0, 0)] * 4
  x1 = np.pad(x1, padding, "constant")

  y = _DepthwiseConv2dNumpyBasic(x1, x2, strides)

  if data_format == "NCHW":
    # Transpose back to NCHW format.
    y = np.transpose(y, (0, 2, 3, 1))

  return y


def ConfigsToTest():
  """Iterator for different convolution shapes, strides and paddings.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

  def Config(input_size,
             filter_size,
             out_size,
             stride=1,
             padding="SAME",
             dilations=None):
    return input_size, filter_size, out_size, stride, padding, dilations

  return [
      Config([4, 5, 5, 48], [1, 1, 48, 2], [4, 5, 5, 96]),
      Config([4, 8, 8, 84], [1, 3, 84, 1], [4, 8, 8, 84]),
      Config([4, 17, 17, 48], [3, 1, 48, 4], [4, 17, 17, 192]),
      Config([4, 9, 27, 8], [3, 3, 8, 1], [4, 9, 27, 8]),
      Config([4, 31, 31, 7], [3, 3, 7, 1], [4, 31, 31, 7]),
      Config([4, 35, 35, 2], [5, 5, 2, 1], [4, 35, 35, 2]),
      Config([4, 147, 147, 2], [3, 3, 2, 8], [4, 49, 49, 16],
             3,
             padding="VALID"),
      Config([3, 299, 299, 3], [3, 2, 3, 8], [3, 150, 150, 24], 2),
      Config([5, 183, 183, 1], [5, 5, 1, 2], [5, 92, 92, 2], 2),
      Config([5, 183, 183, 1], [5, 5, 1, 2], [5, 183, 183, 2], dilations=[2,
                                                                          2]),
      Config([5, 41, 35, 2], [4, 7, 2, 2], [5, 32, 23, 4],
             padding="VALID",
             dilations=[3, 2]),
  ]


def ConfigsToTestExplicit():
  """Iterator for different convolution shapes, strides and explicit paddings.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

  def Config(input_size,
             filter_size,
             out_size,
             stride=1,
             padding=None,
             dilations=None):
    return input_size, filter_size, out_size, stride, padding, dilations

  return [
      Config([4, 5, 5, 48], [1, 1, 48, 2], [4, 8, 12, 96],
             padding=[[1, 2], [3, 4]]),
      Config([4, 1, 1, 3], [3, 3, 3, 2], [4, 29, 39, 6],
             padding=[[10, 20], [15, 25]]),
      Config([4, 9, 27, 8], [3, 3, 8, 1], [4, 14, 31, 8],
             padding=[[3, 4], [4, 2]]),
      Config([4, 31, 31, 7], [3, 3, 7, 1], [4, 29, 29, 7],
             padding=[[0, 0], [0, 0]]),
      Config([3, 299, 299, 3], [3, 2, 3, 8], [3, 150, 153, 24],
             2,
             padding=[[1, 2], [3, 5]]),
      Config([5, 183, 183, 1], [5, 5, 1, 2], [5, 62, 60, 2],
             3,
             padding=[[3, 2], [1, 0]]),
      Config([5, 29, 31, 1], [5, 4, 1, 2], [5, 26, 23, 2],
             padding=[[3, 2], [1, 0]],
             dilations=[2, 3]),
      # These cases test the kernels in depthwise_conv_op_gpu.h which are used
      # if the input size is small.
      Config([4, 5, 5, 48], [3, 3, 48, 1], [4, 5, 5, 48],
             padding=[[0, 2], [0, 2]]),
      Config([1, 8, 7, 2], [8, 7, 2, 1], [1, 8, 7, 2], padding=[[0, 7], [3,
                                                                         3]]),
      Config([2, 4, 3, 2], [3, 2, 2, 1], [2, 4, 3, 2], padding=[[2, 0], [1,
                                                                         0]]),
  ]


def CheckGradConfigsToTest():
  """Iterator for different convolution shapes, strides and paddings.

  compute_gradient_error() is very expensive. So the configs should be
  relatively small.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

  def Config(input_size,
             filter_size,
             out_size,
             stride=1,
             padding="SAME",
             dilations=None):
    return input_size, filter_size, out_size, stride, padding, dilations

  return [
      Config([2, 5, 8, 1], [4, 4, 1, 2], [2, 5, 8, 2]),
      Config([4, 5, 5, 1], [2, 2, 1, 2], [4, 2, 2, 2], 2, padding="VALID"),
      Config([2, 4, 4, 2], [3, 1, 2, 2], [2, 4, 4, 4]),
      Config([1, 15, 15, 2], [1, 3, 2, 1], [1, 15, 15, 2]),
      Config([2, 15, 16, 1], [3, 3, 1, 2], [2, 5, 5, 2], 3, padding="VALID"),
      Config([2, 5, 8, 1], [4, 3, 1, 2], [2, 5, 8, 2], dilations=[1, 2]),
      # These cases test the kernels in depthwise_conv_op_gpu.h which are used
      # if the input size is small.
      Config([1, 3, 1, 2], [2, 1, 2, 1], [1, 3, 1, 2]),
      Config([2, 2, 3, 2], [2, 1, 2, 1], [2, 2, 3, 2]),
      Config([2, 2, 3, 1], [2, 2, 1, 1], [2, 2, 3, 1]),
  ]


def CheckGradConfigsToTestExplicit():
  """Iterator for different convolution shapes, strides and explicit paddings.

  compute_gradient_error() is very expensive. So the configs should be
  relatively small.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

  def Config(input_size,
             filter_size,
             out_size,
             stride=1,
             padding=None,
             dilations=None):
    return input_size, filter_size, out_size, stride, padding, dilations

  return [
      Config([2, 5, 8, 1], [4, 4, 1, 2], [2, 3, 10, 2],
             padding=[[0, 1], [2, 3]]),
      Config([4, 5, 5, 1], [2, 2, 1, 2], [4, 4, 5, 2],
             2,
             padding=[[3, 1], [5, 0]]),
      Config([2, 4, 4, 2], [3, 1, 2, 2], [2, 7, 11, 4],
             padding=[[4, 1], [3, 4]]),
      Config([1, 15, 15, 2], [1, 3, 2, 1], [1, 18, 23, 2],
             padding=[[3, 0], [2, 8]]),
      Config([2, 15, 16, 1], [3, 3, 1, 2], [2, 5, 8, 2],
             3,
             padding=[[0, 0], [10, 0]]),
      Config([2, 5, 8, 1], [3, 4, 1, 2], [2, 5, 10, 2],
             padding=[[3, 1], [2, 3]],
             dilations=[2, 1]),
      # These cases test the kernels in depthwise_conv_op_gpu.h which are used
      # if the input size is small.
      Config([2, 4, 3, 2], [3, 2, 2, 1], [2, 4, 3, 2], padding=[[2, 0], [1,
                                                                         0]]),
  ]


class DepthwiseConv2DBase(test.TestCase):
  """Base test class for depthwise Conv2D tests."""

  # This tests depthwise_conv2d and depthwise_conv2d_native
  def _VerifyValues(self,
                    tensor_in_sizes,
                    filter_in_sizes,
                    stride,
                    padding,
                    data_type,
                    use_gpu,
                    grouped_conv=False,
                    data_format="NHWC",
                    dilations=None,
                    tolerance=None):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [filter_rows, filter_cols,
        input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      data_type: The data type to use.
      use_gpu: Whether to use GPU.
      grouped_conv: Whether to use cuDNN 7's grouped convolution.
      data_format: The data_format of the input. "NHWC" or "NCHW".
      dilations: A list of 2 elements, representing the dilations.
      tolerance: The absolute and relative tolarance when verifying the output.
    """
    input_size = 1
    filter_size = 1
    for s in tensor_in_sizes:
      input_size *= s
    for s in filter_in_sizes:
      filter_size *= s
    # Initializes the input and filter tensor with numbers incrementing to 1.0.
    x1 = [f * 1.0 / input_size for f in range(1, input_size + 1)]
    x1 = np.array(x1).reshape(tensor_in_sizes)
    x2 = [f * 1.0 / filter_size for f in range(1, filter_size + 1)]
    x2 = np.array(x2).reshape(filter_in_sizes)
    # Compute reference result
    strides = [1, stride, stride, 1]
    if isinstance(padding, list):
      padding = [(0, 0)] + padding + [(0, 0)]
    np_result = _DepthwiseConv2dNumpy(x1, x2, strides, padding, "NHWC",
                                      dilations)

    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.session(graph=graph, use_gpu=use_gpu) as sess:
      tolerance = tolerance or {
          dtypes.float16: 4e-2,
          dtypes.float32: 1e-5,
          dtypes.float64: 1e-12,
          dtypes.bfloat16: 1e-2,
      }[data_type]

      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=data_type)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=data_type)

      if data_format == "NCHW":
        # Transpose from NHWC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        t1 = array_ops.transpose(t1, [0, 3, 1, 2])
        strides = [1, 1, stride, stride]
        if isinstance(padding, list):
          padding = [padding[0], padding[3], padding[1], padding[2]]

      # depthwise_conv2d_native does not support dilations except on TPUs.
      if dilations is None:
        with sess.graph._kernel_label_map(  # pylint: disable=protected-access
            {"DepthwiseConv2dNative": "cudnn_grouped_convolution"}
            if grouped_conv else {}):
          conv_native = nn_ops.depthwise_conv2d_native(
              t1, t2, strides=strides, data_format=data_format, padding=padding)

        if data_format == "NCHW":
          # Transpose back from NCHW to NHWC
          conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])

        try:
          # The Numpy array from calling depthwise_conv2d_native
          native_result = self.evaluate(conv_native)
        except errors.InvalidArgumentError as e:
          # Grouped convolution kernel is only registered for cuDNN 7. Silently
          # return when we are running on an earlier version or without GPU.
          if ("No OpKernel was registered to support Op "
              "'DepthwiseConv2dNative'") in e.message:
            tf_logging.warn("Skipping grouped convolution test")
            return
          raise e

      conv_interface = nn_impl.depthwise_conv2d(
          t1,
          t2,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
      if data_format == "NCHW":
        # Transpose back from NCHW to NHWC
        conv_interface = array_ops.transpose(conv_interface, [0, 2, 3, 1])

      # The Numpy array from calling depthwise_conv2d
      interface_result = self.evaluate(conv_interface)

    if dilations is None:
      self.assertAllClose(
          native_result, np_result, atol=tolerance, rtol=tolerance)
    self.assertAllClose(
        interface_result, np_result, atol=tolerance, rtol=tolerance)

  @test_util.run_v1_only("b/120545219")
  @test_util.run_cuda_only
  def testDepthwiseConv2DCudnn(self):
    for index, (input_size, filter_size, _, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      # The CuDNN depthwise conv is turned on only when input/output is NCHW and
      # float16(half). See cudnn release note 7.6.3.
      tf_logging.info(
          "Testing DepthwiseConv2DCudnn, %dth config: %r * %r, stride: %d, "
          "padding: %s", index, input_size, filter_size, stride, padding)
      data_type = dtypes.float16
      self._VerifyValues(
          input_size,
          filter_size,
          stride,
          padding,
          data_type,
          use_gpu=True,
          data_format="NCHW",
          dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2D(self):
    for index, (input_size, filter_size, _, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2D, %dth config: %r * %r, stride: %d, padding: "
          "%s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        tf_logging.info("Testing without grouped_conv")
        tolerance = 1e-4 if data_type == dtypes.float32 else 1e-12
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
            use_gpu=True,
            dilations=dilations,
            tolerance=tolerance)
        tf_logging.info("Testing with grouped_conv")
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
            use_gpu=True,
            grouped_conv=True,
            dilations=dilations,
            tolerance=tolerance)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DWithUnknownShape(self):
    # GitHub issue 22110.
    if not test.is_gpu_available():
      return
    with self.session():
      x = array_ops.placeholder(dtypes.float32)
      f = np.ones([1, 1, 1, 1], np.float32)
      v = nn_impl.depthwise_conv2d(
          x, f, [1, 1, 1, 1], "VALID", rate=[2, 1], data_format="NCHW")
      self.assertAllEqual(
          np.ones([1, 1, 1, 1], np.float32),
          v.eval(feed_dict={x: np.ones([1, 1, 1, 1], np.float32)}))

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DFormat(self):
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, _, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFormat, %dth config: %r * %r, stride: %d, "
          "padding: %s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        tolerance = 1e-4 if data_type == dtypes.float32 else 1e-12
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations,
            tolerance=tolerance)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DExplicit(self):
    for index, (input_size, filter_size, _, stride, padding,
                dilations) in enumerate(ConfigsToTestExplicit()):
      tf_logging.info(
          "Testing DepthwiseConv2D, %dth config: %r * %r, stride: %d, padding: "
          "%s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      data_formats = ["NHWC", "NCHW"] if test.is_gpu_available() else ["NHWC"]
      for data_type in [dtypes.float16, dtypes.float32] + optional_float64:
        for data_format in data_formats:
          self._VerifyValues(
              input_size,
              filter_size,
              stride,
              padding,
              data_type,
              use_gpu=True,
              data_format=data_format,
              dilations=dilations)


# This is testing against hand calculated results.

  def _VerifyHandValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                        expected, use_gpu):
    """Verifies the output values of the depthwise convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [filter_rows, filter_cols,
        input_depth, depth_multiplier].
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
    with self.cached_session(use_gpu=use_gpu) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t1.set_shape(tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      conv = nn_ops.depthwise_conv2d_native(
          t1, t2, strides=[1, stride, stride, 1], padding=padding)
      value = self.evaluate(conv)
    tf_logging.info("value = %r", value)
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

  # Gradient checkers. This tests depthwise gradient computations for both
  # BackpropFilter and BackpropInput by comparing gradients computed by the
  # depthwise gradient ops with the gradients computed numerically (details can
  # be found in the compute_gradient_error().
  # Note this check is very expensive so the input should not be too big.
  def _ConstructAndTestGradient(self,
                                input_shape,
                                filter_shape,
                                output_shape,
                                stride,
                                padding,
                                data_type,
                                test_input,
                                use_gpu,
                                grouped_conv=False,
                                data_format="NHWC",
                                dilations=None):
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    input_np = np.array(input_data).reshape(input_shape)
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    filter_np = np.array(filter_data).reshape(filter_shape)
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.session(graph=graph, use_gpu=use_gpu) as sess:
      tolerance = {
          dtypes.float16: 4e-0,
          dtypes.float32: 8e-4,
          dtypes.float64: 1e-12,
      }[data_type]

      input_tensor = constant_op.constant(
          input_np, shape=input_shape, dtype=data_type, name="input")
      filter_tensor = constant_op.constant(
          filter_np, shape=filter_shape, dtype=data_type, name="filter")

      native_input = input_tensor
      strides = [1, stride, stride, 1]
      if isinstance(padding, list):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        # Transpose from NHWC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        native_input = array_ops.transpose(input_tensor, [0, 3, 1, 2])
        input_shape = [
            input_shape[0], input_shape[3], input_shape[1], input_shape[2]
        ]
        output_shape = [
            output_shape[0], output_shape[3], output_shape[1], output_shape[2]
        ]
        strides = [1, 1, stride, stride]
        if isinstance(padding, list):
          padding = [padding[0], padding[3], padding[1], padding[2]]

      with sess.graph._kernel_label_map({  # pylint: disable=protected-access,g-long-ternary
          "DepthwiseConv2dNative": "cudnn_grouped_convolution",
          "DepthwiseConv2dNativeBackpropInput": "cudnn_grouped_convolution",
          "DepthwiseConv2dNativeBackpropFilter": "cudnn_grouped_convolution",
      } if grouped_conv else {}):
        depthwise_conv2d = nn_impl.depthwise_conv2d(
            native_input,
            filter_tensor,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            name="depthwise_conv2d")

      self.assertEqual(output_shape, depthwise_conv2d.get_shape())

      try:
        if test_input:
          err = gradient_checker.compute_gradient_error(native_input,
                                                        input_shape,
                                                        depthwise_conv2d,
                                                        output_shape)
        else:
          err = gradient_checker.compute_gradient_error(filter_tensor,
                                                        filter_shape,
                                                        depthwise_conv2d,
                                                        output_shape)
      except errors.InvalidArgumentError as e:
        # TODO(xjun): Tests depend on error messages could be brittle.
        # Grouped convolution kernel is only registered for cuDNN 7. Silently
        # return when we are running on an earlier version or without GPU.
        if grouped_conv and ("No OpKernel was registered to support Op "
                             "'DepthwiseConv2dNative'") in e.message:
          tf_logging.warn("Skipping grouped convolution test")
          return
        raise e

      tf_logging.info(
          "data_type: %r, use_gpu: %r, grouped_conv: %r, error = %f", data_type,
          use_gpu, grouped_conv, err)
      self.assertLess(err, tolerance)

  @test_util.run_v1_only("b/120545219")
  @test_util.run_cuda_only
  def testDepthwiseConv2DInputGradCudnn(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      # The CuDNN depthwise conv (input gradient) is turned on only when
      # stride = 1, input/output is NCHW and float16(half). See cudnn release
      # note 7.6.3.
      if stride != 1:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradCudnn, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      data_type = dtypes.float16
      self._ConstructAndTestGradient(
          input_size,
          filter_size,
          output_size,
          stride,
          padding,
          data_type,
          test_input=True,
          use_gpu=True,
          data_format="NCHW",
          dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DInputGrad(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGrad, %dth config: %r * %r, stride: %d, "
          "padding: %s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            dilations=dilations)
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            grouped_conv=True,
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DInputGradFormat(self):
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradFormat, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DInputGradExplicit(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTestExplicit()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradExplicit, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      data_formats = ["NHWC", "NCHW"] if test.is_gpu_available() else ["NHWC"]
      for data_type in [dtypes.float16, dtypes.float32] + optional_float64:
        for data_format in data_formats:
          self._ConstructAndTestGradient(
              input_size,
              filter_size,
              output_size,
              stride,
              padding,
              data_type,
              test_input=True,
              use_gpu=True,
              data_format=data_format,
              dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  @test_util.run_cuda_only
  def testDepthwiseConv2DFilterGradCudnn(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      # The CuDNN depthwise conv (filter gradient) is turned on only when
      # input/output is float16(half). See cudnn release note 7.6.3.
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradCudnn, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      data_type = dtypes.float16
      self._ConstructAndTestGradient(
          input_size,
          filter_size,
          output_size,
          stride,
          padding,
          data_type,
          test_input=False,
          use_gpu=True,
          data_format="NCHW",
          dilations=dilations)
      self._ConstructAndTestGradient(
          input_size,
          filter_size,
          output_size,
          stride,
          padding,
          data_type,
          test_input=False,
          use_gpu=True,
          data_format="NHWC",
          dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DFilterGrad(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGrad, %dth config: %r * %r, stride: "
          "%d, padding: %s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float16, dtypes.float32] + optional_float64):
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=False,
            use_gpu=True,
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DFilterGradFormat(self):
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradFormat, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=False,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DFilterGradExplicit(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTestExplicit()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradExplicit, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      data_formats = ["NHWC", "NCHW"] if test.is_gpu_available() else ["NHWC"]
      for data_type in [dtypes.float16, dtypes.float32] + optional_float64:
        for data_format in data_formats:
          self._ConstructAndTestGradient(
              input_size,
              filter_size,
              output_size,
              stride,
              padding,
              data_type,
              test_input=False,
              use_gpu=True,
              data_format=data_format,
              dilations=dilations)

  def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes,
                            stride, padding, dtype):
    x1 = np.random.rand(*filter_sizes).astype(dtype)
    x2 = np.random.rand(*output_sizes).astype(dtype)
    if isinstance(padding, list):
      padding = [(0, 0)] + padding + [(0, 0)]

    def _GetVal(use_gpu):
      with self.cached_session(use_gpu=use_gpu):
        t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
        t1 = constant_op.constant(x1, shape=filter_sizes)
        t2 = constant_op.constant(x2, shape=output_sizes)
        backprop = nn_ops.depthwise_conv2d_native_backprop_input(
            t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        ret = self.evaluate(backprop)
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_gpu=True)
    cpu_value = _GetVal(use_gpu=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def testDepthwiseConv2DInputGradCompare(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding, "float32")
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      if test.is_built_with_rocm():
        continue
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding, "float64")

  def testDepthwiseConv2DInputGradExplicitCompare(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTestExplicit()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding, "float32")
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      if test.is_built_with_rocm():
        continue
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding, "float64")

  def _CompareBackpropFilter(self, input_sizes, filter_sizes, output_sizes,
                             stride, padding, dtype):
    x0 = np.random.rand(*input_sizes).astype(dtype)
    x2 = np.random.rand(*output_sizes).astype(dtype)
    padding_nhwc = padding
    padding_nchw = padding
    if isinstance(padding, list):
      padding_nhwc = [(0, 0)] + padding + [(0, 0)]
      padding_nchw = [(0, 0)] + [(0, 0)] + padding

    def _GetVal(use_gpu, data_format="NHWC"):
      with self.cached_session(use_gpu=use_gpu):
        t0 = constant_op.constant(x0, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes)
        strides = [1, stride, stride, 1]
        padding = padding_nhwc
        if data_format == "NCHW":
          t0 = array_ops.transpose(t0, [0, 3, 1, 2])
          t2 = array_ops.transpose(t2, [0, 3, 1, 2])
          strides = [1, 1, stride, stride]
          padding = padding_nchw
        backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
            t0, t1, t2, strides=strides, padding=padding)
        ret = self.evaluate(backprop)
        self.assertShapeEqual(ret, backprop)
        return ret

    cpu_value = _GetVal(use_gpu=False)
    for data_format in ["NHWC", "NCHW"]:
      gpu_value = _GetVal(use_gpu=True, data_format=data_format)
      self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def testDepthwiseConv2DFilterGradCompare(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropFilter(input_size, filter_size, output_size, stride,
                                  padding, "float32")
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      if test.is_built_with_rocm():
        continue
      self._CompareBackpropFilter(input_size, filter_size, output_size, stride,
                                  padding, "float64")

  def testDepthwiseConv2DFilterGradExplicitCompare(self):
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTestExplicit()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropFilter(input_size, filter_size, output_size, stride,
                                  padding, "float32")
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      if test.is_built_with_rocm():
        continue
      self._CompareBackpropFilter(input_size, filter_size, output_size, stride,
                                  padding, "float64")

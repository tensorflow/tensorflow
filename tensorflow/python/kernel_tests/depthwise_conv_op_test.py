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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


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
                    data_type,
                    use_gpu,
                    grouped_conv=False,
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
      use_gpu: Whether to use GPU.
      grouped_conv: Whether to use cuDNN 7's grouped convolution.
      data_format: The data_format of the input. "NHWC" or "NCHW".
    """
    input_size = 1
    filter_size = 1
    for s in tensor_in_sizes:
      input_size *= s
    for s in filter_in_sizes:
      filter_size *= s
    # Initializes the input and filter tensor with numbers incrementing from 1.
    x1 = [f * 1.0 / input_size for f in range(1, input_size + 1)]
    x2 = [f * 1.0 / filter_size for f in range(1, filter_size + 1)]
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.test_session(graph=graph, use_gpu=use_gpu) as sess:
      tolerance = {
          dtypes.float16: 4e-2,
          dtypes.float32: 1e-8,
          dtypes.float64: 1e-13,
      }[data_type]

      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=data_type)
      t1.set_shape(tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=data_type)

      native_t1 = t1
      strides = [1, stride, stride, 1]
      if data_format == "NCHW":
        # Transpose from NHWC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        native_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
        strides = [1, 1, stride, stride]

      with sess.graph._kernel_label_map({
          "DepthwiseConv2dNative": "cudnn_grouped_convolution"
      } if grouped_conv else {}):
        conv_native = nn_ops.depthwise_conv2d_native(
            native_t1,
            t2,
            strides=strides,
            data_format=data_format,
            padding=padding)

      if data_format == "NCHW":
        # Transpose back from NCHW to NHWC
        conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])

      try:
        native_result = sess.run(conv_native)
      except errors.InvalidArgumentError as e:
        # Grouped convolution kernel is only registered for cuDNN 7. Silently
        # return when we are running on an earlier version or without GPU.
        if e.message.startswith(
            "No OpKernel was registered to support Op 'DepthwiseConv2dNative'"):
          tf_logging.warn("Skipping grouped convolution test")
          return
        raise e

      conv_interface = nn_impl.depthwise_conv2d(
          t1, t2, strides=[1, stride, stride, 1], padding=padding)
      interface_result = sess.run(conv_interface)

    tf_logging.info(
        "data_type: %r, use_gpu: %r, grouped_conv: %r, max diff = %f",
        data_type, use_gpu, grouped_conv,
        np.amax(np.absolute(native_result - interface_result)))
    self.assertArrayNear(
        np.ravel(native_result), np.ravel(interface_result), tolerance)
    self.assertShapeEqual(native_result, conv_native)
    self.assertShapeEqual(native_result, conv_interface)

  def testDepthwiseConv2D(self):
    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2D, %dth config: %r * %r, stride: %d, padding: "
          "%s", index, input_size, filter_size, stride, padding)
      for data_type in [dtypes.float16, dtypes.float32, dtypes.float64]:
        tf_logging.info("Testing without grouped_conv")
        self._VerifyValues(
            input_size, filter_size, stride, padding, data_type, use_gpu=True)
        tf_logging.info("Testing with grouped_conv")
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
            use_gpu=True,
            grouped_conv=True)

  def testDepthwiseConv2DFormat(self):
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, _, stride,
                padding) in enumerate(ConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFormat, %dth config: %r * %r, stride: %d, "
          "padding: %s", index, input_size, filter_size, stride, padding)
      for data_type in [dtypes.float16, dtypes.float32, dtypes.float64]:
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
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
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t1.set_shape(tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      conv = nn_ops.depthwise_conv2d_native(
          t1, t2, strides=[1, stride, stride, 1], padding=padding)
      value = sess.run(conv)
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
                                data_format="NHWC"):
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.test_session(graph=graph, use_gpu=use_gpu) as sess:
      tolerance = {
          dtypes.float16: 4e-0,
          dtypes.float32: 8e-4,
          dtypes.float64: 1e-12,
      }[data_type]

      input_tensor = constant_op.constant(
          input_data, shape=input_shape, dtype=data_type, name="input")
      filter_tensor = constant_op.constant(
          filter_data, shape=filter_shape, dtype=data_type, name="filter")

      native_input = input_tensor
      strides = [1, stride, stride, 1]
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

      with sess.graph._kernel_label_map({
          "DepthwiseConv2dNative": "cudnn_grouped_convolution",
          "DepthwiseConv2dNativeBackpropInput": "cudnn_grouped_convolution",
          "DepthwiseConv2dNativeBackpropFilter": "cudnn_grouped_convolution",
      } if grouped_conv else {}):
        depthwise_conv2d = nn_ops.depthwise_conv2d_native(
            native_input,
            filter_tensor,
            strides,
            padding,
            data_format=data_format,
            name="depthwise_conv2d")

      self.assertEqual(output_shape, depthwise_conv2d.get_shape())

      try:
        if test_input:
          err = gradient_checker.compute_gradient_error(
              native_input, input_shape, depthwise_conv2d, output_shape)
        else:
          err = gradient_checker.compute_gradient_error(
              filter_tensor, filter_shape, depthwise_conv2d, output_shape)
      except errors.InvalidArgumentError as e:
        # Grouped convolution kernel is only registered for cuDNN 7. Silently
        # return when we are running on an earlier version or without GPU.
        if grouped_conv and e.message.startswith(
            "No OpKernel was registered to support Op 'DepthwiseConv2dNative'"):
          tf_logging.warn("Skipping grouped convolution test")
          return
        raise e

      tf_logging.info(
          "data_type: %r, use_gpu: %r, grouped_conv: %r, error = %f", data_type,
          use_gpu, grouped_conv, err)
      self.assertLess(err, tolerance)

  def testDepthwiseConv2DInputGrad(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGrad, %dth config: %r * %r, stride: %d, "
          "padding: %s", index, input_size, filter_size, stride, padding)
      for data_type in [dtypes.float16, dtypes.float32, dtypes.float64]:
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True)
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            grouped_conv=True)

  def testDepthwiseConv2DInputGradFormat(self):
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradFormat, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      for data_type in [dtypes.float16, dtypes.float32, dtypes.float64]:
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            data_format="NCHW")

  def testDepthwiseConv2DFilterGrad(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGrad, %dth config: %r * %r, stride: "
          "%d, padding: %s", index, input_size, filter_size, stride, padding)
      for data_type in [dtypes.float16, dtypes.float32, dtypes.float64]:
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=False,
            use_gpu=True)

  def testDepthwiseConv2DFilterGradFormat(self):
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradFormat, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      for data_type in [dtypes.float16, dtypes.float32, dtypes.float64]:
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=False,
            use_gpu=True,
            data_format="NCHW")

  def _CompareBackpropInputFloat(self, input_sizes, filter_sizes, output_sizes,
                                 stride, padding):
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(use_gpu):
      with self.test_session(use_gpu=use_gpu):
        t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
        t1 = constant_op.constant(x1, shape=filter_sizes)
        t2 = constant_op.constant(x2, shape=output_sizes)
        backprop = nn_ops.depthwise_conv2d_native_backprop_input(
            t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
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
        t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
        t1 = constant_op.constant(x1, shape=filter_sizes)
        t2 = constant_op.constant(x2, shape=output_sizes)
        backprop = nn_ops.depthwise_conv2d_native_backprop_input(
            t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        ret = backprop.eval()
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_gpu=True)
    cpu_value = _GetVal(use_gpu=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def testDepthwiseConv2DInputGradCompare(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
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
        t0 = constant_op.constant(x0, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes)
        backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
            t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
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
        t0 = constant_op.constant(x0, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes)
        backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
            t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        ret = backprop.eval()
        self.assertShapeEqual(ret, backprop)
        return ret

    gpu_value = _GetVal(use_gpu=True)
    cpu_value = _GetVal(use_gpu=False)
    self.assertAllClose(cpu_value, gpu_value, rtol=1e-4, atol=1e-4)

  def testDepthwiseConv2DFilterGradCompare(self):
    for index, (input_size, filter_size, output_size, stride,
                padding) in enumerate(ConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropFilterFloat(input_size, filter_size, output_size,
                                       stride, padding)
      self._CompareBackpropFilterDouble(input_size, filter_size, output_size,
                                        stride, padding)


if __name__ == "__main__":
  test.main()

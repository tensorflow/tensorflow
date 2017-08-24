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
"""Functional tests for fused conv2d bias and activation operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.fused_conv.python.ops import fused_conv2d_bias_activation_op
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def GetShrunkInceptionShapes(shrink=10):
  """Iterator for smaller versions of convolution shapes in 2015 Inception.

  Relative to inception, each depth value is `depth // shrink`.

  Args:
    shrink: Factor to shrink each depth value by relative to Inception.

  Yields:
    Tuple (input_size, filter_size, out_size, stride, padding), the convolution
    parameters of Inception layers.
  """
  input_sizes = [[4, 5, 5, 1248], [4, 8, 8, 384], [4, 8, 8, 384], [
      4, 8, 8, 2048
  ], [4, 8, 8, 448], [4, 8, 8, 2048], [4, 8, 8, 2048], [4, 8, 8, 2048], [
      4, 8, 8, 1760
  ], [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 17, 17, 192], [
      4, 17, 17, 192
  ], [4, 17, 17, 1248], [4, 17, 17, 128], [4, 17, 17, 1248], [4, 17, 17, 224], [
      4, 17, 17, 192
  ], [4, 17, 17, 192], [4, 17, 17, 1216], [4, 17, 17, 1216], [4, 17, 17, 224], [
      4, 17, 17, 192
  ], [4, 17, 17, 192], [4, 17, 17, 1152], [4, 17, 17, 1152], [4, 17, 17, 192], [
      4, 17, 17, 160
  ], [4, 17, 17, 1152], [4, 17, 17, 1024], [4, 17, 17, 128], [4, 17, 17, 1024],
                 [4, 17, 17, 128], [4, 17, 17, 1024], [4, 17, 17, 128], [
                     4, 17, 17, 768
                 ], [4, 17, 17, 128], [4, 17, 17, 128], [4, 17, 17, 768],
                 [4, 17, 17, 768], [4, 35, 35, 96], [4, 35, 35, 288], [
                     4, 35, 35, 64
                 ], [4, 35, 35, 288], [4, 35, 35, 256], [4, 35, 35, 48], [
                     4, 35, 35, 256
                 ], [4, 35, 35, 96], [4, 35, 35, 192], [4, 35, 35, 192], [
                     4, 35, 35, 192
                 ], [4, 73, 73, 64], [4, 73, 73, 64], [4, 147, 147, 24]]
  filter_sizes = [[1, 1, 1248, 128], [1, 3, 384, 384], [3, 1, 384, 384], [
      1, 1, 2048, 192
  ], [3, 3, 448, 384], [1, 1, 2048, 320], [1, 1, 2048, 448], [1, 1, 2048, 384],
                  [1, 1, 1760, 384], [1, 1, 1760, 192], [1, 1, 1760, 448], [
                      1, 1, 1760, 320
                  ], [3, 3, 192, 192], [3, 3, 192, 192], [1, 1, 1248, 192], [
                      3, 3, 128, 320
                  ], [1, 1, 1248, 128], [1, 3, 224, 224], [3, 1, 192, 256], [
                      1, 3, 192, 256
                  ], [1, 1, 1216, 192], [1, 1, 1216, 96], [3, 1, 224, 224], [
                      3, 3, 192, 224
                  ], [1, 3, 192, 192], [1, 1, 1152, 192], [1, 1, 1152, 128], [
                      3, 1, 192, 192
                  ], [3, 3, 160, 192], [1, 1, 1152, 160], [1, 1, 1024, 128], [
                      1, 3, 128, 192
                  ], [1, 1, 1024, 160], [3, 1, 128, 192], [1, 1, 1024, 256], [
                      3, 1, 128, 128
                  ], [1, 1, 768, 192], [1, 3, 128, 128], [3, 3, 128, 128], [
                      1, 1, 768, 128
                  ], [1, 1, 768, 320], [3, 3, 96, 96], [3, 3, 288, 384], [
                      3, 3, 64, 96
                  ], [1, 1, 288, 64], [1, 1, 256, 64], [5, 5, 48, 64],
                  [1, 1, 256, 48], [3, 3, 96, 96], [1, 1, 192, 32], [
                      1, 1, 192, 64
                  ], [1, 1, 192, 48], [3, 3, 64, 192], [1, 1, 64,
                                                        64], [1, 1, 24, 64]]
  out_sizes = [[4, 5, 5, 128], [4, 8, 8, 384], [4, 8, 8, 384], [4, 8, 8, 192], [
      4, 8, 8, 384
  ], [4, 8, 8, 320], [4, 8, 8, 448], [4, 8, 8, 384], [4, 8, 8, 384], [
      4, 8, 8, 192
  ], [4, 8, 8, 448], [4, 8, 8, 320], [4, 8, 8, 192], [4, 17, 17, 192], [
      4, 17, 17, 192
  ], [4, 8, 8, 320], [4, 17, 17, 128], [4, 17, 17, 224], [4, 17, 17, 256], [
      4, 17, 17, 256
  ], [4, 17, 17, 192], [4, 17, 17, 96], [4, 17, 17, 224], [4, 17, 17, 224], [
      4, 17, 17, 192
  ], [4, 17, 17, 192], [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 192], [
      4, 17, 17, 160
  ], [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 160], [4, 17, 17, 192], [
      4, 17, 17, 256
  ], [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 128], [4, 17, 17, 128], [
      4, 17, 17, 128
  ], [4, 17, 17, 320], [4, 17, 17, 96], [4, 17, 17, 384], [4, 35, 35, 96], [
      4, 35, 35, 64
  ], [4, 35, 35, 64], [4, 35, 35, 64], [4, 35, 35, 48], [4, 35, 35, 96],
               [4, 35, 35, 32], [4, 35, 35, 64], [4, 35, 35, 48],
               [4, 71, 71, 192], [4, 73, 73, 64], [4, 147, 147, 64]]
  strides = [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1
  ]
  # Shrink sizes to make the test faster
  for i in input_sizes:
    i[3] //= shrink
  for f in filter_sizes:
    f[2] //= shrink
    f[3] //= shrink
  for o in out_sizes:
    o[3] //= shrink
  # pylint: disable=invalid-name
  VALID = "VALID"
  SAME = "SAME"
  # pylint: enable=invalid-name
  paddings = [
      SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      VALID, SAME, SAME, VALID, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, SAME, VALID, VALID, SAME, SAME, SAME, SAME, SAME,
      SAME, SAME, SAME, SAME, VALID, VALID, VALID
  ]
  for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                           paddings):
    yield i, f, o, s, p


def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NCHW", True), ("NHWC", True)]
  return test_configs


class FusedConv2DBiasActivationTest(test.TestCase):

  def _DtypesToTest(self, use_gpu):
    return [dtypes.float32]

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, bias,
                            strides, padding, activation_mode, data_format,
                            dtype):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      bias: 1-D bias tensor of length output_depth.
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      activation_mode: Activation mode.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
    Returns:
      Symbolic tensor value and reference value that can be used to
      execute the computation and verify the results.
    """
    input_size = np.prod(tensor_in_sizes)
    filter_size = np.prod(filter_in_sizes)
    bias_size = filter_in_sizes[-1]  # equals to output depth
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, input_size + 1)]
    x2 = [f * 1.0 for f in range(1, filter_size + 1)]
    # This is to guarantee that there is always negative values after
    # bias add so that we can test whether relu works correctly.
    x3 = bias
    with self.test_session(use_gpu=True):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      t3 = constant_op.constant(x3, shape=[bias_size], dtype=dtype)
      strides = [1] + strides + [1]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
      output = fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
          t1,
          t2,
          t3,
          strides=strides,
          padding=padding,
          data_format=data_format,
          activation_mode=activation_mode)
      ref_conv_output = nn_ops.conv2d(
          t1, t2, strides=strides, padding=padding, data_format=data_format)
      ref_bias_output = nn_ops.bias_add(
          ref_conv_output, t3, data_format=data_format)
      ref_output = nn_ops.relu(ref_bias_output)
      if data_format == "NCHW":
        output = test_util.NCHWToNHWC(output)
        ref_output = test_util.NCHWToNHWC(ref_output)

      return output, ref_output

  def _CompareFwdValues(self, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that CPU and GPU produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    x2 = np.random.rand(*filter_in_sizes).astype(np.float32)
    x3 = np.random.rand(*[filter_in_sizes[-1]]).astype(np.float32)

    def _SetupVal(data_format, use_gpu):
      with self.test_session(use_gpu=use_gpu):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        t3 = constant_op.constant(x3, shape=[filter_in_sizes[-1]])
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t1 = test_util.NHWCToNCHW(t1)
          strides = test_util.NHWCToNCHW(strides)
        output = fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
            t1,
            t2,
            t3,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation_mode="Relu")

        if data_format == "NCHW":
          output = test_util.NCHWToNHWC(output)
        return output

    tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      tensors.append(_SetupVal(data_format, use_gpu))
    with self.test_session() as sess:
      values = sess.run(tensors)
      for i in range(1, len(values)):
        self.assertAllClose(values[0], values[i], rtol=1e-5, atol=1e-5)

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, bias, strides,
                    padding):
    tensors = []
    ref_tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      for dtype in self._DtypesToTest(use_gpu):
        result, expected = self._SetupValuesForDevice(
            tensor_in_sizes, filter_in_sizes, bias, strides, padding, "Relu",
            data_format, dtype)
        tensors.append(result)
        ref_tensors.append(expected)
      with self.test_session() as sess:
        values = sess.run(tensors)
        ref_values = sess.run(ref_tensors)
        for i in range(len(tensors)):
          conv = tensors[i]
          value = values[i]
          ref_value = ref_values[i]
          print("expected = ", ref_value)
          print("actual = ", value)
          tol = 1e-5
          if value.dtype == np.float16:
            tol = 1e-3
          self.assertAllClose(
              np.ravel(ref_value), np.ravel(value), atol=tol, rtol=tol)
          self.assertShapeEqual(value, conv)

  def testConv2D1x1Filter(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2D1x1Filter test.")
      return
    # expected_output = [
    #    0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 57.0, 0.0, 0.0, 93.0, 41.0, 0.0, 129.0,
    #    86.0, 43.0, 165.0, 131.0, 97.0
    # ]
    medians = [-45.0, -130.0, -215.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        bias=medians,
        strides=[1, 1],
        padding="VALID")

  def testConv2DEmpty(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2DEmpty test.")
      return
    # expected_output = []
    self._VerifyValues(
        tensor_in_sizes=[0, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        bias=[0.0, 0.0, 0.0],
        strides=[1, 1],
        padding="VALID")

  def testConv2D2x2Filter(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2D2x2Filter test.")
      return
    # expected_output = [0.0, 0.0, 0.0, 401.0, 533.0, 665.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        bias=[-2500.0, -2500.0, -2500.0],
        strides=[1, 1],
        padding="VALID")

  def testConv2D1x2Filter(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2D1x2Filter test.")
      return
    # expected_output = [
    #    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 190.0, 265.0, 340.0, 343.0, 436.0, 529.0
    # ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        bias=[-500.0, -500.0, -500.0],
        strides=[1, 1],
        padding="VALID")

  def testConv2D2x2FilterStride2(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2D2x2FilterStride2 test.")
      return
    # expected_output = [0.0, 67.0, 163.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        bias=[-2300.0, -2300.0, -2300.0],
        strides=[2, 2],
        padding="VALID")

  def testConv2D2x2FilterStride2Same(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2D2x2FilterStride2Same test.")
      return
    # expected_output = [0.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        bias=[-2300.0, -1000.0, -1000.0],
        strides=[2, 2],
        padding="SAME")

  def testConv2D2x2FilterStride1x2(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2D2x2FilterStride1x2 test.")
      return
    # expected_output = [0.0, 0.0, 8.0, 28.0, 48.0, 68.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 6, 1],
        filter_in_sizes=[2, 2, 1, 1],
        bias=[-90.0],
        strides=[1, 2],
        padding="VALID")

  def testConv2DKernelSmallerThanStrideValid(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2DKernelSmallerThanStrideValid test.")
      return
    # expected_output = [0, 0, 175, 205]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 1],
        filter_in_sizes=[2, 2, 1, 1],
        bias=[-100.0],
        strides=[3, 3],
        padding="VALID")

  def testConv2DKernelSmallerThanStrideSame(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2DKernelSmallerThanStrideSame test.")
      return
    # expected = [0, 0, 2, 4]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1],
        bias=[-5.0],
        strides=[2, 2],
        padding="SAME")

    # expected = [0, 0, 4, 6]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[1, 1, 1, 1],
        bias=[-5.0],
        strides=[2, 2],
        padding="SAME")

    # expected = [4, 0, 1, 0]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[2, 2, 1, 1],
        bias=[-40.0],
        strides=[3, 3],
        padding="SAME")

  def testConv2DKernelSizeMatchesInputSize(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping Conv2DKernelSizeMatchesInputSize test.")
      return
    # expected = [0, 5]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 2, 1],
        filter_in_sizes=[2, 2, 1, 2],
        bias=[-50.0, -55.0],
        strides=[1, 1],
        padding="VALID")

    # expected = [0, 2, 282, 322]
    self._VerifyValues(
        tensor_in_sizes=[1, 8, 8, 1],
        filter_in_sizes=[2, 2, 1, 1],
        bias=[-200.0],
        strides=[4, 4],
        padding="SAME")

  def testShapeFunctionEdgeCases(self):
    # All shapes unknown.
    c1 = fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.float32),
        strides=[1, 1, 1, 1],
        padding="SAME",
        activation_mode="Relu")
    self.assertEqual([None, None, None, None], c1.get_shape().as_list())

    # Incorrect input shape.
    with self.assertRaises(ValueError):
      fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
          array_ops.placeholder(dtypes.float32, shape=[1, 3]),
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding="SAME",
          activation_mode="Relu")

    # Incorrect filter shape.
    with self.assertRaises(ValueError):
      fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.float32, shape=[1, 3]),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding="SAME",
          activation_mode="Relu")

    # Depth mismatch.
    with self.assertRaises(ValueError):
      fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
          array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
          array_ops.placeholder(dtypes.float32, shape=[4, 4, 2, 2]),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding="SAME",
          activation_mode="Relu")

  def testOpEdgeCases(self, gpu_only=True):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping OpEdgeCases tests.")
      return
    with self.test_session() as sess:
      # Illegal strides.
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "strides in the batch and depth"):
        sess.run(
            fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
                array_ops.placeholder(dtypes.float32),
                array_ops.placeholder(dtypes.float32),
                array_ops.placeholder(dtypes.float32),
                strides=[2, 1, 1, 1],
                padding="SAME",
                activation_mode="Relu"))
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "strides in the batch and depth"):
        sess.run(
            fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
                array_ops.placeholder(dtypes.float32),
                array_ops.placeholder(dtypes.float32),
                array_ops.placeholder(dtypes.float32),
                strides=[1, 1, 1, 2],
                padding="SAME",
                activation_mode="Relu"))

      # Illegal activation mode.
      with self.assertRaisesRegexp(ValueError,
                                   "Op passed string 'Tanh' not in:"):
        sess.run(
            fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
                array_ops.placeholder(dtypes.float32),
                array_ops.placeholder(dtypes.float32),
                array_ops.placeholder(dtypes.float32),
                strides=[1, 1, 1, 1],
                padding="SAME",
                activation_mode="Tanh"))

      # Filter larger than input.
      with self.assertRaisesRegexp(ValueError, "Negative dimension size"):
        sess.run(
            fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
                array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
                array_ops.placeholder(dtypes.float32, shape=[20, 21, 3, 2]),
                array_ops.placeholder(dtypes.float32, shape=[2]),
                strides=[1, 1, 1, 1],
                padding="VALID",
                activation_mode="Relu"))
      with self.assertRaisesRegexp(ValueError, "Negative dimension size"):
        sess.run(
            fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
                array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
                array_ops.placeholder(dtypes.float32, shape=[21, 20, 3, 2]),
                array_ops.placeholder(dtypes.float32, shape=[2]),
                strides=[1, 1, 1, 1],
                padding="VALID",
                activation_mode="Relu"))


def GetInceptionFwdTest(input_size, filter_size, stride, padding,
                        gpu_only=True):

  def Test(self):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping InceptionFwd %s", (input_size, filter_size,
                                                   stride, padding))
      return
    tf_logging.info("Testing InceptionFwd %s", (input_size, filter_size, stride,
                                                padding))
    self._CompareFwdValues(input_size, filter_size, [stride, stride], padding)

  return Test


if __name__ == "__main__":
  for index, (input_size_, filter_size_, output_size_, stride_,
              padding_) in enumerate(GetShrunkInceptionShapes()):
    setattr(FusedConv2DBiasActivationTest, "testInceptionFwd_" + str(index),
            GetInceptionFwdTest(input_size_, filter_size_, stride_, padding_))

  # TODO(b/35359731)
  # Fwd, BckInput, and BackFilter to test that for certain input parameter
  # set, winograd nonfused algorithm will be excluded from conv autotune. If
  # in such case, winograd nonfused algorithm is added as one option of the
  # conv autotune, and cuDNN version is smaller than 7, the following tests
  # will fail.
  ishape = [1, 400, 400, 1]
  fshape = [1, 1, 1, 256]
  oshape = [1, 400, 400, 256]
  setattr(FusedConv2DBiasActivationTest,
          "testInceptionFwd_No_Winograd_Nonfused",
          GetInceptionFwdTest(ishape, fshape, 1, "SAME", gpu_only=True))
  test.main()

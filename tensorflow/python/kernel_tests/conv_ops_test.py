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
"""Functional tests for convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib import layers
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
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
  input_sizes = [[4, 5, 5, 1248], [4, 8, 8, 384], [4, 8, 8, 384],
                 [4, 8, 8, 2048], [4, 8, 8, 448], [4, 8, 8, 2048],
                 [4, 8, 8, 2048], [4, 8, 8, 2048], [4, 8, 8, 1760],
                 [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 8, 8, 1760],
                 [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1248],
                 [4, 17, 17, 128], [4, 17, 17, 1248], [4, 17, 17, 224],
                 [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1216],
                 [4, 17, 17, 1216], [4, 17, 17, 224], [4, 17, 17, 192],
                 [4, 17, 17, 192], [4, 17, 17, 1152], [4, 17, 17, 1152],
                 [4, 17, 17, 192], [4, 17, 17, 160], [4, 17, 17, 1152],
                 [4, 17, 17, 1024], [4, 17, 17, 128], [4, 17, 17, 1024],
                 [4, 17, 17, 128], [4, 17, 17, 1024], [4, 17, 17, 128],
                 [4, 17, 17, 768], [4, 17, 17, 128], [4, 17, 17, 128],
                 [4, 17, 17, 768], [4, 17, 17, 768], [4, 35, 35, 96],
                 [4, 35, 35, 288], [4, 35, 35, 64], [4, 35, 35, 288],
                 [4, 35, 35, 256], [4, 35, 35, 48], [4, 35, 35, 256],
                 [4, 35, 35, 96], [4, 35, 35, 192], [4, 35, 35, 192],
                 [4, 35, 35, 192], [4, 73, 73, 64], [4, 73, 73, 64],
                 [4, 147, 147, 24]]
  filter_sizes = [[1, 1, 1248, 128], [1, 3, 384, 384], [3, 1, 384, 384],
                  [1, 1, 2048, 192], [3, 3, 448, 384], [1, 1, 2048, 320],
                  [1, 1, 2048, 448], [1, 1, 2048, 384], [1, 1, 1760, 384],
                  [1, 1, 1760, 192], [1, 1, 1760, 448], [1, 1, 1760, 320],
                  [3, 3, 192, 192], [3, 3, 192, 192], [1, 1, 1248, 192],
                  [3, 3, 128, 320], [1, 1, 1248, 128], [1, 3, 224, 224],
                  [3, 1, 192, 256], [1, 3, 192, 256], [1, 1, 1216, 192],
                  [1, 1, 1216, 96], [3, 1, 224, 224], [3, 3, 192, 224],
                  [1, 3, 192, 192], [1, 1, 1152, 192], [1, 1, 1152, 128],
                  [3, 1, 192, 192], [3, 3, 160, 192], [1, 1, 1152, 160],
                  [1, 1, 1024, 128], [1, 3, 128, 192], [1, 1, 1024, 160],
                  [3, 1, 128, 192], [1, 1, 1024, 256], [3, 1, 128, 128],
                  [1, 1, 768, 192], [1, 3, 128, 128], [3, 3, 128, 128],
                  [1, 1, 768, 128], [1, 1, 768, 320], [3, 3, 96, 96],
                  [3, 3, 288, 384], [3, 3, 64, 96], [1, 1, 288, 64],
                  [1, 1, 256, 64], [5, 5, 48, 64], [1, 1, 256, 48],
                  [3, 3, 96, 96], [1, 1, 192, 32], [1, 1, 192, 64],
                  [1, 1, 192, 48], [3, 3, 64, 192], [1, 1, 64, 64],
                  [1, 1, 24, 64]]
  out_sizes = [[4, 5, 5, 128], [4, 8, 8, 384], [4, 8, 8, 384],
               [4, 8, 8, 192], [4, 8, 8, 384], [4, 8, 8, 320],
               [4, 8, 8, 448], [4, 8, 8, 384], [4, 8, 8, 384],
               [4, 8, 8, 192], [4, 8, 8, 448], [4, 8, 8, 320],
               [4, 8, 8, 192], [4, 17, 17, 192], [4, 17, 17, 192],
               [4, 8, 8, 320], [4, 17, 17, 128], [4, 17, 17, 224],
               [4, 17, 17, 256], [4, 17, 17, 256], [4, 17, 17, 192],
               [4, 17, 17, 96], [4, 17, 17, 224], [4, 17, 17, 224],
               [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 128],
               [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 160],
               [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 160],
               [4, 17, 17, 192], [4, 17, 17, 256], [4, 17, 17, 128],
               [4, 17, 17, 192], [4, 17, 17, 128], [4, 17, 17, 128],
               [4, 17, 17, 128], [4, 17, 17, 320], [4, 17, 17, 96],
               [4, 17, 17, 384], [4, 35, 35, 96], [4, 35, 35, 64],
               [4, 35, 35, 64], [4, 35, 35, 64], [4, 35, 35, 48],
               [4, 35, 35, 96], [4, 35, 35, 32], [4, 35, 35, 64],
               [4, 35, 35, 48], [4, 71, 71, 192], [4, 73, 73, 64],
               [4, 147, 147, 64]]
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


def NHWCToNCHW(input_tensor):
  """Convert the input from NHWC format to NCHW.

  Args:
    input_tensor:  a 4-D tensor, or a 4-element array representing the same.
  Returns:
    the converted tensor or a shape array
  """
  if isinstance(input_tensor, ops.Tensor):
    return array_ops.transpose(input_tensor, [0, 3, 1, 2])
  else:
    return [input_tensor[0], input_tensor[3], input_tensor[1], input_tensor[2]]


def NCHWToNHWC(input_tensor):
  """Convert the input from NCHW format to NHWC.

  Args:
    input_tensor:  a 4-D tensor, or a 4-element array representing the same.
  Returns:
    the converted tensor or a shape array
  """
  if isinstance(input_tensor, ops.Tensor):
    return array_ops.transpose(input_tensor, [0, 2, 3, 1])
  else:
    return [input_tensor[0], input_tensor[2], input_tensor[3], input_tensor[1]]


def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NHWC", False), ("NHWC", True)]
  if test.is_gpu_available(cuda_only=True):
    # "NCHW" format is only supported on CUDA.
    test_configs += [("NCHW", True)]
  return test_configs


class Conv2DTest(test.TestCase):

  def _DtypesToTest(self, use_gpu):
    if use_gpu and not test_util.CudaSupportsHalfMatMulAndConv():
      return [dtypes.float32]
    else:
      # It is important that float32 comes before float16 here,
      # as we will be using its gradients as reference for fp16 gradients.
      return [dtypes.float32, dtypes.float16]

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, strides,
                            padding, data_format, dtype, use_gpu):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_gpu: True if the operations should be run on GPU
    Returns:
      Symbolic tensor value that can be used to execute the computation
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
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      if data_format == "NCHW":
        t1 = NHWCToNCHW(t1)
        strides = NHWCToNCHW(strides)
      conv = nn_ops.conv2d(
          t1, t2, strides=strides, padding=padding, data_format=data_format)
      if data_format == "NCHW":
        conv = NCHWToNHWC(conv)

      return conv

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

    def _SetupVal(data_format, use_gpu):
      with self.test_session(use_gpu=use_gpu):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t1 = NHWCToNCHW(t1)
          strides = NHWCToNCHW(strides)
        conv = nn_ops.conv2d(
            t1, t2, strides=strides, padding=padding, data_format=data_format)
        if data_format == "NCHW":
          conv = NCHWToNHWC(conv)
        return conv

    tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      tensors.append(_SetupVal(data_format, use_gpu))
    with self.test_session() as sess:
      values = sess.run(tensors)
      for i in range(1, len(values)):
        self.assertAllClose(values[0], values[i], rtol=1e-5, atol=1e-5)

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, strides, padding,
                    expected):
    tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      for dtype in self._DtypesToTest(use_gpu):
        result = self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            strides,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu)
        tensors.append(result)
      with self.test_session() as sess:
        values = sess.run(tensors)
        for i in range(len(tensors)):
          conv = tensors[i]
          value = values[i]
          print("expected = ", expected)
          print("actual = ", value)
          tol = 1e-5
          if value.dtype == np.float16:
            tol = 1e-3
          self.assertAllClose(expected, np.ravel(value), atol=tol, rtol=tol)
          self.assertShapeEqual(value, conv)

  def testConv2D1x1Filter(self):
    expected_output = [
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2DEmpty(self):
    expected_output = []
    self._VerifyValues(
        tensor_in_sizes=[0, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2Filter(self):
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2Filter(self):
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2(self):
    expected_output = [2271.0, 2367.0, 2463.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2Same(self):
    expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)

  def testConv2D2x2FilterStride1x2(self):
    expected_output = [58.0, 78.0, 98.0, 118.0, 138.0, 158.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 6, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[1, 2],
        padding="VALID",
        expected=expected_output)

  def testConv2DKernelSmallerThanStrideValid(self):
    expected_output = [65, 95, 275, 305]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  def testConv2DKernelSmallerThanStrideSame(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1],
        strides=[2, 2],
        padding="SAME",
        expected=[1, 3, 7, 9])

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[1, 1, 1, 1],
        strides=[2, 2],
        padding="SAME",
        expected=[1, 3, 9, 11])

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[3, 3],
        padding="SAME",
        expected=[44, 28, 41, 16])

  def testConv2DKernelSizeMatchesInputSize(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 2, 1],
        filter_in_sizes=[2, 2, 1, 2],
        strides=[1, 1],
        padding="VALID",
        expected=[50, 60])

    # TODO this currently fails.
    #self._VerifyValues(tensor_in_sizes=[1, 8, 8, 1],
    #                   filter_in_sizes=[2, 2, 1, 1],
    #                   strides=[4, 4], padding="SAME",
    #                   expected=[72, 112, 392, 432])

    # Testing for backprops
  def _RunAndVerifyBackpropInput(self, input_sizes, filter_sizes, output_sizes,
                                 strides, padding, expected, data_format,
                                 use_gpu, err):
    total_output_size = 1
    total_filter_size = 1
    for s in output_sizes:
      total_output_size *= s
    for s in filter_sizes:
      total_filter_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_filter_size + 1)]
    x2 = [f * 1.0 for f in range(1, total_output_size + 1)]
    with self.test_session(use_gpu=use_gpu) as sess:
      if data_format == "NCHW":
        input_sizes = NHWCToNCHW(input_sizes)
      t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
      t1 = constant_op.constant(x1, shape=filter_sizes)
      t2 = constant_op.constant(x2, shape=output_sizes)
      strides = [1] + strides + [1]
      if data_format == "NCHW":
        t2 = NHWCToNCHW(t2)
        strides = NHWCToNCHW(strides)
      conv = nn_ops.conv2d_backprop_input(
          t0, t1, t2, strides=strides, padding=padding, data_format=data_format)
      if data_format == "NCHW":
        conv = NCHWToNHWC(conv)
      # "values" consists of two tensors for two backprops
      value = sess.run(conv)
      self.assertShapeEqual(value, conv)
    print("expected = ", expected)
    print("actual = ", value)
    self.assertArrayNear(expected, value.flatten(), err)

  def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes,
                            conv_strides, padding):
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(data_format, use_gpu):
      with self.test_session(use_gpu=use_gpu) as sess:
        if data_format == "NCHW":
          new_input_sizes = NHWCToNCHW(input_sizes)
        else:
          new_input_sizes = input_sizes
        t0 = constant_op.constant(new_input_sizes, shape=[len(new_input_sizes)])
        t1 = constant_op.constant(x1, shape=filter_sizes)
        t2 = constant_op.constant(x2, shape=output_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t2 = NHWCToNCHW(t2)
          strides = NHWCToNCHW(strides)
        conv = nn_ops.conv2d_backprop_input(
            t0,
            t1,
            t2,
            strides=strides,
            padding=padding,
            data_format=data_format)
        if data_format == "NCHW":
          conv = NCHWToNHWC(conv)
        ret = conv.eval()
        self.assertShapeEqual(ret, conv)
        return ret

    values = []
    for (data_format, use_gpu) in GetTestConfigs():
      values.append(_GetVal(data_format, use_gpu))

    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-4, atol=1e-4)

  def testConv2D2x2Depth1ValidBackpropInput(self):
    expected_output = [1.0, 4.0, 4.0, 3.0, 10.0, 8.0]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[1, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[1, 1, 2, 1],
          strides=[1, 1],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5)

  def testConv2D2x2Depth3ValidBackpropInput(self):
    expected_output = [
        14.0, 32.0, 50.0, 100.0, 163.0, 226.0, 167.0, 212.0, 257.0, 122.0,
        140.0, 158.0, 478.0, 541.0, 604.0, 437.0, 482.0, 527.0
    ]
    for (data_format, use_gpu) in GetTestConfigs():
      # The GPU version of this test is not very stable. So adjusting the
      # error threshold to 1e-4.
      self._RunAndVerifyBackpropInput(
          input_sizes=[1, 2, 3, 3],
          filter_sizes=[2, 2, 3, 3],
          output_sizes=[1, 1, 2, 3],
          strides=[1, 1],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-4)

  def testConv2D2x2Depth3ValidBackpropInputStride1x2(self):
    expected_output = [
        1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 7.0, 12.0, 11.0, 18.0, 15.0, 24.0, 12.0,
        16.0, 15.0, 20.0, 18.0, 24.0
    ]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[1, 3, 6, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[1, 2, 3, 1],
          strides=[1, 2],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5)

  def testConv2DStrideTwoFilterOneSameBackpropInput(self):
    expected_output = [
        1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[1, 4, 4, 1],
          filter_sizes=[1, 1, 1, 1],
          output_sizes=[1, 2, 2, 1],
          strides=[2, 2],
          padding="SAME",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5)

  def testConv2DKernelSizeMatchesInputSizeBackpropInput(self):
    expected_output = [5.0, 11.0, 17.0, 23.0]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[1, 2, 2, 1],
          filter_sizes=[2, 2, 1, 2],
          output_sizes=[1, 1, 1, 2],
          strides=[1, 1],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5)

  # Testing for backprops
  def _RunAndVerifyBackpropFilter(self, input_sizes, filter_sizes, output_sizes,
                                  strides, padding, expected, data_format,
                                  use_gpu):
    total_input_size = 1
    total_output_size = 1
    for s in input_sizes:
      total_input_size *= s
    for s in output_sizes:
      total_output_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x0 = [f * 1.0 for f in range(1, total_input_size + 1)]
    x2 = [f * 1.0 for f in range(1, total_output_size + 1)]
    for dtype in self._DtypesToTest(use_gpu=use_gpu):
      with self.test_session(use_gpu=use_gpu) as sess:
        t0 = constant_op.constant(x0, shape=input_sizes, dtype=dtype)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
        explicit_strides = [1] + strides + [1]
        if data_format == "NCHW":
          t0 = NHWCToNCHW(t0)
          t2 = NHWCToNCHW(t2)
          explicit_strides = NHWCToNCHW(explicit_strides)
        conv = nn_ops.conv2d_backprop_filter(
            t0,
            t1,
            t2,
            strides=explicit_strides,
            padding=padding,
            data_format=data_format)
        value = sess.run(conv)
        self.assertShapeEqual(value, conv)
      print("expected = ", expected)
      print("actual = ", value)
      self.assertArrayNear(expected, value.flatten(), 1e-5)

  def _CompareBackFilter(self, input_sizes, filter_sizes, output_sizes,
                         conv_strides, padding):
    x0 = np.random.rand(*input_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _GetVal(data_format, use_gpu):
      with self.test_session(use_gpu=use_gpu) as sess:
        t0 = constant_op.constant(x0, shape=input_sizes)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t0 = NHWCToNCHW(t0)
          t2 = NHWCToNCHW(t2)
          strides = NHWCToNCHW(strides)
        conv = nn_ops.conv2d_backprop_filter(
            t0,
            t1,
            t2,
            strides=strides,
            padding=padding,
            data_format=data_format)
        ret = conv.eval()
        self.assertShapeEqual(ret, conv)
        return ret

    values = []
    for (data_format, use_gpu) in GetTestConfigs():
      values.append(_GetVal(data_format, use_gpu))
    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-4, atol=1e-4)

  def testConv2D2x2Depth1ValidBackpropFilter(self):
    expected = [5.0, 8.0, 14.0, 17.0]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[1, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[1, 1, 2, 1],
          strides=[1, 1],
          padding="VALID",
          expected=expected,
          data_format=data_format,
          use_gpu=use_gpu)

  def testConv2D2x2Depth3ValidBackpropFilter(self):
    expected = [
        17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0, 32.0, 43.0, 54.0,
        37.0, 50.0, 63.0, 42.0, 57.0, 72.0, 62.0, 85.0, 108.0, 67.0, 92.0,
        117.0, 72.0, 99.0, 126.0, 77.0, 106.0, 135.0, 82.0, 113.0, 144.0, 87.0,
        120.0, 153.0
    ]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[1, 2, 3, 3],
          filter_sizes=[2, 2, 3, 3],
          output_sizes=[1, 1, 2, 3],
          strides=[1, 1],
          padding="VALID",
          expected=expected,
          data_format=data_format,
          use_gpu=use_gpu)

  def testConv2D2x2Depth3ValidBackpropFilterStride1x2(self):
    expected = [161.0, 182.0, 287.0, 308.0]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[1, 3, 6, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[1, 2, 3, 1],
          strides=[1, 2],
          padding="VALID",
          expected=expected,
          data_format=data_format,
          use_gpu=use_gpu)

  def testConv2DStrideTwoFilterOneSameBackpropFilter(self):
    expected_output = [78.]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[1, 4, 4, 1],
          filter_sizes=[1, 1, 1, 1],
          output_sizes=[1, 2, 2, 1],
          strides=[2, 2],
          padding="SAME",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu)

  def testConv2DKernelSizeMatchesInputSizeBackpropFilter(self):
    expected_output = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0]
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropFilter(
          input_sizes=[1, 2, 2, 1],
          filter_sizes=[2, 2, 1, 2],
          output_sizes=[1, 1, 1, 2],
          strides=[1, 1],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu)

  # Gradient checkers
  def ConstructAndTestGradient(self, batch, input_rows, input_cols, filter_rows,
                               filter_cols, in_depth, out_depth, stride_rows,
                               stride_cols, padding, test_input, data_format,
                               use_gpu):
    input_shape = [batch, input_rows, input_cols, in_depth]
    filter_shape = [filter_rows, filter_cols, in_depth, out_depth]
    # TODO(yangke): re-factor the computation of output shape.
    if padding == "VALID":
      output_rows = (input_rows - filter_rows + stride_rows) // stride_rows
      output_cols = (input_cols - filter_cols + stride_cols) // stride_cols
    else:
      output_rows = (input_rows + stride_rows - 1) // stride_rows
      output_cols = (input_cols + stride_cols - 1) // stride_cols
    output_shape = [batch, output_rows, output_cols, out_depth]
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    # Conv2DGrad functions are not compiled for double due to
    # a problem in the way Eigen's Conv2DGrad works for double.
    # So we disable the DOUBLE path.  We should re-enable this
    # when double support returns for CPU and/or GPU.
    for dtype in self._DtypesToTest(use_gpu=use_gpu):
      with self.test_session(use_gpu=use_gpu):
        input_tensor = constant_op.constant(
            input_data, shape=input_shape, dtype=dtype, name="input")
        filter_tensor = constant_op.constant(
            filter_data, shape=filter_shape, dtype=dtype, name="filter")
        strides = [1, stride_rows, stride_cols, 1]
        if data_format == "NCHW":
          new_input_tensor = NHWCToNCHW(input_tensor)
          strides = NHWCToNCHW(strides)
        else:
          new_input_tensor = input_tensor
        conv = nn_ops.conv2d(
            new_input_tensor,
            filter_tensor,
            strides,
            padding,
            data_format=data_format,
            name="conv")
        if data_format == "NCHW":
          conv = NCHWToNHWC(conv)
        self.assertEqual(output_shape, conv.get_shape())
        if test_input:
          jacob_t, jacob_n = gradient_checker.compute_gradient(input_tensor,
                                                               input_shape,
                                                               conv,
                                                               output_shape)
        else:
          jacob_t, jacob_n = gradient_checker.compute_gradient(filter_tensor,
                                                               filter_shape,
                                                               conv,
                                                               output_shape)
        if dtype == dtypes.float32:
          reference_jacob_t = jacob_t
          err = np.fabs(jacob_t - jacob_n).max()
        else:
          # Compare fp16 theoretical gradients to fp32 theoretical gradients,
          # since fp16 numerical gradients are too imprecise.
          err = np.fabs(jacob_t - reference_jacob_t).max()

        print("conv_2d gradient error = ", err)
        self.assertLess(err, 0.002)

  def testInputGradientValidPaddingStrideOne(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=5,
          input_cols=4,
          filter_rows=3,
          filter_cols=3,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="VALID",
          test_input=True,
          data_format=data_format,
          use_gpu=use_gpu)

  def testFilterGradientValidPaddingStrideOne(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=4,
          input_rows=6,
          input_cols=5,
          filter_rows=2,
          filter_cols=2,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="VALID",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  def testInputGradientValidPaddingStrideTwo(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=4,
          input_cols=5,
          filter_rows=3,
          filter_cols=3,
          in_depth=2,
          out_depth=3,
          stride_rows=2,
          stride_cols=2,
          padding="VALID",
          test_input=True,
          data_format=data_format,
          use_gpu=use_gpu)

  def testFilterGradientValidPaddingStrideTwo(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=4,
          input_rows=6,
          input_cols=5,
          filter_rows=2,
          filter_cols=2,
          in_depth=2,
          out_depth=3,
          stride_rows=2,
          stride_cols=2,
          padding="VALID",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  def testInputGradientValidPaddingStrideThree(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=7,
          input_cols=6,
          filter_rows=3,
          filter_cols=3,
          in_depth=4,
          out_depth=5,
          stride_rows=3,
          stride_cols=3,
          padding="VALID",
          test_input=True,
          data_format=data_format,
          use_gpu=use_gpu)

  def testFilterGradientValidPaddingStrideThree(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=8,
          input_cols=7,
          filter_rows=4,
          filter_cols=4,
          in_depth=2,
          out_depth=3,
          stride_rows=3,
          stride_cols=3,
          padding="VALID",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  def testInputGradientSamePaddingStrideOne(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=7,
          input_cols=6,
          filter_rows=3,
          filter_cols=3,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="SAME",
          test_input=True,
          data_format=data_format,
          use_gpu=use_gpu)

  def testFilterGradientSamePaddingStrideOne(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=4,
          input_rows=6,
          input_cols=5,
          filter_rows=2,
          filter_cols=2,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="SAME",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  def testInputGradientSamePaddingStrideTwo(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=5,
          input_cols=4,
          filter_rows=3,
          filter_cols=3,
          in_depth=3,
          out_depth=3,
          stride_rows=2,
          stride_cols=2,
          padding="SAME",
          test_input=True,
          data_format=data_format,
          use_gpu=use_gpu)

  def testFilterGradientSamePaddingStrideTwo(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=4,
          input_rows=6,
          input_cols=5,
          filter_rows=2,
          filter_cols=2,
          in_depth=2,
          out_depth=3,
          stride_rows=2,
          stride_cols=2,
          padding="SAME",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  def testInputGradientSamePaddingStrideThree(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=7,
          input_cols=6,
          filter_rows=3,
          filter_cols=3,
          in_depth=4,
          out_depth=5,
          stride_rows=3,
          stride_cols=3,
          padding="SAME",
          test_input=True,
          data_format=data_format,
          use_gpu=use_gpu)

  def testFilterGradientSamePaddingStrideThree(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=8,
          input_cols=7,
          filter_rows=4,
          filter_cols=4,
          in_depth=2,
          out_depth=3,
          stride_rows=3,
          stride_cols=3,
          padding="SAME",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  def testFilterGradientSamePaddingStride2x1(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=8,
          input_cols=7,
          filter_rows=4,
          filter_cols=4,
          in_depth=2,
          out_depth=3,
          stride_rows=2,
          stride_cols=1,
          padding="SAME",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  def testInputGradientKernelSizeMatchesInputSize(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=4,
          input_cols=3,
          filter_rows=4,
          filter_cols=3,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="VALID",
          test_input=True,
          data_format=data_format,
          use_gpu=use_gpu)

  def testFilterGradientKernelSizeMatchesInputSize(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=4,
          input_cols=3,
          filter_rows=4,
          filter_cols=3,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="VALID",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  def testShapeFunctionEdgeCases(self):
    # All shapes unknown.
    c1 = nn_ops.conv2d(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.float32),
        strides=[1, 1, 1, 1],
        padding="SAME")
    self.assertEqual([None, None, None, None], c1.get_shape().as_list())

    # Incorrect input shape.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(
              dtypes.float32, shape=[1, 3]),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding="SAME")

    # Incorrect filter shape.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(
              dtypes.float32, shape=[1, 3]),
          strides=[1, 1, 1, 1],
          padding="SAME")

    # Depth mismatch.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(
              dtypes.float32, shape=[32, 20, 20, 3]),
          array_ops.placeholder(
              dtypes.float32, shape=[4, 4, 2, 2]),
          strides=[1, 1, 1, 1],
          padding="SAME")

  def testOpEdgeCases(self):
    with self.test_session() as sess:
      # Illegal strides.
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "strides in the batch and depth"):
        sess.run(
            nn_ops.conv2d(
                array_ops.placeholder(dtypes.float32),
                array_ops.placeholder(dtypes.float32),
                strides=[2, 1, 1, 1],
                padding="SAME"))
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "strides in the batch and depth"):
        sess.run(
            nn_ops.conv2d(
                array_ops.placeholder(dtypes.float32),
                array_ops.placeholder(dtypes.float32),
                strides=[1, 1, 1, 2],
                padding="SAME"))

      # Filter larger than input.
      with self.assertRaisesRegexp(ValueError, "Negative dimension size"):
        sess.run(
            nn_ops.conv2d(
                array_ops.placeholder(
                    dtypes.float32, shape=[32, 20, 20, 3]),
                array_ops.placeholder(
                    dtypes.float32, shape=[20, 21, 3, 2]),
                strides=[1, 1, 1, 1],
                padding="VALID"))
      with self.assertRaisesRegexp(ValueError, "Negative dimension size"):
        sess.run(
            nn_ops.conv2d(
                array_ops.placeholder(
                    dtypes.float32, shape=[32, 20, 20, 3]),
                array_ops.placeholder(
                    dtypes.float32, shape=[21, 20, 3, 2]),
                strides=[1, 1, 1, 1],
                padding="VALID"))


# This is only a very simple test. More comprehensive tests live in
# //learning/dist_belief/experimental/brain_compatibility/conv_nn_test.py
# where we compare the numeric results of the depthwise conv op with the
# depthwise weighted sum transformer in dist_belief.
class DepthwiseConv2DTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    expected):
    """Verifies the output values of the convolution function.

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
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.test_session() as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t1.set_shape(tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      conv = nn_impl.depthwise_conv2d(
          t1, t2, strides=[1, stride, stride, 1], padding=padding)
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
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        stride=1,
        padding="VALID",
        expected=expected_output)


class SeparableConv2DTest(test.TestCase):

  def _InitValues(self, sizes):
    """Initializes values for input tensors.

    Args:
      sizes: Tensor dimensions.

    Returns:
      Tensor initialized to values.
    """
    total_size = 1
    for s in sizes:
      total_size *= s
    x = [f * 0.5 for f in range(1, total_size + 1)]
    return constant_op.constant(x, shape=sizes)

  def _VerifyValues(self, tensor_in_sizes, depthwise_filter_in_sizes,
                    pointwise_filter_in_sizes, stride, padding, expected):
    """Verifies the output values of the separable convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions.
      depthwise_filter_in_sizes: Depthwise filter tensor dimensions.
      pointwise_filter_in_sizes: Pointwise filter tensor dimensions.
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
    """
    with self.test_session() as sess:
      t1 = self._InitValues(tensor_in_sizes)
      f1 = self._InitValues(depthwise_filter_in_sizes)
      f1.set_shape(depthwise_filter_in_sizes)
      f2 = self._InitValues(pointwise_filter_in_sizes)
      conv = nn_impl.separable_conv2d(
          t1, f1, f2, strides=[1, stride, stride, 1], padding=padding)
      value = sess.run(conv)
    print("value = ", value)
    self.assertArrayNear(expected, np.ravel(value), 1e-5)
    self.assertShapeEqual(value, conv)

  def testSeparableConv2D(self):
    # The output is the result of two convolutions:
    # First with tensor_in[1, 4, 4, 2] * filter1[2, 2, 2, 3].
    # Second with intermediate_out[1, 4, 4, 6] * filter2[1, 1, 6, 7].
    # Complexity is O(2*3*2*2 + 6*7*1*1) as opposed to O(2*7*2*2).
    expected_output = [
        6644.5, 6971.5, 7298.5, 7625.5, 7952.5, 8279.5, 8606.5, 8154.5, 8556.5,
        8958.5, 9360.5, 9762.5, 10164.5, 10566.5, 9664.5, 10141.5, 10618.5,
        11095.5, 11572.5, 12049.5, 12526.5, 4145.5, 4346.5, 4547.5, 4748.5,
        4949.5, 5150.5, 5351.5, 12684.5, 13311.5, 13938.5, 14565.5, 15192.5,
        15819.5, 16446.5, 14194.5, 14896.5, 15598.5, 16300.5, 17002.5, 17704.5,
        18406.5, 15704.5, 16481.5, 17258.5, 18035.5, 18812.5, 19589.5, 20366.5,
        6499.5, 6814.5, 7129.5, 7444.5, 7759.5, 8074.5, 8389.5, 18724.5,
        19651.5, 20578.5, 21505.5, 22432.5, 23359.5, 24286.5, 20234.5, 21236.5,
        22238.5, 23240.5, 24242.5, 25244.5, 26246.5, 21744.5, 22821.5, 23898.5,
        24975.5, 26052.5, 27129.5, 28206.5, 8853.5, 9282.5, 9711.5, 10140.5,
        10569.5, 10998.5, 11427.5, 5746.75, 6010.75, 6274.75, 6538.75, 6802.75,
        7066.75, 7330.75, 6168.75, 6452.25, 6735.75, 7019.25, 7302.75, 7586.25,
        7869.75, 6590.75, 6893.75, 7196.75, 7499.75, 7802.75, 8105.75, 8408.75,
        2036.25, 2119.5, 2202.75, 2286.0, 2369.25, 2452.5, 2535.75
    ]

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 2],
        depthwise_filter_in_sizes=[2, 2, 2, 3],
        pointwise_filter_in_sizes=[1, 1, 6, 7],
        stride=1,
        padding="SAME",
        expected=expected_output)

  def testSeparableConv2DEqualInputOutputDepth(self):
    # The output is the result of two convolutions:
    # First with tensor_in[1, 4, 4, 2] * filter1[2, 2, 3, 3].
    # Second with intermediate_out[1, 4, 4, 6] * filter2[1, 1, 6, 6].
    # Complexity is O(2*3*2*2 + 6*6*1*1) as opposed to O(2*6*2*2).
    expected_output = [
        5742.0, 6069.0, 6396.0, 6723.0, 7050.0, 7377.0, 7047.0, 7449.0, 7851.0,
        8253.0, 8655.0, 9057.0, 8352.0, 8829.0, 9306.0, 9783.0, 10260.0,
        10737.0, 3582.0, 3783.0, 3984.0, 4185.0, 4386.0, 4587.0, 10962.0,
        11589.0, 12216.0, 12843.0, 13470.0, 14097.0, 12267.0, 12969.0, 13671.0,
        14373.0, 15075.0, 15777.0, 13572.0, 14349.0, 15126.0, 15903.0, 16680.0,
        17457.0, 5616.0, 5931.0, 6246.0, 6561.0, 6876.0, 7191.0, 16182.0,
        17109.0, 18036.0, 18963.0, 19890.0, 20817.0, 17487.0, 18489.0, 19491.0,
        20493.0, 21495.0, 22497.0, 18792.0, 19869.0, 20946.0, 22023.0, 23100.0,
        24177.0, 7650.0, 8079.0, 8508.0, 8937.0, 9366.0, 9795.0, 4963.5, 5227.5,
        5491.5, 5755.5, 6019.5, 6283.5, 5328.0, 5611.5, 5895.0, 6178.5, 6462.0,
        6745.5, 5692.5, 5995.5, 6298.5, 6601.5, 6904.5, 7207.5, 1757.25, 1840.5,
        1923.75, 2007.0, 2090.25, 2173.5
    ]

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 2],
        depthwise_filter_in_sizes=[2, 2, 2, 3],
        pointwise_filter_in_sizes=[1, 1, 6, 6],
        stride=1,
        padding="SAME",
        expected=expected_output)

  def testSeparableConv2DIllegalCases(self):
    # Output depth less then input depth.
    with self.assertRaisesRegexp(
        ValueError,
        "Refusing to perform an overparameterized separable convolution"):
      self._VerifyValues(
          tensor_in_sizes=[1, 4, 4, 2],
          depthwise_filter_in_sizes=[2, 2, 2, 3],
          pointwise_filter_in_sizes=[1, 1, 6, 5],
          stride=1,
          padding="SAME",
          expected=None)


class DeepConv2DTest(test.TestCase):

  def _CompareFwdConv2D(self, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that DeepConv2D and Conv2D produce the same values.

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

    with self.test_session(use_gpu=False) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      strides = [1] + conv_strides + [1]

      conv = nn_ops.conv2d(t1, t2, strides=strides, padding=padding)

      os.environ["TF_USE_DEEP_CONV2D"] = "0"
      values_expect = sess.run([conv])

      os.environ["TF_USE_DEEP_CONV2D"] = "1"
      values_test = sess.run([conv])

      self.assertAllClose(values_expect, values_test, rtol=1e-5, atol=1e-5)

  def _RunTestCases(self, conv_strides, padding):
    input_sizes = [[5, 5, 5, 1248], [3, 17, 17, 192], [2, 35, 35, 288],
                   [2, 6, 8, 517], [2, 7, 4, 81], [3, 11, 3, 77]]
    filter_sizes = [[3, 3, 1248, 128], [3, 3, 192, 192], [3, 3, 288, 384],
                    [3, 3, 517, 64], [3, 3, 81, 77], [3, 3, 77, 181]]
    for input_shape, filter_shape in zip(input_sizes, filter_sizes):
      self._CompareFwdConv2D(input_shape, filter_shape, conv_strides, padding)

  def testConv2D3x3FilterStride1x1Valid(self):
    self._RunTestCases([1, 1], "VALID")

  def testConv2D3x3FilterStride1x1Same(self):
    self._RunTestCases([1, 1], "SAME")


class Conv2DBenchmark(test.Benchmark):

  def benchmarkGPUConvStackFirst(self):
    # Benchmark the first iteration of a conv-net with many identical conv
    # operations.
    if not test.is_gpu_available():
      return

    with ops.Graph().as_default(), session_lib.Session() as session:
      batch_size = 1
      timesteps = 600
      features = 1

      inputs = random_ops.random_uniform(
          [batch_size, 1, timesteps, features], seed=1234)
      num_outputs_list = [512] * 40 + [1]
      kernel_w = 3
      x = inputs
      for num_outputs in num_outputs_list:
        x = layers.convolution2d(x, num_outputs, [1, kernel_w])
      outputs = x

      variables.global_variables_initializer().run()
      num_iterations = 4
      for iter_index in xrange(num_iterations):
        start = time.time()
        session.run(outputs)
        wall_time = time.time() - start
        self.report_benchmark(
            name="conv_stack_iter_%d" % iter_index, wall_time=wall_time)
        print("conv_stack_iter_%d: %.4f" % (iter_index, wall_time))


def GetInceptionFwdTest(input_size, filter_size, stride, padding):

  def Test(self):
    tf_logging.info("Testing InceptionFwd %s", (input_size, filter_size, stride,
                                                padding))
    self._CompareFwdValues(input_size, filter_size, [stride, stride], padding)

  return Test


def GetInceptionBackInputTest(input_size, filter_size, output_size, stride,
                              padding):

  def Test(self):
    tf_logging.info("Testing InceptionBackInput %s",
                    (input_size, filter_size, output_size, stride, padding))
    self._CompareBackpropInput(input_size, filter_size, output_size,
                               [stride, stride], padding)

  return Test


def GetInceptionBackFilterTest(input_size, filter_size, output_size, strides,
                               padding):

  def Test(self):
    tf_logging.info("Testing InceptionBackFilter %s",
                    (input_size, filter_size, output_size, strides, padding))
    self._CompareBackFilter(input_size, filter_size, output_size, strides,
                            padding)

  return Test


if __name__ == "__main__":
  for index, (input_size_, filter_size_, output_size_, stride_,
              padding_) in enumerate(GetShrunkInceptionShapes()):
    setattr(Conv2DTest, "testInceptionFwd_" + str(index),
            GetInceptionFwdTest(input_size_, filter_size_, stride_, padding_))
    setattr(Conv2DTest, "testInceptionBackInput_" + str(index),
            GetInceptionBackInputTest(input_size_, filter_size_, output_size_,
                                      stride_, padding_))
    setattr(Conv2DTest, "testInceptionBackFilter_" + str(index),
            GetInceptionBackFilterTest(input_size_, filter_size_, output_size_,
                                       [stride_, stride_], padding_))

  test.main()

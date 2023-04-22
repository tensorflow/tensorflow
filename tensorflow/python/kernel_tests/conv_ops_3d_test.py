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
"""Functional tests for 3d convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.eager import context


def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NDHWC", False), ("NDHWC", True)]
  if test.is_gpu_available(cuda_only=True):
    # "NCDHW" format is only supported on CUDA.
    test_configs += [("NCDHW", True)]
  return test_configs


@test_util.run_all_without_tensor_float_32(
    "Tests Conv3d, which in some cases is implemented with a matmul. With "
    "TensorFloat-32, tests fail in some of those cases (and as of August 13 "
    "2020, only those cases)")
class Conv3DTest(test.TestCase):

  def _DtypesToTest(self, use_gpu):
    # double datatype is currently not supported for convolution ops
    # on the ROCm platform
    optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
    if use_gpu:
      if not test_util.GpuSupportsHalfMatMulAndConv():
        return optional_float64 + [dtypes.float32]
      else:
        # It is important that float32 comes before float16 here,
        # as we will be using its gradients as reference for fp16 gradients.
        return optional_float64 + [dtypes.float32, dtypes.float16]
    else:
      return optional_float64 + [dtypes.float32, dtypes.float16]

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, stride,
                            padding, data_format, dtype, use_gpu):
    total_size_tensor = np.prod(tensor_in_sizes)
    total_size_filter = np.prod(filter_in_sizes)

    # Initializes the input tensor with array containing numbers from 0 to 1.
    # We keep the input tensor values fairly small to avoid overflowing float16
    # during the conv3d.
    x1 = [f * 1.0 / total_size_tensor for f in range(1, total_size_tensor + 1)]
    x2 = [f * 1.0 / total_size_filter for f in range(1, total_size_filter + 1)]
    with self.cached_session(use_gpu=use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)

      if isinstance(stride, collections_abc.Iterable):
        strides = [1] + list(stride) + [1]
      else:
        strides = [1, stride, stride, stride, 1]

      if data_format == "NCDHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
      conv = nn_ops.conv3d(t1, t2, strides, padding=padding,
                           data_format=data_format)
      if data_format == "NCDHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    expected):
    results = []
    for data_format, use_gpu in GetTestConfigs():
      for dtype in self._DtypesToTest(use_gpu):
        result = self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            stride,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu)
        results.append(result)

      with self.cached_session() as sess:
        values = self.evaluate(results)
        for value in values:
          print("expected = ", expected)
          print("actual = ", value)
          tol = 1e-6
          if value.dtype == np.float16:
            tol = 1e-3

          self.assertAllClose(expected, value.flatten(), atol=tol, rtol=tol)

  def _ComputeReferenceDilatedConv(self, tensor_in_sizes, filter_in_sizes,
                                   stride, dilation, padding, data_format,
                                   use_gpu):
    total_size_tensor = np.prod(tensor_in_sizes)
    total_size_filter = np.prod(filter_in_sizes)

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_tensor + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_filter + 1)]
    with self.cached_session(use_gpu=use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      if isinstance(stride, collections_abc.Iterable):
        strides = list(stride)
      else:
        strides = [stride, stride, stride]
      if data_format == "NCDHW":
        t1 = test_util.NHWCToNCHW(t1)
        full_strides = [1, 1] + strides
        full_dilation = [1, 1] + dilation
      else:
        full_strides = [1] + strides + [1]
        full_dilation = [1] + dilation + [1]
      expected = nn_ops.convolution(
          t1,
          t2,
          padding=padding,
          strides=strides,
          dilation_rate=dilation,
          data_format=data_format)
      computed = nn_ops.conv3d(
          t1,
          t2,
          strides=full_strides,
          dilations=full_dilation,
          padding=padding,
          data_format=data_format)
      if data_format == "NCDHW":
        expected = test_util.NCHWToNHWC(expected)
        computed = test_util.NCHWToNHWC(computed)
    return expected, computed

  def _VerifyDilatedConvValues(self, tensor_in_sizes, filter_in_sizes, stride,
                               padding, dilations):
    expected_results = []
    computed_results = []
    default_dilations = (
        dilations[0] == 1 and dilations[1] == 1 and dilations[2] == 1)
    for data_format, use_gpu in GetTestConfigs():
      # If any dilation rate is larger than 1, only do test on the GPU
      # because we currently do not have a CPU implementation for arbitrary
      # dilation rates.
      if default_dilations or use_gpu:
        expected, computed = self._ComputeReferenceDilatedConv(
            tensor_in_sizes, filter_in_sizes, stride, dilations, padding,
            data_format, use_gpu)
        expected_results.append(expected)
        computed_results.append(computed)
        tolerance = 1e-2 if use_gpu else 1e-5
        with self.cached_session() as sess:
          expected_values = self.evaluate(expected_results)
          computed_values = self.evaluate(computed_results)
          for e_value, c_value in zip(expected_values, computed_values):
            print("expected = ", e_value)
            print("actual = ", c_value)
            self.assertAllClose(
                e_value.flatten(), c_value.flatten(), atol=tolerance, rtol=1e-6)

  def _CreateNumpyTensor(self, sizes):
    return np.asarray([f * 1.0 for f in range(1,
                                              np.prod(sizes) + 1)],
                      dtype=np.float32).reshape(sizes)

  @test_util.run_in_graph_and_eager_modes
  def testConv3DExpandedBatch(self):
    tensor_in_sizes_batch = [10, 2, 3, 1, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 1, 3]
    filter_in_sizes = [1, 1, 1, 3, 3]
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)
    conv1 = nn_ops.conv3d_v2(
        x1, filter_in, strides=[1, 1, 1, 1, 1], padding="VALID")
    conv2 = nn_ops.conv3d_v2(
        x2, filter_in, strides=[1, 1, 1, 1, 1], padding="VALID")
    self.assertEqual(conv1.shape, tensor_in_sizes_batch)
    self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
    self.assertAllClose(conv1, self.evaluate(conv2).reshape(conv1.shape))

  @test_util.run_in_graph_and_eager_modes
  def testConvolutionClass3DExpandedBatch(self):
    tensor_in_sizes_batch = [10, 2, 3, 1, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 1, 3]
    filter_in_sizes = [1, 1, 1, 3, 3]
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)
    convolver1 = nn_ops.Convolution(
        input_shape=x1.shape,
        filter_shape=filter_in.shape,
        strides=[1, 1, 1],
        padding="VALID")
    self.assertEqual(convolver1.num_batch_dims, 1)
    convolver2 = nn_ops.Convolution(
        input_shape=x2.shape,
        filter_shape=filter_in.shape,
        strides=[1, 1, 1],
        padding="VALID")
    self.assertEqual(convolver2.num_batch_dims, 2)
    conv1 = convolver1(x1, filter_in)
    conv2 = convolver2(x2, filter_in)
    self.assertEqual(conv1.shape, tensor_in_sizes_batch)
    self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
    self.assertAllClose(conv1, self.evaluate(conv2).reshape(conv1.shape))

  @test_util.run_in_graph_and_eager_modes
  def testConvolutionWith2SpatialDimensionsAndExpandedBatch(self):
    tensor_in_sizes_batch = [10, 2, 3, 1, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 1, 3]
    filter_in_sizes = [1, 1, 1, 3, 3]
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)
    conv1 = nn_ops.convolution(
        x1, filter_in, strides=[1, 1, 1], padding="VALID")
    conv2 = nn_ops.convolution(
        x2, filter_in, strides=[1, 1, 1], padding="VALID")
    self.assertEqual(conv1.shape, tensor_in_sizes_batch)
    self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
    self.assertAllClose(conv1, self.evaluate(conv2).reshape(conv1.shape))

  def testConv3D1x1x1Filter(self):
    expected_output = [
        0.18518519, 0.22222222, 0.25925926, 0.40740741, 0.5, 0.59259259,
        0.62962963, 0.77777778, 0.92592593, 0.85185185, 1.05555556, 1.25925926,
        1.07407407, 1.33333333, 1.59259259, 1.2962963, 1.61111111, 1.92592593
    ]

    # These are equivalent to the Conv2D1x1 case.
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 1, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 1, 2, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv3D1x1x1Filter2x1x1Dilation(self):
    ctx = context.context()
    is_eager = ctx is not None and ctx.executing_eagerly()
    if test.is_gpu_available(cuda_only=True) or \
      (test_util.IsMklEnabled() and is_eager is False):
      self._VerifyDilatedConvValues(
          tensor_in_sizes=[1, 3, 6, 1, 1],
          filter_in_sizes=[1, 1, 1, 1, 1],
          stride=1,
          padding="VALID",
          dilations=[2, 1, 1])

  # Expected values computed using scipy's correlate function.
  def testConv3D2x2x2Filter(self):
    expected_output = [
        3.77199074, 3.85069444, 3.92939815, 4.2650463, 4.35763889, 4.45023148,
        6.73032407, 6.89236111, 7.05439815, 7.22337963, 7.39930556, 7.57523148,
        9.68865741, 9.93402778, 10.17939815, 10.18171296, 10.44097222,
        10.70023148
    ]
    # expected_shape = [1, 3, 1, 2, 5]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],  # b, z, y, x, fin
        filter_in_sizes=[2, 2, 2, 3, 3],  # z, y, x, fin, fout
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv3D2x2x2Filter1x2x1Dilation(self):
    ctx = context.context()
    is_eager = ctx is not None and ctx.executing_eagerly()
    if test.is_gpu_available(cuda_only=True) or \
      (test_util.IsMklEnabled() and is_eager is False):
      self._VerifyDilatedConvValues(
          tensor_in_sizes=[1, 4, 6, 3, 1],
          filter_in_sizes=[2, 2, 2, 1, 1],
          stride=1,
          padding="VALID",
          dilations=[1, 2, 1])

  def testConv3DStrides(self):
    expected_output = [
        0.06071429, 0.08988095, 0.10238095, 0.11488095, 0.12738095, 0.13988095,
        0.08452381, 0.26071429, 0.35238095, 0.36488095, 0.37738095, 0.38988095,
        0.40238095, 0.23452381, 0.46071429, 0.61488095, 0.62738095, 0.63988095,
        0.65238095, 0.66488095, 0.38452381, 1.12738095, 1.48988095, 1.50238095,
        1.51488095, 1.52738095, 1.53988095, 0.88452381, 1.32738095, 1.75238095,
        1.76488095, 1.77738095, 1.78988095, 1.80238095, 1.03452381, 1.52738095,
        2.01488095, 2.02738095, 2.03988095, 2.05238095, 2.06488095, 1.18452381,
        2.19404762, 2.88988095, 2.90238095, 2.91488095, 2.92738095, 2.93988095,
        1.68452381, 2.39404762, 3.15238095, 3.16488095, 3.17738095, 3.18988095,
        3.20238095, 1.83452381, 2.59404762, 3.41488095, 3.42738095, 3.43988095,
        3.45238095, 3.46488095, 1.98452381
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 5, 8, 7, 1],
        filter_in_sizes=[1, 2, 3, 1, 1],
        stride=[2, 3, 1],  # different stride for each spatial dimension
        padding="SAME",
        expected=expected_output)

  def testConv3D2x2x2FilterStride2(self):
    expected_output = [
        3.77199074, 3.85069444, 3.92939815, 9.68865741, 9.93402778, 10.17939815
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2,
        padding="VALID",
        expected=expected_output)

  def testConv3DStride3(self):
    expected_output = [
        1.51140873, 1.57167659, 1.63194444, 1.56349206, 1.62673611, 1.68998016,
        1.6155754, 1.68179563, 1.74801587, 1.9280754, 2.01215278, 2.09623016,
        1.98015873, 2.0672123, 2.15426587, 2.03224206, 2.12227183, 2.21230159,
        4.4280754, 4.65500992, 4.88194444, 4.48015873, 4.71006944, 4.93998016,
        4.53224206, 4.76512897, 4.99801587, 4.84474206, 5.09548611, 5.34623016,
        4.8968254, 5.15054563, 5.40426587, 4.94890873, 5.20560516, 5.46230159
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 6, 7, 8, 2],
        filter_in_sizes=[3, 2, 1, 2, 3],
        stride=3,
        padding="VALID",
        expected=expected_output)

  def testConv3D2x2x2FilterStride2Same(self):
    expected_output = [
        3.77199074, 3.85069444, 3.92939815, 2.0162037, 2.06597222, 2.11574074,
        9.68865741, 9.93402778, 10.17939815, 4.59953704, 4.73263889, 4.86574074
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2,
        padding="SAME",
        expected=expected_output)

  def _TestConv3DEmptyTensorOutputShape(self):
    """Verifies the output shape of the Conv3D op when output tensor is empty.

    Args: none
    """
    input_shape = [0, 112, 112, 112, 32]
    filter_shape = [3, 3, 3, 32, 64]

    output_shape = [0, 112, 112, 112, 64]
    input_data = 1
    filter_data = 1
    for data_type in self._DtypesToTest(False):
      input_tensor = constant_op.constant(
          input_data, shape=input_shape, dtype=data_type, name="input")
      filter_tensor = constant_op.constant(
          filter_data, shape=filter_shape, dtype=data_type, name="filter")
      conv = nn_ops.conv3d(
          input_tensor,
          filter_tensor,
          strides=[1, 1, 1, 1, 1],
          dilations=[1, 1, 1, 1, 1],
          padding="SAME",
          data_format="NDHWC",
          name="conv")
      values = self.evaluate(conv)
      self.assertEqual(values.shape, tensor_shape.TensorShape(output_shape))

  def testKernelSmallerThanStride(self):
    expected_output = [
        0.03703704, 0.11111111, 0.25925926, 0.33333333, 0.7037037, 0.77777778,
        0.92592593, 1.
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2,
        padding="SAME",
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2,
        padding="VALID",
        expected=expected_output)

    expected_output = [
        0.54081633, 0.58017493, 0.28061224, 0.81632653, 0.85568513, 0.40306122,
        0.41873178, 0.4340379, 0.19642857, 2.46938776, 2.50874636, 1.1377551,
        2.74489796, 2.78425656, 1.26020408, 1.16873178, 1.1840379, 0.51785714,
        1.09511662, 1.10604956, 0.44642857, 1.17164723, 1.18258017, 0.47704082,
        0.3691691, 0.37244898, 0.125
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3,
        padding="SAME",
        expected=expected_output)

    expected_output = [
        0.540816, 0.580175, 0.816327, 0.855685, 2.469388, 2.508746, 2.744898,
        2.784257
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3,
        padding="VALID",
        expected=expected_output)

  def testKernelSizeMatchesInputSize(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 2, 1],
        filter_in_sizes=[2, 1, 2, 1, 2],
        stride=1,
        padding="VALID",
        expected=[1.5625, 1.875])

  def _ConstructAndTestGradientForConfig(
      self, batch, input_shape, filter_shape, in_depth, out_depth, stride,
      padding, test_input, data_format, use_gpu):

    input_planes, input_rows, input_cols = input_shape
    filter_planes, filter_rows, filter_cols = filter_shape

    input_shape = [batch, input_planes, input_rows, input_cols, in_depth]
    filter_shape = [
        filter_planes, filter_rows, filter_cols, in_depth, out_depth
    ]

    if isinstance(stride, collections_abc.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]

    if padding == "VALID":
      output_planes = int(
          math.ceil((input_planes - filter_planes + 1.0) / strides[1]))
      output_rows = int(
          math.ceil((input_rows - filter_rows + 1.0) / strides[2]))
      output_cols = int(
          math.ceil((input_cols - filter_cols + 1.0) / strides[3]))
    else:
      output_planes = int(math.ceil(float(input_planes) / strides[1]))
      output_rows = int(math.ceil(float(input_rows) / strides[2]))
      output_cols = int(math.ceil(float(input_cols) / strides[3]))
    output_shape = [batch, output_planes, output_rows, output_cols, out_depth]
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]

    for data_type in self._DtypesToTest(use_gpu=use_gpu):
      # TODO(mjanusz): Modify gradient_checker to also provide max relative
      # error and synchronize the tolerance levels between the tests for forward
      # and backward computations.
      if data_type == dtypes.float64:
        tolerance = 1e-8
      elif data_type == dtypes.float32:
        tolerance = 5e-3
      elif data_type == dtypes.float16:
        tolerance = 1e-3

      with self.cached_session(use_gpu=use_gpu):
        orig_input_tensor = constant_op.constant(
            input_data, shape=input_shape, dtype=data_type, name="input")
        filter_tensor = constant_op.constant(
            filter_data, shape=filter_shape, dtype=data_type, name="filter")

        if data_format == "NCDHW":
          input_tensor = test_util.NHWCToNCHW(orig_input_tensor)
          new_strides = test_util.NHWCToNCHW(strides)
        else:
          input_tensor = orig_input_tensor
          new_strides = strides

        conv = nn_ops.conv3d(
            input_tensor,
            filter_tensor,
            new_strides,
            padding,
            data_format=data_format,
            name="conv")

        if data_format == "NCDHW":
          conv = test_util.NCHWToNHWC(conv)

        self.assertEqual(conv.shape, tensor_shape.TensorShape(output_shape))

        if test_input:
          jacob_t, jacob_n = gradient_checker.compute_gradient(
              orig_input_tensor, input_shape, conv, output_shape)
        else:
          jacob_t, jacob_n = gradient_checker.compute_gradient(
              filter_tensor, filter_shape, conv, output_shape)

        if data_type != dtypes.float16:
          reference_jacob_t = jacob_t
          err = np.fabs(jacob_t - jacob_n).max()
        else:
          # Compare fp16 theoretical gradients to fp32 theoretical gradients,
          # since fp16 numerical gradients are too imprecise.
          err = np.fabs(jacob_t - reference_jacob_t).max()

      print("conv3d gradient error = ", err)
      self.assertLess(err, tolerance)

  def ConstructAndTestGradient(self, **kwargs):
    for data_format, use_gpu in GetTestConfigs():
      self._ConstructAndTestGradientForConfig(data_format=data_format,
                                              use_gpu=use_gpu, **kwargs)

  @test_util.run_deprecated_v1
  def testInputGradientValidPaddingStrideOne(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 5, 4),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientValidPaddingStrideOne(self):
    self.ConstructAndTestGradient(
        batch=4,
        input_shape=(4, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientValidPaddingStrideTwo(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(6, 3, 5),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="VALID",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientValidPaddingStrideTwo(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(7, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="VALID",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientValidPaddingStrideThree(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 7, 6),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="VALID",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientValidPaddingStrideThree(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(4, 4, 7),
        filter_shape=(4, 4, 4),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="VALID",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientSamePaddingStrideOne(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 2, 2),
        filter_shape=(3, 2, 1),
        in_depth=2,
        out_depth=1,
        stride=1,
        padding="SAME",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientSamePaddingStrideOne(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="SAME",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientSamePaddingStrideTwo(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(6, 3, 4),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="SAME",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientSamePaddingStrideTwo(self):
    self.ConstructAndTestGradient(
        batch=4,
        input_shape=(7, 3, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="SAME",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientSamePaddingStrideThree(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(9, 3, 6),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="SAME",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientSamePaddingStrideThree(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(9, 4, 7),
        filter_shape=(4, 4, 4),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="SAME",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientSamePaddingDifferentStrides(self):
    self.ConstructAndTestGradient(
        batch=1,
        input_shape=(5, 8, 7),
        filter_shape=(1, 2, 3),
        in_depth=2,
        out_depth=3,
        stride=[2, 3, 1],
        padding="SAME",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientKernelSizeMatchesInputSize(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(5, 4, 3),
        filter_shape=(5, 4, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=False)

  @test_util.run_deprecated_v1
  def testInputGradientKernelSizeMatchesInputSize(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(5, 4, 3),
        filter_shape=(5, 4, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=True)

  def disabledtestFilterGradientSamePaddingDifferentStrides(self):
    self.ConstructAndTestGradient(
        batch=1,
        input_shape=(5, 8, 7),
        filter_shape=(1, 2, 3),
        in_depth=2,
        out_depth=3,
        stride=[2, 3, 1],
        padding="SAME",
        test_input=False)

  # Test the fast path in gemm_pack_rhs/gemm_pack_colmajor_block, when channel
  # dimension is a multiple of packet size.
  @test_util.run_deprecated_v1
  def testInputGradientValidPaddingStrideOneFastPath(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 5, 4),
        filter_shape=(2, 2, 2),
        in_depth=8,
        out_depth=2,
        stride=1,
        padding="VALID",
        test_input=True)

  @test_util.run_deprecated_v1
  def testFilterGradientValidPaddingStrideOneFastPath(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(4, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=8,
        out_depth=2,
        stride=1,
        padding="VALID",
        test_input=False)

  # Testing for backprops
  def _RunAndVerifyBackprop(self, input_sizes, filter_sizes, output_sizes,
                            strides, dilations, padding, data_format, use_gpu,
                            err, mode):
    total_input_size = 1
    total_filter_size = 1
    for s in input_sizes:
      total_input_size *= s
    for s in filter_sizes:
      total_filter_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_input_size + 1)]
    x2 = [f * 1.0 for f in range(1, total_filter_size + 1)]
    default_dilations = (
        dilations[0] == 1 and dilations[1] == 1 and dilations[2] == 1)

    # If any dilation rate is larger than 1, only do test on the GPU
    # because we currently do not have a CPU implementation for arbitrary
    # dilation rates.
    if default_dilations or use_gpu:
      with self.cached_session(use_gpu=use_gpu) as sess:
        if data_format == "NCDHW":
          input_sizes = test_util.NHWCToNCHW(input_sizes)
        t1 = constant_op.constant(x1, shape=input_sizes)
        t2 = constant_op.constant(x2, shape=filter_sizes)
        full_strides = [1] + strides + [1]
        full_dilations = [1] + dilations + [1]
        if data_format == "NCDHW":
          full_strides = test_util.NHWCToNCHW(full_strides)
          full_dilations = test_util.NHWCToNCHW(full_dilations)
        actual = nn_ops.conv3d(
            t1,
            t2,
            strides=full_strides,
            dilations=full_dilations,
            padding=padding,
            data_format=data_format)
        expected = nn_ops.convolution(
            t1,
            t2,
            padding=padding,
            strides=strides,
            dilation_rate=dilations,
            data_format=data_format)
        if data_format == "NCDHW":
          actual = test_util.NCHWToNHWC(actual)
          expected = test_util.NCHWToNHWC(expected)
        actual_grad = gradients_impl.gradients(actual, t1
                                               if mode == "input" else t2)[0]
        expected_grad = gradients_impl.gradients(expected, t1
                                                 if mode == "input" else t2)[0]
        # "values" consists of two tensors for two backprops
        actual_value = self.evaluate(actual_grad)
        expected_value = self.evaluate(expected_grad)
        self.assertShapeEqual(actual_value, actual_grad)
        self.assertShapeEqual(expected_value, expected_grad)
      print("expected = ", expected_value)
      print("actual = ", actual_value)
      self.assertArrayNear(expected_value.flatten(), actual_value.flatten(),
                           err)

  @test_util.run_deprecated_v1
  def testConv3D2x2Depth3ValidBackpropFilterStride1x1Dilation2x1(self):
    if test.is_gpu_available(cuda_only=True):
      for (data_format, use_gpu) in GetTestConfigs():
        self._RunAndVerifyBackprop(
            input_sizes=[1, 3, 6, 1, 1],
            filter_sizes=[2, 2, 1, 1, 1],
            output_sizes=[1, 1, 5, 1, 1],
            strides=[1, 1, 1],
            dilations=[2, 1, 1],
            padding="VALID",
            data_format=data_format,
            use_gpu=use_gpu,
            err=1e-5,
            mode="filter")

  @test_util.run_deprecated_v1
  def testConv3D2x2Depth3ValidBackpropInputStride1x1Dilation2x1(self):
    if test.is_gpu_available(cuda_only=True):
      for (data_format, use_gpu) in GetTestConfigs():
        self._RunAndVerifyBackprop(
            input_sizes=[1, 3, 6, 1, 1],
            filter_sizes=[2, 2, 1, 1, 1],
            output_sizes=[1, 1, 5, 1, 1],
            strides=[1, 1, 1],
            dilations=[2, 1, 1],
            padding="VALID",
            data_format=data_format,
            use_gpu=use_gpu,
            err=1e-5,
            mode="input")


if __name__ == "__main__":
  test.main()

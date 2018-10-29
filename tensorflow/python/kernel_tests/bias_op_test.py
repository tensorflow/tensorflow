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
"""Functional tests for BiasAdd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class BiasAddTest(test.TestCase):

  def _npBias(self, inputs, bias):
    assert len(bias.shape) == 1
    print(inputs.shape)
    print(bias.shape)
    assert inputs.shape[-1] == bias.shape[0]
    return inputs + bias.reshape(([1] * (len(inputs.shape) - 1)) +
                                 [bias.shape[0]])

  def testNpBias(self):
    self.assertAllClose(
        np.array([[11, 22, 33], [41, 52, 63]]),
        self._npBias(
            np.array([[10, 20, 30], [40, 50, 60]]), np.array([1, 2, 3])))

  def _testBias(self, np_inputs, np_bias, use_gpu=False):
    np_val = self._npBias(np_inputs, np_bias)
    with self.cached_session(use_gpu=use_gpu):
      tf_val = nn_ops.bias_add(np_inputs, np_bias).eval()
    self.assertAllCloseAccordingToType(np_val, tf_val)

  def _AtLeast3d(self, np_value):
    # fill the input value to at least 3-dimension
    if np_value.ndim < 3:
      return np.reshape(np_value, (1,) * (3 - np_value.ndim) + np_value.shape)
    return np_value

  def _NHWCToNCHW(self, np_value):
    # fill the input value to at least 3-dimension
    np_value = self._AtLeast3d(np_value)
    # move the last dimension to second
    np_dim = list(range(np_value.ndim))
    np_dim_new = list(np_dim[0:1]) + list(np_dim[-1:]) + list(np_dim[1:-1])
    return np.transpose(np_value, np_dim_new)

  def _NCHWToNHWC(self, np_value):
    assert len(np_value.shape) >= 3
    np_dim = list(range(np_value.ndim))
    # move the second dimension to the last
    np_dim_new = list(np_dim[0:1]) + list(np_dim[2:]) + list(np_dim[1:2])
    return np.transpose(np_value, np_dim_new)

  def _testBiasNCHW(self, np_inputs, np_bias, use_gpu):
    np_val = self._npBias(np_inputs, np_bias)
    np_inputs = self._NHWCToNCHW(np_inputs)
    with self.cached_session(use_gpu=use_gpu):
      tf_val = nn_ops.bias_add(np_inputs, np_bias, data_format="NCHW").eval()
    tf_val = self._NCHWToNHWC(tf_val)
    self.assertAllCloseAccordingToType(self._AtLeast3d(np_val), tf_val)

  def _testAll(self, np_inputs, np_bias):
    self._testBias(np_inputs, np_bias, use_gpu=False)
    self._testBiasNCHW(np_inputs, np_bias, use_gpu=False)
    if np_inputs.dtype in [np.float16, np.float32, np.float64]:
      self._testBias(np_inputs, np_bias, use_gpu=True)
      self._testBiasNCHW(np_inputs, np_bias, use_gpu=True)


  def testInputDims(self):
    with self.assertRaises(ValueError):
      nn_ops.bias_add([1, 2], [1])

  def testBiasVec(self):
    with self.assertRaises(ValueError):
      nn_ops.bias_add(
          array_ops.reshape(
              [1, 2], shape=[1, 2]),
          array_ops.reshape(
              [1, 2], shape=[1, 2]))

  def testBiasInputsMatch(self):
    with self.assertRaises(ValueError):
      nn_ops.bias_add(
          array_ops.reshape(
              [1, 2], shape=[1, 2]),
          array_ops.reshape(
              [1], shape=[1]))

  def testIntTypes(self):
    for t in [np.int8, np.int16, np.int32, np.int64]:
      self._testAll(
          np.array([[10, 20, 30], [40, 50, 60]]).astype(t),
          np.array([1, 2, 3]).astype(t))

  def testFloatTypes(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testAll(
          np.random.rand(4, 3, 3).astype(t), np.random.rand(3).astype(t))

  def test4DFloatTypes(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testAll(
          np.random.rand(4, 3, 2, 3).astype(t),
          np.random.rand(3).astype(t))

  def test5DFloatTypes(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testAll(
          np.random.rand(4, 3, 2, 3, 4).astype(t),
          np.random.rand(4).astype(t))

  def _testGradient(self, np_input, bias, dtype, data_format, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      if data_format == "NCHW":
        np_input = self._NHWCToNCHW(np_input)
      input_tensor = constant_op.constant(
          np_input, shape=np_input.shape, dtype=dtype)
      bias_tensor = constant_op.constant(bias, shape=bias.shape, dtype=dtype)
      output_tensor = nn_ops.bias_add(
          input_tensor, bias_tensor, data_format=data_format)
      tensor_jacob_t, tensor_jacob_n = gradient_checker.compute_gradient(
          input_tensor, np_input.shape, output_tensor, np_input.shape)
      bias_jacob_t, bias_jacob_n = gradient_checker.compute_gradient(
          bias_tensor, bias.shape, output_tensor, np_input.shape)

      # Test gradient of BiasAddGrad
      bias_add_grad = gradients_impl.gradients(
          nn_ops.l2_loss(output_tensor), bias_tensor)[0]
      grad_jacob_t, grad_jacob_n = gradient_checker.compute_gradient(
          output_tensor, np_input.shape, bias_add_grad, bias.shape)

      if dtype == np.float16:
        # Compare fp16 theoretical gradients to fp32 numerical gradients,
        # since fp16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker check (in particular, it does not test the op itself,
        # only its gradient), but it's much better than nothing.
        input_tensor = constant_op.constant(
            np_input, shape=np_input.shape, dtype=np.float32)
        bias_tensor = constant_op.constant(
            bias, shape=bias.shape, dtype=np.float32)
        output_tensor = nn_ops.bias_add(
            input_tensor, bias_tensor, data_format=data_format)
        _, tensor_jacob_n = gradient_checker.compute_gradient(input_tensor,
                                                              np_input.shape,
                                                              output_tensor,
                                                              np_input.shape)
        _, bias_jacob_n = gradient_checker.compute_gradient(bias_tensor,
                                                            bias.shape,
                                                            output_tensor,
                                                            np_input.shape)

        bias_add_grad = gradients_impl.gradients(
            nn_ops.l2_loss(output_tensor), bias_tensor)[0]
        _, grad_jacob_n = gradient_checker.compute_gradient(output_tensor,
                                                            np_input.shape,
                                                            bias_add_grad,
                                                            bias.shape)

      threshold = 2e-3
      if dtype == dtypes.float64:
        threshold = 1e-10
      self.assertAllClose(tensor_jacob_t, tensor_jacob_n, threshold, threshold)
      self.assertAllClose(bias_jacob_t, bias_jacob_n, threshold, threshold)
      self.assertAllClose(grad_jacob_t, grad_jacob_n, threshold, threshold)

  def testGradientTensor(self):
    # TODO(yongtang): BiasAddGrad with NCHW only works 4D. Reenable once
    # all dimensions are supported.
    for (data_format, use_gpu) in ("NHWC", False), ("NHWC", True):
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64):
        np_input = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dtype=dtype.as_numpy_dtype).reshape(3, 2)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  def testGradientTensor4D(self):
    # BiasAddGrad with NCHW support 4D so all are enabled.
    for (data_format, use_gpu) in [("NHWC", False), ("NHWC", True),
                                   ("NCHW", False), ("NCHW", True)]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64):
        np_input = np.arange(
            1.0, 49.0, dtype=dtype.as_numpy_dtype).reshape(
                [2, 3, 4, 2]).astype(np.float32)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  def testEmpty(self):
    np.random.seed(7)
    for shape in (0, 0), (2, 0), (0, 2), (4, 3, 0), (4, 0, 3), (0, 4, 3):
      self._testAll(np.random.randn(*shape), np.random.randn(shape[-1]))

  def testEmptyGradient(self):
    # TODO(yongtang): BiasAddGrad with NCHW only works 4D. Reenable once
    # all dimensions are supported.
    for (data_format, use_gpu) in ("NHWC", False), ("NHWC", True):
      for shape in (0, 0), (2, 0), (0, 2), (4, 3, 0), (4, 0, 3), (0, 4, 3):
        self._testGradient(
            np.random.randn(*shape),
            np.random.randn(shape[-1]), dtypes.float64, data_format, use_gpu)


if __name__ == "__main__":
  test.main()

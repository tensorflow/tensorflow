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

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class BiasAddTestBase(test.TestCase):

  def _npBias(self, inputs, bias):
    assert len(bias.shape) == 1
    assert inputs.shape[-1] == bias.shape[0]
    return inputs + bias.reshape(([1] *
                                  (len(inputs.shape) - 1)) + [bias.shape[0]])

  def testNpBias(self):
    self.assertAllClose(
        np.array([[11, 22, 33], [41, 52, 63]]),
        self._npBias(
            np.array([[10, 20, 30], [40, 50, 60]]), np.array([1, 2, 3])))

  def _testBias(self, np_inputs, np_bias, use_gpu=False):
    np_val = self._npBias(np_inputs, np_bias)
    with self.cached_session(use_gpu=use_gpu):
      tf_val = self.evaluate(nn_ops.bias_add(np_inputs, np_bias))
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
      tf_val = self.evaluate(
          nn_ops.bias_add(np_inputs, np_bias, data_format="NCHW"))
    tf_val = self._NCHWToNHWC(tf_val)
    self.assertAllCloseAccordingToType(self._AtLeast3d(np_val), tf_val)

  def _testAll(self, np_inputs, np_bias):
    self._testBias(np_inputs, np_bias, use_gpu=False)
    self._testBiasNCHW(np_inputs, np_bias, use_gpu=False)
    if np_inputs.dtype in [np.float16, np.float32, np.float64, np.int32]:
      self._testBias(np_inputs, np_bias, use_gpu=True)
      self._testBiasNCHW(np_inputs, np_bias, use_gpu=True)

  def _expectedException(self):
    if context.executing_eagerly():
      return errors_impl.InvalidArgumentError
    else:
      return ValueError

  def testInputDims(self):
    with self.assertRaises(self._expectedException()):
      nn_ops.bias_add([1, 2], [1])

  def testBiasVec(self):
    with self.assertRaises(self._expectedException()):
      nn_ops.bias_add(
          array_ops.reshape([1, 2], shape=[1, 2]),
          array_ops.reshape([1, 2], shape=[1, 2]))

  def testBiasInputsMatch(self):
    with self.assertRaises(self._expectedException()):
      nn_ops.bias_add(
          array_ops.reshape([1, 2], shape=[1, 2]),
          array_ops.reshape([1], shape=[1]))

  def testIntTypes(self):
    for t in [np.int8, np.int16, np.int32, np.int64]:
      self._testAll(
          np.array([[10, 20, 30], [40, 50, 60]]).astype(t),
          np.array([1, 2, 3]).astype(t))

  def testFloatTypes(self):
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testAll(
          np.random.rand(4, 3, 3).astype(t),
          np.random.rand(3).astype(t))

  def test4DFloatTypes(self):
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testAll(
          np.random.rand(4, 3, 2, 3).astype(t),
          np.random.rand(3).astype(t))
      self._testAll(
          np.random.rand(2048, 4, 4, 4).astype(t),
          np.random.rand(4).astype(t))
      self._testAll(
          np.random.rand(4, 4, 4, 2048).astype(t),
          np.random.rand(2048).astype(t))

  def test5DFloatTypes(self):
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testAll(
          np.random.rand(4, 3, 2, 3, 4).astype(t),
          np.random.rand(4).astype(t))

  def _random_tensor(self, shape, dtype):
    return constant_op.constant(2 * np.random.rand(*shape) - 1, dtype=dtype)

  def _computeGradient(self, np_input, bias, dtype, data_format):
    input_shape = output_shape = np_input.shape
    bias_shape = bias.shape
    input_tensor = constant_op.constant(
        np_input, shape=input_shape, dtype=dtype)
    bias_tensor = constant_op.constant(bias, shape=bias_shape, dtype=dtype)

    if context.executing_eagerly():

      def bias_add(input_tensor, bias_tensor):
        return nn_ops.bias_add(
            input_tensor, bias_tensor, data_format=data_format)

      # The following is a work-around for TF issue 33660. Instead of
      # calculating the analytical and numerical gradients for both
      # inputs in a single call to compute_gradient, compute_gradient
      # is called for each input separately.
      def bias_add_1(input_tensor):
        return bias_add(input_tensor, bias_tensor)

      def bias_add_2(bias_tensor):
        return bias_add(input_tensor, bias_tensor)

      input_jacob_a, input_jacob_n = gradient_checker_v2.compute_gradient(
          bias_add_1, [input_tensor])
      bias_jacob_a, bias_jacob_n = gradient_checker_v2.compute_gradient(
          bias_add_2, [bias_tensor])

      # Test gradient of BiasAddGrad
      def bias_add_grad_function(upstream_gradients):
        with backprop.GradientTape() as tape:
          tape.watch(bias_tensor)
          bias_add_output = bias_add(input_tensor, bias_tensor)
          gradient_injector_output = bias_add_output * upstream_gradients
          return tape.gradient(gradient_injector_output, bias_tensor)

      upstream_tensor = self._random_tensor(output_shape, dtype)
      grad_jacob_a, grad_jacob_n = gradient_checker_v2.compute_gradient(
          bias_add_grad_function, [upstream_tensor])
    else:
      output_tensor = nn_ops.bias_add(
          input_tensor, bias_tensor, data_format=data_format)
      jacobians = gradient_checker.compute_gradient([input_tensor, bias_tensor],
                                                    [input_shape, bias_shape],
                                                    output_tensor, output_shape)
      (input_jacob_a, input_jacob_n), (bias_jacob_a, bias_jacob_n) = jacobians
      # Test gradient of BiasAddGrad
      if dtype == dtypes.bfloat16:
        # L2Loss is not supported for bfloat16 on CPU.
        output_tensor = math_ops.cast(output_tensor, dtype=dtypes.float32)
      bias_add_grad = gradients_impl.gradients(
          nn_ops.l2_loss(output_tensor), bias_tensor)[0]
      grad_jacob_a, grad_jacob_n = gradient_checker.compute_gradient(
          output_tensor, output_shape, bias_add_grad, bias_shape)

    return ((input_jacob_a, bias_jacob_a, grad_jacob_a),
            (input_jacob_n, bias_jacob_n, grad_jacob_n))

  def _testGradient(self, np_input, bias, dtype, data_format, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      if data_format == "NCHW":
        np_input = self._NHWCToNCHW(np_input)
      jacob_a, jacob_n = self._computeGradient(np_input, bias, dtype,
                                               data_format)
      input_jacob_a, bias_jacob_a, grad_jacob_a = jacob_a
      input_jacob_n, bias_jacob_n, grad_jacob_n = jacob_n

      if dtype in [np.float16, dtypes.bfloat16.as_numpy_dtype]:
        # Compare fp16/bf16 analytical gradients to fp32 numerical gradients,
        # since fp16/bf16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker, but pragmatic, check (in particular, it does not test
        # the op itself, only its gradient).
        _, jacob_n = self._computeGradient(np_input, bias, np.float32,
                                           data_format)
        input_jacob_n, bias_jacob_n, grad_jacob_n = jacob_n

      if dtype == dtypes.float64:
        threshold = 1e-10
      elif np_input.size >= 512:
        # The 5e-3 threshold seems to have been marginal in these cases, and
        # small changes in the test were pushing it over the limit.
        threshold = 5e-2
      else:
        threshold = 5e-3
      self.assertAllClose(input_jacob_a, input_jacob_n, threshold, threshold)
      self.assertAllClose(bias_jacob_a, bias_jacob_n, threshold, threshold)
      self.assertAllClose(grad_jacob_a, grad_jacob_n, threshold, threshold)

  def testGradientTensor2D(self):
    for (data_format, use_gpu) in ("NHWC", False), ("NHWC", True):
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.bfloat16):
        np_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            dtype=dtype.as_numpy_dtype).reshape(3, 2)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  def testGradientTensor3D(self):
    for (data_format, use_gpu) in [("NHWC", False), ("NHWC", True),
                                   ("NCHW", False), ("NCHW", True)]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.bfloat16):
        # pylint: disable=too-many-function-args
        np_input = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dtype=dtype.as_numpy_dtype).reshape(1, 3, 2)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  def testGradientTensor4D(self):
    for (data_format, use_gpu) in [("NHWC", False)]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.bfloat16):
        np_input = np.arange(
            1.0, 49.0,
            dtype=dtype.as_numpy_dtype).reshape([2, 3, 4, 2]).astype(np.float32)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)
        np_input = np.arange(
            1.0, 513.0,
            dtype=dtype.as_numpy_dtype).reshape([64, 2, 2,
                                                 2]).astype(np.float32)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)
        np_input = np.arange(
            1.0, 513.0,
            dtype=dtype.as_numpy_dtype).reshape([2, 2, 2,
                                                 64]).astype(np.float32)
        self._testGradient(np_input,
                           np.random.rand(64).astype(dtype.as_numpy_dtype),
                           dtype, data_format, use_gpu)

  def testGradientTensor5D(self):
    for (data_format, use_gpu) in [("NHWC", False), ("NHWC", True),
                                   ("NCHW", False), ("NCHW", True)]:
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                    dtypes.bfloat16):
        np_input = np.arange(
            1.0, 49.0,
            dtype=dtype.as_numpy_dtype).reshape([1, 2, 3, 4,
                                                 2]).astype(np.float32)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  def test1x1Image(self):
    for (data_format, use_gpu) in [("NHWC", False), ("NCHW", False)]:
      np_input = np.arange(1.0, 129.0).reshape([4, 1, 1, 32]).astype(np.float32)
      self._testGradient(np_input,
                         np.random.rand(32).astype(np.float32), dtypes.float32,
                         data_format, use_gpu)

  def testEmpty(self):
    np.random.seed(7)
    for shape in (0, 0), (2, 0), (0, 2), (4, 3, 0), (4, 0, 3), (0, 4, 3):
      self._testAll(np.random.randn(*shape), np.random.randn(shape[-1]))

  def testEmptyGradient(self):
    for (data_format, use_gpu) in ("NHWC", False), ("NHWC", True):
      for shape in (0, 0), (2, 0), (0, 2):
        self._testGradient(
            np.random.randn(*shape), np.random.randn(shape[-1]), dtypes.float64,
            data_format, use_gpu)

    for (data_format, use_gpu) in [("NHWC", False), ("NHWC", True),
                                   ("NCHW", False), ("NCHW", True)]:
      for shape in (4, 3, 0), (4, 0, 3), (0, 4, 3):
        self._testGradient(
            np.random.randn(*shape), np.random.randn(shape[-1]), dtypes.float64,
            data_format, use_gpu)

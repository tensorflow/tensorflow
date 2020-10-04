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
"""Tests for Python ops defined in math_grad.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class SquaredDifferenceOpTest(test.TestCase):

  def _testGrad(self, left_shape, right_shape):

    if len(left_shape) > len(right_shape):
      output_shape = left_shape
    else:
      output_shape = right_shape
    l = np.random.randn(*left_shape)
    r = np.random.randn(*right_shape)

    with self.cached_session(use_gpu=True):
      left_tensor = constant_op.constant(l, shape=left_shape)
      right_tensor = constant_op.constant(r, shape=right_shape)
      output = math_ops.squared_difference(left_tensor, right_tensor)
      left_err = gradient_checker.compute_gradient_error(
          left_tensor, left_shape, output, output_shape, x_init_value=l)
      right_err = gradient_checker.compute_gradient_error(
          right_tensor, right_shape, output, output_shape, x_init_value=r)
    self.assertLess(left_err, 1e-10)
    self.assertLess(right_err, 1e-10)

  @test_util.run_deprecated_v1
  def testGrad(self):
    self._testGrad([1, 2, 3, 2], [3, 2])
    self._testGrad([2, 4], [3, 2, 4])


class AbsOpTest(test.TestCase):

  def _biasedRandN(self, shape, bias=0.1, sigma=1.0):
    """Returns samples from a normal distribution shifted `bias` away from 0."""
    value = np.random.randn(*shape) * sigma
    return value + np.sign(value) * bias

  def _testGrad(self, shape, dtype=None, max_error=None, bias=None, sigma=None):
    np.random.seed(7)
    if dtype in (dtypes.complex64, dtypes.complex128):
      value = math_ops.complex(
          self._biasedRandN(
              shape, bias=bias, sigma=sigma),
          self._biasedRandN(
              shape, bias=bias, sigma=sigma))
    else:
      value = ops.convert_to_tensor(
          self._biasedRandN(
              shape, bias=bias), dtype=dtype)

    with self.cached_session(use_gpu=True):
      output = math_ops.abs(value)
      error = gradient_checker.compute_gradient_error(
          value, shape, output, output.get_shape().as_list())
    self.assertLess(error, max_error)

  @test_util.run_deprecated_v1
  def testComplexAbs(self):
    # Bias random test values away from zero to avoid numeric instabilities.
    self._testGrad(
        [3, 3], dtype=dtypes.float32, max_error=2e-5, bias=0.1, sigma=1.0)
    self._testGrad(
        [3, 3], dtype=dtypes.complex64, max_error=2e-5, bias=0.1, sigma=1.0)

    # Ensure stability near the pole at zero.
    self._testGrad(
        [3, 3], dtype=dtypes.float32, max_error=100.0, bias=0.0, sigma=0.1)
    self._testGrad(
        [3, 3], dtype=dtypes.complex64, max_error=100.0, bias=0.0, sigma=0.1)


class MinOrMaxGradientTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testMinGradient(self):
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    outputs = math_ops.reduce_min(array_ops.concat([inputs, inputs], 0))
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], outputs, [])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testMaxGradient(self):
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    outputs = math_ops.reduce_max(array_ops.concat([inputs, inputs], 0))
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], outputs, [])
      self.assertLess(error, 1e-4)


class MaximumOrMinimumGradientTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testMaximumGradient(self):
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32)
    outputs = math_ops.maximum(inputs, 3.0)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [4], outputs, [4])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testMinimumGradient(self):
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32)
    outputs = math_ops.minimum(inputs, 2.0)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [4], outputs, [4])
      self.assertLess(error, 1e-4)


class ProdGradientTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testProdGradient(self):
    inputs = constant_op.constant([[1., 2.], [3., 4.]],
                                  dtype=dtypes.float32)
    outputs = math_ops.reduce_prod(inputs)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testProdGradientForNegativeAxis(self):
    inputs = constant_op.constant([[1., 2.], [3., 4.]],
                                  dtype=dtypes.float32)
    outputs = math_ops.reduce_prod(inputs, -1)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testProdGradientComplex(self):
    for dtype in dtypes.complex64, dtypes.complex128:
      inputs = constant_op.constant([[1 + 3j, 2 - 1j], [3j, 4]],
                                    dtype=dtype)
      outputs = math_ops.reduce_prod(inputs)
      with self.cached_session():
        error = gradient_checker.compute_gradient_error(
            inputs, inputs.get_shape().as_list(),
            outputs, outputs.get_shape().as_list())
        self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testProdGradientForNegativeAxisComplex(self):
    for dtype in dtypes.complex64, dtypes.complex128:
      inputs = constant_op.constant([[1 + 3j, 2 - 1j], [3j, 4]],
                                    dtype=dtype)
      outputs = math_ops.reduce_prod(inputs, -1)
      with self.cached_session():
        error = gradient_checker.compute_gradient_error(
            inputs, inputs.get_shape().as_list(),
            outputs, outputs.get_shape().as_list())
        self.assertLess(error, 1e-4)


@test_util.run_all_in_graph_and_eager_modes
class EuclideanNormGradientTest(test.TestCase):

  def testBasic(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([3], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testNegative(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([-3], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testKeepdims(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([3], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testGradientChain(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([3], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x) * 5, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testTwoElements(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([3, -4], dtype=dtype)
      grad = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grad)
      self.assertLess(err, 1e-3)

  def testNegativeZero(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([1.0, -0.0], dtype=dtype)

      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = math_ops.reduce_euclidean_norm(x)

      dx = tape.gradient(y, x)
      dx_answer = constant_op.constant([1.0, -0.0], dtype=dtype)
      self.assertAllClose(dx, dx_answer)
      self.assertAllClose(1.0 / dx, 1.0 / dx_answer)

  def testZeros(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([0.0, -0.0], dtype=dtype)

      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = math_ops.reduce_euclidean_norm(x)

      dx = tape.gradient(y, x)
      dx_answer = constant_op.constant(
          [float("NaN"), float("NaN")], dtype=dtype)
      self.assertAllClose(dx, dx_answer)

  def test2D_1(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[-3, 5], [7, 11]], dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 1e-3)

  def test2D_2(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[-3, 5], [7, 11]], dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 0), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 1e-3)

  def test2D_3(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[-3, 5], [7, 11]], dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 1), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 1e-3)

  def test2D_4(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[3], [4]], dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 1), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 1e-3)

  def test3D_1(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[[-3, 5], [7, 11]], [[13, 17], [19, 23]]],
                               dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          math_ops.reduce_euclidean_norm, [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 2e-3)

  def test3D_2(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[[-3, 5], [7, 11]], [[13, 17], [19, 23]]],
                               dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 0), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 2e-3)

  def test3D_3(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[[-3, 5], [7, 11]], [[13, 17], [19, 23]]],
                               dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 1), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 3e-3)

  def test3D_4(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[[-3, 5], [7, 11]], [[13, 17], [19, 23]]],
                               dtype=dtype)
      grads = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.reduce_euclidean_norm(x, 2), [x])
      err = gradient_checker_v2.max_error(*grads)
      self.assertLess(err, 2e-3)


class SegmentMinOrMaxGradientTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testSegmentMinGradient(self):
    data = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 1], dtype=dtypes.int64)
    segment_min = math_ops.segment_min(data, segment_ids)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(data, [3], segment_min,
                                                      [2])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testSegmentMaxGradient(self):
    data = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 1], dtype=dtypes.int64)
    segment_max = math_ops.segment_max(data, segment_ids)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(data, [3], segment_max,
                                                      [2])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testSegmentMinGradientWithTies(self):
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    data = array_ops.concat([inputs, inputs], 0)
    segment_ids = constant_op.constant([0, 0], dtype=dtypes.int64)
    segment_min = math_ops.segment_min(data, segment_ids)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], segment_min,
                                                      [1])
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testSegmentMaxGradientWithTies(self):
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    data = array_ops.concat([inputs, inputs], 0)
    segment_ids = constant_op.constant([0, 0], dtype=dtypes.int64)
    segment_max = math_ops.segment_max(data, segment_ids)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], segment_max,
                                                      [1])
      self.assertLess(error, 1e-4)


class FloorModGradientTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testFloorModGradient(self):
    # Making sure the input is not near the discontinuity point where
    # x/y == floor(x/y)
    ns = constant_op.constant([17.], dtype=dtypes.float32)
    inputs = constant_op.constant([131.], dtype=dtypes.float32)
    floor_mod = math_ops.floormod(inputs, ns)
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(inputs, [1],
                                                      floor_mod, [1])
      self.assertLess(error, 1e-4)


class DivNoNanGradientTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBasicGradient(self):
    inputs = constant_op.constant(np.arange(-3, 3),
                                  dtype=dtypes.float32)
    outputs = math_ops.div_no_nan(inputs, 1 + math_ops.abs(inputs))
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.get_shape().as_list(), outputs,
          outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithDenominatorIsZero(self):
    x = constant_op.constant(np.arange(-3, 3),
                             dtype=dtypes.float32)
    y = array_ops.zeros_like(x,
                             dtype=dtypes.float32)
    outputs = math_ops.div_no_nan(x, y)
    with self.cached_session():
      dx, dy = gradients.gradients(outputs, [x, y])
      self.assertAllClose(dx, np.zeros(x.shape.as_list()))
      self.assertAllClose(dy, np.zeros(y.shape.as_list()))


class MulNoNanGradientTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBasicGradient(self):
    inputs = constant_op.constant(np.arange(-3, 3), dtype=dtypes.float32)
    outputs = math_ops.mul_no_nan(inputs, 1 + math_ops.abs(inputs))
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.get_shape().as_list(), outputs,
          outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithRhsIsZero(self):
    x_vals = [0, 1.0, np.nan, np.inf, np.NINF]
    x = constant_op.constant(x_vals, dtype=dtypes.float32)
    y = array_ops.zeros_like(x, dtype=dtypes.float32)
    outputs = math_ops.mul_no_nan(x, y)
    with self.cached_session():
      dx, dy = gradients.gradients(outputs, [x, y])
      self.assertAllClose(dx, np.zeros(x.shape.as_list()))
      self.assertAllClose(dy, x_vals)


class XlogyTest(test.TestCase):

  def _xlogy_gradients(self, x, y):
    xlogy_xgrad = self.evaluate(gradients.gradients(math_ops.xlogy(x, y), x)[0])
    xlogy_ygrad = self.evaluate(gradients.gradients(math_ops.xlogy(x, y), y)[0])
    return xlogy_xgrad, xlogy_ygrad

  @test_util.run_deprecated_v1
  def testNonZeroValuesGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xlogy_xgrad, xlogy_ygrad = self._xlogy_gradients(x, y)
      xlogy_expected_xgrad = self.evaluate(math_ops.log(y))
      xlogy_expected_ygrad = self.evaluate(x / y)
      self.assertAllClose(xlogy_expected_xgrad, xlogy_xgrad)
      self.assertAllClose(xlogy_expected_ygrad, xlogy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xlogy_xgrad, xlogy_ygrad = self._xlogy_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xlogy_xgrad)
      self.assertAllClose(zero, xlogy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroYGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(0., dtype=dtype)
      xlogy_xgrad, xlogy_ygrad = self._xlogy_gradients(x, y)
      self.assertAllClose(-np.inf, xlogy_xgrad)
      self.assertAllClose(np.inf, xlogy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXYGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(0., dtype=dtype)
      xlogy_xgrad, xlogy_ygrad = self._xlogy_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xlogy_xgrad)
      self.assertAllClose(zero, xlogy_ygrad)


class Xlog1pyTest(test.TestCase):

  def _xlog1py_gradients(self, x, y):
    xlog1py_xgrad = self.evaluate(
        gradients.gradients(math_ops.xlog1py(x, y), x)[0])
    xlog1py_ygrad = self.evaluate(
        gradients.gradients(math_ops.xlog1py(x, y), y)[0])
    return xlog1py_xgrad, xlog1py_ygrad

  @test_util.run_deprecated_v1
  def testNonZeroValuesGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xlog1py_xgrad, xlog1py_ygrad = self._xlog1py_gradients(x, y)
      xlog1py_expected_xgrad = self.evaluate(math_ops.log1p(y))
      xlog1py_expected_ygrad = self.evaluate(x / (1. + y))
      self.assertAllClose(xlog1py_expected_xgrad, xlog1py_xgrad)
      self.assertAllClose(xlog1py_expected_ygrad, xlog1py_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xlog1py_xgrad, xlog1py_ygrad = self._xlog1py_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xlog1py_xgrad)
      self.assertAllClose(zero, xlog1py_ygrad)

  @test_util.run_deprecated_v1
  def testNegOneYGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(-1., dtype=dtype)
      xlog1py_xgrad, xlog1py_ygrad = self._xlog1py_gradients(x, y)
      self.assertAllClose(-np.inf, xlog1py_xgrad)
      self.assertAllClose(np.inf, xlog1py_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXNegOneYGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(-1., dtype=dtype)
      xlog1py_xgrad, xlog1py_ygrad = self._xlog1py_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xlog1py_xgrad)
      self.assertAllClose(zero, xlog1py_ygrad)


class XdivyTest(test.TestCase):

  def _xdivy_gradients(self, x, y):
    xdivy_xgrad = self.evaluate(gradients.gradients(math_ops.xdivy(x, y), x)[0])
    xdivy_ygrad = self.evaluate(gradients.gradients(math_ops.xdivy(x, y), y)[0])
    return xdivy_xgrad, xdivy_ygrad

  @test_util.run_deprecated_v1
  def testNonZeroValuesGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xdivy_xgrad, xdivy_ygrad = self._xdivy_gradients(x, y)
      xdivy_expected_xgrad = self.evaluate(1 / y)
      xdivy_expected_ygrad = self.evaluate(-x / y**2)
      self.assertAllClose(xdivy_expected_xgrad, xdivy_xgrad)
      self.assertAllClose(xdivy_expected_ygrad, xdivy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(3.1, dtype=dtype)
      xdivy_xgrad, xdivy_ygrad = self._xdivy_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xdivy_xgrad)
      self.assertAllClose(zero, xdivy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroYGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0.1, dtype=dtype)
      y = constant_op.constant(0., dtype=dtype)
      xdivy_xgrad, xdivy_ygrad = self._xdivy_gradients(x, y)
      self.assertAllClose(np.inf, xdivy_xgrad)
      self.assertAllClose(-np.inf, xdivy_ygrad)

  @test_util.run_deprecated_v1
  def testZeroXYGrad(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(0., dtype=dtype)
      y = constant_op.constant(0., dtype=dtype)
      xdivy_xgrad, xdivy_ygrad = self._xdivy_gradients(x, y)
      zero = self.evaluate(x)
      self.assertAllClose(zero, xdivy_xgrad)
      self.assertAllClose(zero, xdivy_ygrad)


@test_util.run_all_in_graph_and_eager_modes
class PowGradTest(test.TestCase):

  def test_zero_grad_tf_gradients(self):
    if context.executing_eagerly():
      self.skipTest("tf.gradients not supported in eager.")

    x = constant_op.constant([-1., 0., 1.])
    g = self.evaluate(gradients.gradients(math_ops.pow(x, 2), x)[0])
    self.assertAllClose([-2., 0., 2.], g)

  def test_zero_grad_tape(self):
    try:
      check_numerics_callback.enable_check_numerics()
      x = constant_op.constant([-1, 0., 1.])
      with backprop.GradientTape() as tape:
        tape.watch(x)
        g = tape.gradient(math_ops.pow(x, 2), x)
      g = self.evaluate(g)
      self.assertAllClose([-2., 0., 2.], g)
    finally:
      check_numerics_callback.disable_check_numerics()


@test_util.run_all_in_graph_and_eager_modes
class NextAfterTest(test.TestCase):

  def _nextafter_gradient(self, x1, x2):
    with backprop.GradientTape() as tape:
      tape.watch(x1)
      tape.watch(x2)
      y = math_ops.nextafter(x1, x2)
      return tape.gradient(y, [x1, x2])

  def testBasic(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x1 = constant_op.constant(0.1, dtype=dtype)
      x2 = constant_op.constant(3.1, dtype=dtype)
      dx1, dx2 = self._nextafter_gradient(x1, x2)
      expected_dx1 = constant_op.constant(1, dtype=dtype)
      expected_dx2 = constant_op.constant(0, dtype=dtype)
      self.assertAllClose(expected_dx1, dx1)
      self.assertAllClose(expected_dx2, dx2)

  def testDynamicShapes(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      default_x1 = constant_op.constant(0.1, dtype=dtype)
      default_x2 = constant_op.constant(3.1, dtype=dtype)
      x1 = array_ops.placeholder_with_default(default_x1, shape=None)
      x2 = array_ops.placeholder_with_default(default_x2, shape=None)
      dx1, dx2 = self._nextafter_gradient(x1, x2)
      expected_dx1 = constant_op.constant(1, dtype=dtype)
      expected_dx2 = constant_op.constant(0, dtype=dtype)
      self.assertAllClose(expected_dx1, dx1)
      self.assertAllClose(expected_dx2, dx2)

  def testWithGradientChecker(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        x1 = np.array([-1, 0, 1, 2, 3], dtype=dtype.as_numpy_dtype)
        x2 = np.array([2, 2, 2, 2, 2], dtype=dtype.as_numpy_dtype)
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: math_ops.nextafter(x, x2), [x1]))  # pylint: disable=cell-var-from-loop
        self.assertLess(err, 1e-3)

  def testBroadcastingWithGradientChecker(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        x1 = np.array([-1, 0, 1, 2, 3], dtype=dtype.as_numpy_dtype)
        x2 = np.array([2], dtype=dtype.as_numpy_dtype)
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                lambda x: math_ops.nextafter(x, x2), [x1]))  # pylint: disable=cell-var-from-loop
        self.assertLess(err, 1e-3)


if __name__ == "__main__":
  test.main()

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
"""Test cases for binary operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class BinaryOpsTest(XLATestCase):
  """Test cases for binary operators."""

  def _testBinary(self, op, a, b, expected, equality_test=None):
    with self.test_session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name="b")
        output = op(pa, pb)
      result = session.run(output, {pa: a, pb: b})
      if equality_test is None:
        equality_test = self.assertAllClose
      equality_test(result, expected, rtol=1e-3)

  def _testSymmetricBinary(self, op, a, b, expected, equality_test=None):
    self._testBinary(op, a, b, expected, equality_test)
    self._testBinary(op, b, a, expected, equality_test)

  def ListsAreClose(self, result, expected, rtol):
    """Tests closeness of two lists of floats."""
    self.assertEqual(len(result), len(expected))
    for i in range(len(result)):
      self.assertAllClose(result[i], expected[i], rtol)

  def testFloatOps(self):
    for dtype in self.float_types:
      self._testBinary(
          lambda x, y: math_ops.approximate_equal(x, y, tolerance=0.0001),
          np.array([[[[-1, 2.00009999], [-3, 4.01]]]], dtype=dtype),
          np.array([[[[-1.001, 2], [-3.00009, 4]]]], dtype=dtype),
          expected=np.array([[[[False, True], [True, False]]]], dtype=dtype))

      self._testBinary(
          gen_math_ops._real_div,
          np.array([3, 3, -1.5, -8, 44], dtype=dtype),
          np.array([2, -2, 7, -4, 0], dtype=dtype),
          expected=np.array(
              [1.5, -1.5, -0.2142857, 2, float("inf")], dtype=dtype))

      self._testBinary(math_ops.pow, dtype(3), dtype(4), expected=dtype(81))

      self._testBinary(
          math_ops.pow,
          np.array([1, 2], dtype=dtype),
          np.zeros(shape=[0, 2], dtype=dtype),
          expected=np.zeros(shape=[0, 2], dtype=dtype))
      self._testBinary(
          math_ops.pow,
          np.array([10, 4], dtype=dtype),
          np.array([2, 3], dtype=dtype),
          expected=np.array([100, 64], dtype=dtype))
      self._testBinary(
          math_ops.pow,
          dtype(2),
          np.array([3, 4], dtype=dtype),
          expected=np.array([8, 16], dtype=dtype))
      self._testBinary(
          math_ops.pow,
          np.array([[2], [3]], dtype=dtype),
          dtype(4),
          expected=np.array([[16], [81]], dtype=dtype))

      self._testBinary(
          math_ops.atan2,
          np.array([0, np.sqrt(2), 1, np.sqrt(2), 0], dtype),
          np.array([1, np.sqrt(2), 0, -np.sqrt(2), -1], dtype),
          expected=np.array(
              [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi], dtype=dtype))

      self._testBinary(
          gen_math_ops._reciprocal_grad,
          np.array([4, -3, -2, 1], dtype=dtype),
          np.array([5, -6, 7, -8], dtype=dtype),
          expected=np.array([-80, 54, -28, 8], dtype=dtype))

      self._testBinary(
          gen_math_ops._sigmoid_grad,
          np.array([4, 3, 2, 1], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([-60, -36, -14, 0], dtype=dtype))

      self._testBinary(
          gen_math_ops._rsqrt_grad,
          np.array([4, 3, 2, 1], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([-160, -81, -28, -4], dtype=dtype))

      self._testBinary(
          gen_math_ops._sqrt_grad,
          np.array([4, 3, 2, 1], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([0.625, 1, 1.75, 4], dtype=dtype))

      self._testBinary(
          gen_nn_ops._softplus_grad,
          np.array([4, 3, 2, 1], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array(
              [3.97322869, 2.99258232, 1.99817801, 0.99966466], dtype=dtype))

      self._testBinary(
          gen_nn_ops._softsign_grad,
          np.array([4, 3, 2, 1], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array(
              [0.11111111, 0.06122449, 0.03125, 0.01234568], dtype=dtype))

      self._testBinary(
          gen_math_ops._tanh_grad,
          np.array([4, 3, 2, 1], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array([-75, -48, -21, 0], dtype=dtype))

      self._testBinary(
          gen_nn_ops._elu_grad,
          np.array([1, 2, 3, 4, 5, 6], dtype=dtype),
          np.array([-.6, -.4, -.2, 0, .2, .4], dtype=dtype),
          expected=np.array([0.4, 1.2, 2.4, 4, 5, 6], dtype=dtype))

      self._testBinary(
          gen_nn_ops._selu_grad,
          np.array([1, 2, 3, 4, 5, 6], dtype=dtype),
          np.array([-.6, -.4, -.2, .2, .4, .6], dtype=dtype),
          expected=np.array(
              [1.158099340847, 2.7161986816948, 4.67429802254,
               4.202803949422, 5.2535049367774, 6.30420592413], dtype=dtype))

      self._testBinary(
          gen_nn_ops._relu_grad,
          np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype),
          np.array([0, 0, 0, 0, 0, 0.1, 0.3, 0.5, 0.7, 0.9], dtype=dtype),
          expected=np.array([0, 0, 0, 0, 0, 6, 7, 8, 9, 10], dtype=dtype))

      self._testBinary(
          gen_nn_ops._relu6_grad,
          np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=dtype),
          np.array(
              [0, 0, 0, 0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 6.1, 10.0], dtype=dtype),
          expected=np.array([0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 0], dtype=dtype))

      self._testBinary(
          gen_nn_ops._softmax_cross_entropy_with_logits,
          np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype),
          np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=dtype),
          expected=[
              np.array([1.44019, 2.44019], dtype=dtype),
              np.array([[-0.067941, -0.112856, -0.063117, 0.243914],
                        [-0.367941, -0.212856, 0.036883, 0.543914]],
                       dtype=dtype),
          ],
          equality_test=self.ListsAreClose)

      self._testBinary(
          gen_nn_ops._sparse_softmax_cross_entropy_with_logits,
          np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2]], dtype=dtype),
          np.array([2, 1, 7], dtype=np.int32),
          expected=[
              np.array([1.342536, 1.442536, np.nan], dtype=dtype),
              np.array([[0.213838, 0.236328, -0.738817, 0.288651],
                        [0.213838, -0.763672, 0.261183, 0.288651],
                        [np.nan, np.nan, np.nan, np.nan]],
                       dtype=dtype),
          ],
          equality_test=self.ListsAreClose)

  def testIntOps(self):
    for dtype in self.int_types:
      self._testBinary(
          gen_math_ops._truncate_div,
          np.array([3, 3, -1, -9, -8], dtype=dtype),
          np.array([2, -2, 7, 2, -4], dtype=dtype),
          expected=np.array([1, -1, 0, -4, 2], dtype=dtype))
      self._testSymmetricBinary(
          bitwise_ops.bitwise_and,
          np.array([0b1, 0b101, 0b1000], dtype=dtype),
          np.array([0b0, 0b101, 0b1001], dtype=dtype),
          expected=np.array([0b0, 0b101, 0b1000], dtype=dtype))
      self._testSymmetricBinary(
          bitwise_ops.bitwise_or,
          np.array([0b1, 0b101, 0b1000], dtype=dtype),
          np.array([0b0, 0b101, 0b1001], dtype=dtype),
          expected=np.array([0b1, 0b101, 0b1001], dtype=dtype))

      lhs = np.array([0, 5, 3, 14], dtype=dtype)
      rhs = np.array([5, 0, 7, 11], dtype=dtype)
      self._testBinary(
          bitwise_ops.left_shift, lhs, rhs,
          expected=np.left_shift(lhs, rhs))
      self._testBinary(
          bitwise_ops.right_shift, lhs, rhs,
          expected=np.right_shift(lhs, rhs))

      if dtype in [np.int8, np.int16, np.int32, np.int64]:
        lhs = np.array([-1, -5, -3, -14], dtype=dtype)
        rhs = np.array([5, 0, 1, 11], dtype=dtype)
        self._testBinary(
            bitwise_ops.right_shift, lhs, rhs,
            expected=np.right_shift(lhs, rhs))

  def testNumericOps(self):
    for dtype in self.numeric_types:
      self._testBinary(
          math_ops.add,
          np.array([1, 2], dtype=dtype),
          np.array([10, 20], dtype=dtype),
          expected=np.array([11, 22], dtype=dtype))
      self._testBinary(
          math_ops.add,
          dtype(5),
          np.array([1, 2], dtype=dtype),
          expected=np.array([6, 7], dtype=dtype))
      self._testBinary(
          math_ops.add,
          np.array([[1], [2]], dtype=dtype),
          dtype(7),
          expected=np.array([[8], [9]], dtype=dtype))

      self._testBinary(
          math_ops.subtract,
          np.array([1, 2], dtype=dtype),
          np.array([10, 20], dtype=dtype),
          expected=np.array([-9, -18], dtype=dtype))
      self._testBinary(
          math_ops.subtract,
          dtype(5),
          np.array([1, 2], dtype=dtype),
          expected=np.array([4, 3], dtype=dtype))
      self._testBinary(
          math_ops.subtract,
          np.array([[1], [2]], dtype=dtype),
          dtype(7),
          expected=np.array([[-6], [-5]], dtype=dtype))

      if dtype not in self.complex_types:  # min/max not supported for complex
        self._testBinary(
            math_ops.maximum,
            np.array([1, 2], dtype=dtype),
            np.array([10, 20], dtype=dtype),
            expected=np.array([10, 20], dtype=dtype))
        self._testBinary(
            math_ops.maximum,
            dtype(5),
            np.array([1, 20], dtype=dtype),
            expected=np.array([5, 20], dtype=dtype))
        self._testBinary(
            math_ops.maximum,
            np.array([[10], [2]], dtype=dtype),
            dtype(7),
            expected=np.array([[10], [7]], dtype=dtype))

        self._testBinary(
            math_ops.minimum,
            np.array([1, 20], dtype=dtype),
            np.array([10, 2], dtype=dtype),
            expected=np.array([1, 2], dtype=dtype))
        self._testBinary(
            math_ops.minimum,
            dtype(5),
            np.array([1, 20], dtype=dtype),
            expected=np.array([1, 5], dtype=dtype))
        self._testBinary(
            math_ops.minimum,
            np.array([[10], [2]], dtype=dtype),
            dtype(7),
            expected=np.array([[7], [2]], dtype=dtype))

      self._testBinary(
          math_ops.multiply,
          np.array([1, 20], dtype=dtype),
          np.array([10, 2], dtype=dtype),
          expected=np.array([10, 40], dtype=dtype))
      self._testBinary(
          math_ops.multiply,
          dtype(5),
          np.array([1, 20], dtype=dtype),
          expected=np.array([5, 100], dtype=dtype))
      self._testBinary(
          math_ops.multiply,
          np.array([[10], [2]], dtype=dtype),
          dtype(7),
          expected=np.array([[70], [14]], dtype=dtype))

      # Complex support for squared_difference is incidental, see b/68205550
      if dtype not in self.complex_types:
        self._testBinary(
            math_ops.squared_difference,
            np.array([1, 2], dtype=dtype),
            np.array([10, 20], dtype=dtype),
            expected=np.array([81, 324], dtype=dtype))
        self._testBinary(
            math_ops.squared_difference,
            dtype(5),
            np.array([1, 2], dtype=dtype),
            expected=np.array([16, 9], dtype=dtype))
        self._testBinary(
            math_ops.squared_difference,
            np.array([[1], [2]], dtype=dtype),
            dtype(7),
            expected=np.array([[36], [25]], dtype=dtype))

      self._testBinary(
          nn_ops.bias_add,
          np.array([[1, 2], [3, 4]], dtype=dtype),
          np.array([2, -1], dtype=dtype),
          expected=np.array([[3, 1], [5, 3]], dtype=dtype))
      self._testBinary(
          nn_ops.bias_add,
          np.array([[[[1, 2], [3, 4]]]], dtype=dtype),
          np.array([2, -1], dtype=dtype),
          expected=np.array([[[[3, 1], [5, 3]]]], dtype=dtype))

  def testComplexOps(self):
    for dtype in self.complex_types:
      ctypes = {np.complex64: np.float32}
      self._testBinary(
          math_ops.complex,
          np.array([[[[-1, 2], [2, 0]]]], dtype=ctypes[dtype]),
          np.array([[[[2, -3], [0, 4]]]], dtype=ctypes[dtype]),
          expected=np.array([[[[-1 + 2j, 2 - 3j], [2, 4j]]]], dtype=dtype))

      self._testBinary(
          lambda x, y: math_ops.approximate_equal(x, y, tolerance=0.0001),
          np.array(
              [[[[-1 + 2j, 2.00009999 - 3j], [2 - 3j, 3 + 4.01j]]]],
              dtype=dtype),
          np.array(
              [[[[-1.001 + 2j, 2 - 3j], [2 - 3.00009j, 3 + 4j]]]], dtype=dtype),
          expected=np.array([[[[False, True], [True, False]]]], dtype=dtype))

      self._testBinary(
          gen_math_ops._real_div,
          np.array([3, 3j, -1.5j, -8, 2 + 3j, 2 + 4j], dtype=dtype),
          np.array([2, -2, 7j, -4j, 4 - 6j, 1 + 2j], dtype=dtype),
          expected=np.array(
              [1.5, -1.5j, -0.2142857, -2j, (2 + 3j) / (4 - 6j), 2],
              dtype=dtype))

      # Test inf/nan scenarios.
      self._testBinary(
          gen_math_ops._real_div,
          np.array([4 + 3j, 4, 3j, -4, -4j, 2 - 3j], dtype=dtype),
          np.array([0, 0, 0, 0, 0, 0], dtype=dtype),
          expected=np.array(
              [
                  dtype(1 + 1j) / 0,
                  dtype(1) / 0,
                  dtype(1j) / 0,
                  dtype(-1) / 0,
                  dtype(-1j) / 0,
                  dtype(1 - 1j) / 0
              ],
              dtype=dtype))

      self._testBinary(
          math_ops.pow,
          dtype(3 + 2j),
          dtype(4 - 5j),
          expected=np.power(dtype(3 + 2j), dtype(4 - 5j)))
      self._testBinary(  # empty rhs
          math_ops.pow,
          np.array([1 + 2j, 2 - 3j], dtype=dtype),
          np.zeros(shape=[0, 2], dtype=dtype),
          expected=np.zeros(shape=[0, 2], dtype=dtype))
      self._testBinary(  # to zero power
          math_ops.pow,
          np.array([1 + 2j, 2 - 3j], dtype=dtype),
          np.zeros(shape=[1, 2], dtype=dtype),
          expected=np.ones(shape=[1, 2], dtype=dtype))
      lhs = np.array([1 - 2j, 4 + 3j, 2 - 3j, 3, 2j, 1, 4], dtype=dtype)
      rhs = np.array([2, 3j, 3 + 4j, 2 + 3j, 3 - 2j, 2, 3 + 3j], dtype=dtype)
      scalar = dtype(2 + 2j)
      self._testBinary(math_ops.pow, lhs, rhs, expected=np.power(lhs, rhs))
      self._testBinary(
          math_ops.pow, scalar, rhs, expected=np.power(scalar, rhs))
      self._testBinary(math_ops.pow, lhs, scalar, np.power(lhs, scalar))

      lhs = np.array([4 + 2j, -3 - 1j, 2j, 1], dtype=dtype)
      rhs = np.array([5, -6j, 7 - 3j, -8j], dtype=dtype)
      self._testBinary(
          gen_math_ops._reciprocal_grad, lhs, rhs, expected=-rhs * lhs * lhs)

      self._testBinary(
          gen_math_ops._sigmoid_grad, lhs, rhs, expected=rhs * lhs * (1 - lhs))

      self._testBinary(
          gen_math_ops._rsqrt_grad, lhs, rhs, expected=lhs**3 * rhs / -2)

      self._testBinary(
          gen_math_ops._sqrt_grad, lhs, rhs, expected=rhs / (2 * lhs))

      self._testBinary(
          gen_math_ops._tanh_grad, lhs, rhs, expected=rhs * (1 - lhs * lhs))

  def testComplexMath(self):
    for dtype in self.complex_types:
      self._testBinary(
          math_ops.add,
          np.array([1 + 3j, 2 + 7j], dtype=dtype),
          np.array([10 - 4j, 20 + 17j], dtype=dtype),
          expected=np.array([11 - 1j, 22 + 24j], dtype=dtype))
      self._testBinary(
          math_ops.add,
          dtype(5 - 7j),
          np.array([1 + 2j, 2 + 4j], dtype=dtype),
          expected=np.array([6 - 5j, 7 - 3j], dtype=dtype))
      self._testBinary(
          math_ops.add,
          np.array([[1 - 2j], [2 + 1j]], dtype=dtype),
          dtype(7 + 5j),
          expected=np.array([[8 + 3j], [9 + 6j]], dtype=dtype))

      self._testBinary(
          math_ops.subtract,
          np.array([1 + 3j, 2 + 7j], dtype=dtype),
          np.array([10 - 4j, 20 + 17j], dtype=dtype),
          expected=np.array([-9 + 7j, -18 - 10j], dtype=dtype))
      self._testBinary(
          math_ops.subtract,
          dtype(5 - 7j),
          np.array([1 + 2j, 2 + 4j], dtype=dtype),
          expected=np.array([4 - 9j, 3 - 11j], dtype=dtype))
      self._testBinary(
          math_ops.subtract,
          np.array([[1 - 2j], [2 + 1j]], dtype=dtype),
          dtype(7 + 5j),
          expected=np.array([[-6 - 7j], [-5 - 4j]], dtype=dtype))

      self._testBinary(
          math_ops.multiply,
          np.array([1 + 3j, 2 + 7j], dtype=dtype),
          np.array([10 - 4j, 20 + 17j], dtype=dtype),
          expected=np.array(
              [(1 + 3j) * (10 - 4j), (2 + 7j) * (20 + 17j)], dtype=dtype))
      self._testBinary(
          math_ops.multiply,
          dtype(5 - 7j),
          np.array([1 + 2j, 2 + 4j], dtype=dtype),
          expected=np.array(
              [(5 - 7j) * (1 + 2j), (5 - 7j) * (2 + 4j)], dtype=dtype))
      self._testBinary(
          math_ops.multiply,
          np.array([[1 - 2j], [2 + 1j]], dtype=dtype),
          dtype(7 + 5j),
          expected=np.array(
              [[(7 + 5j) * (1 - 2j)], [(7 + 5j) * (2 + 1j)]], dtype=dtype))

      self._testBinary(
          math_ops.div,
          np.array([8 - 1j, 2 + 16j], dtype=dtype),
          np.array([2 + 4j, 4 - 8j], dtype=dtype),
          expected=np.array(
              [(8 - 1j) / (2 + 4j), (2 + 16j) / (4 - 8j)], dtype=dtype))
      self._testBinary(
          math_ops.div,
          dtype(1 + 2j),
          np.array([2 + 4j, 4 - 8j], dtype=dtype),
          expected=np.array(
              [(1 + 2j) / (2 + 4j), (1 + 2j) / (4 - 8j)], dtype=dtype))
      self._testBinary(
          math_ops.div,
          np.array([2 + 4j, 4 - 8j], dtype=dtype),
          dtype(1 + 2j),
          expected=np.array(
              [(2 + 4j) / (1 + 2j), (4 - 8j) / (1 + 2j)], dtype=dtype))

      # TODO(b/68205550): math_ops.squared_difference shouldn't be supported.

      self._testBinary(
          nn_ops.bias_add,
          np.array([[1 + 2j, 2 + 7j], [3 - 5j, 4 + 2j]], dtype=dtype),
          np.array([2 + 6j, -1 - 3j], dtype=dtype),
          expected=np.array([[3 + 8j, 1 + 4j], [5 + 1j, 3 - 1j]], dtype=dtype))
      self._testBinary(
          nn_ops.bias_add,
          np.array([[[[1 + 4j, 2 - 1j], [3 + 7j, 4]]]], dtype=dtype),
          np.array([2 + 1j, -1 + 2j], dtype=dtype),
          expected=np.array(
              [[[[3 + 5j, 1 + 1j], [5 + 8j, 3 + 2j]]]], dtype=dtype))

  def _testDivision(self, dtype):
    """Test cases for division operators."""
    self._testBinary(
        math_ops.div,
        np.array([10, 20], dtype=dtype),
        np.array([10, 2], dtype=dtype),
        expected=np.array([1, 10], dtype=dtype))
    self._testBinary(
        math_ops.div,
        dtype(40),
        np.array([2, 20], dtype=dtype),
        expected=np.array([20, 2], dtype=dtype))
    self._testBinary(
        math_ops.div,
        np.array([[10], [4]], dtype=dtype),
        dtype(2),
        expected=np.array([[5], [2]], dtype=dtype))

    if dtype not in self.complex_types:  # floordiv unsupported for complex.
      self._testBinary(
          gen_math_ops._floor_div,
          np.array([3, 3, -1, -9, -8], dtype=dtype),
          np.array([2, -2, 7, 2, -4], dtype=dtype),
          expected=np.array([1, -2, -1, -5, 2], dtype=dtype))

  def testIntDivision(self):
    for dtype in self.int_types:
      self._testDivision(dtype)

  def testFloatDivision(self):
    for dtype in self.float_types | self.complex_types:
      self._testDivision(dtype)

  def _testRemainder(self, dtype):
    """Test cases for remainder operators."""
    self._testBinary(
        gen_math_ops._floor_mod,
        np.array([3, 3, -1, -8], dtype=dtype),
        np.array([2, -2, 7, -4], dtype=dtype),
        expected=np.array([1, -1, 6, 0], dtype=dtype))
    self._testBinary(
        gen_math_ops._truncate_mod,
        np.array([3, 3, -1, -8], dtype=dtype),
        np.array([2, -2, 7, -4], dtype=dtype),
        expected=np.array([1, 1, -1, 0], dtype=dtype))

  def testIntRemainder(self):
    for dtype in self.int_types:
      self._testRemainder(dtype)

  def testFloatRemainder(self):
    for dtype in self.float_types:
      self._testRemainder(dtype)

  def testLogicalOps(self):
    self._testBinary(
        math_ops.logical_and,
        np.array([[True, False], [False, True]], dtype=np.bool),
        np.array([[False, True], [False, True]], dtype=np.bool),
        expected=np.array([[False, False], [False, True]], dtype=np.bool))

    self._testBinary(
        math_ops.logical_or,
        np.array([[True, False], [False, True]], dtype=np.bool),
        np.array([[False, True], [False, True]], dtype=np.bool),
        expected=np.array([[True, True], [False, True]], dtype=np.bool))

  def testComparisons(self):
    self._testBinary(
        math_ops.equal,
        np.array([1, 5, 20], dtype=np.float32),
        np.array([10, 5, 2], dtype=np.float32),
        expected=np.array([False, True, False], dtype=np.bool))
    self._testBinary(
        math_ops.equal,
        np.float32(5),
        np.array([1, 5, 20], dtype=np.float32),
        expected=np.array([False, True, False], dtype=np.bool))
    self._testBinary(
        math_ops.equal,
        np.array([[10], [7], [2]], dtype=np.float32),
        np.float32(7),
        expected=np.array([[False], [True], [False]], dtype=np.bool))

    self._testBinary(
        math_ops.not_equal,
        np.array([1, 5, 20], dtype=np.float32),
        np.array([10, 5, 2], dtype=np.float32),
        expected=np.array([True, False, True], dtype=np.bool))
    self._testBinary(
        math_ops.not_equal,
        np.float32(5),
        np.array([1, 5, 20], dtype=np.float32),
        expected=np.array([True, False, True], dtype=np.bool))
    self._testBinary(
        math_ops.not_equal,
        np.array([[10], [7], [2]], dtype=np.float32),
        np.float32(7),
        expected=np.array([[True], [False], [True]], dtype=np.bool))

    for greater_op in [math_ops.greater, (lambda x, y: x > y)]:
      self._testBinary(
          greater_op,
          np.array([1, 5, 20], dtype=np.float32),
          np.array([10, 5, 2], dtype=np.float32),
          expected=np.array([False, False, True], dtype=np.bool))
      self._testBinary(
          greater_op,
          np.float32(5),
          np.array([1, 5, 20], dtype=np.float32),
          expected=np.array([True, False, False], dtype=np.bool))
      self._testBinary(
          greater_op,
          np.array([[10], [7], [2]], dtype=np.float32),
          np.float32(7),
          expected=np.array([[True], [False], [False]], dtype=np.bool))

    for greater_equal_op in [math_ops.greater_equal, (lambda x, y: x >= y)]:
      self._testBinary(
          greater_equal_op,
          np.array([1, 5, 20], dtype=np.float32),
          np.array([10, 5, 2], dtype=np.float32),
          expected=np.array([False, True, True], dtype=np.bool))
      self._testBinary(
          greater_equal_op,
          np.float32(5),
          np.array([1, 5, 20], dtype=np.float32),
          expected=np.array([True, True, False], dtype=np.bool))
      self._testBinary(
          greater_equal_op,
          np.array([[10], [7], [2]], dtype=np.float32),
          np.float32(7),
          expected=np.array([[True], [True], [False]], dtype=np.bool))

    for less_op in [math_ops.less, (lambda x, y: x < y)]:
      self._testBinary(
          less_op,
          np.array([1, 5, 20], dtype=np.float32),
          np.array([10, 5, 2], dtype=np.float32),
          expected=np.array([True, False, False], dtype=np.bool))
      self._testBinary(
          less_op,
          np.float32(5),
          np.array([1, 5, 20], dtype=np.float32),
          expected=np.array([False, False, True], dtype=np.bool))
      self._testBinary(
          less_op,
          np.array([[10], [7], [2]], dtype=np.float32),
          np.float32(7),
          expected=np.array([[False], [False], [True]], dtype=np.bool))

    for less_equal_op in [math_ops.less_equal, (lambda x, y: x <= y)]:
      self._testBinary(
          less_equal_op,
          np.array([1, 5, 20], dtype=np.float32),
          np.array([10, 5, 2], dtype=np.float32),
          expected=np.array([True, True, False], dtype=np.bool))
      self._testBinary(
          less_equal_op,
          np.float32(5),
          np.array([1, 5, 20], dtype=np.float32),
          expected=np.array([False, True, True], dtype=np.bool))
      self._testBinary(
          less_equal_op,
          np.array([[10], [7], [2]], dtype=np.float32),
          np.float32(7),
          expected=np.array([[False], [True], [True]], dtype=np.bool))

  def testBroadcasting(self):
    """Tests broadcasting behavior of an operator."""

    for dtype in self.numeric_types:
      self._testBinary(
          math_ops.add,
          np.array(3, dtype=dtype),
          np.array([10, 20], dtype=dtype),
          expected=np.array([13, 23], dtype=dtype))
      self._testBinary(
          math_ops.add,
          np.array([10, 20], dtype=dtype),
          np.array(4, dtype=dtype),
          expected=np.array([14, 24], dtype=dtype))

      # [1,3] x [4,1] => [4,3]
      self._testBinary(
          math_ops.add,
          np.array([[10, 20, 30]], dtype=dtype),
          np.array([[1], [2], [3], [4]], dtype=dtype),
          expected=np.array(
              [[11, 21, 31], [12, 22, 32], [13, 23, 33], [14, 24, 34]],
              dtype=dtype))

      # [3] * [4,1] => [4,3]
      self._testBinary(
          math_ops.add,
          np.array([10, 20, 30], dtype=dtype),
          np.array([[1], [2], [3], [4]], dtype=dtype),
          expected=np.array(
              [[11, 21, 31], [12, 22, 32], [13, 23, 33], [14, 24, 34]],
              dtype=dtype))

  def testFill(self):
    for dtype in self.numeric_types:
      self._testBinary(
          array_ops.fill,
          np.array([], dtype=np.int32),
          dtype(-42),
          expected=dtype(-42))
      self._testBinary(
          array_ops.fill,
          np.array([1, 2], dtype=np.int32),
          dtype(7),
          expected=np.array([[7, 7]], dtype=dtype))
      self._testBinary(
          array_ops.fill,
          np.array([3, 2], dtype=np.int32),
          dtype(50),
          expected=np.array([[50, 50], [50, 50], [50, 50]], dtype=dtype))

  # Helper method used by testMatMul, testSparseMatMul, testBatchMatMul below.
  def _testMatMul(self, op):
    for dtype in self.float_types:
      self._testBinary(
          op,
          np.array([[-0.25]], dtype=dtype),
          np.array([[8]], dtype=dtype),
          expected=np.array([[-2]], dtype=dtype))
      self._testBinary(
          op,
          np.array([[100, 10, 0.5]], dtype=dtype),
          np.array([[1, 3], [2, 5], [6, 8]], dtype=dtype),
          expected=np.array([[123, 354]], dtype=dtype))
      self._testBinary(
          op,
          np.array([[1, 3], [2, 5], [6, 8]], dtype=dtype),
          np.array([[100], [10]], dtype=dtype),
          expected=np.array([[130], [250], [680]], dtype=dtype))
      self._testBinary(
          op,
          np.array([[1000, 100], [10, 1]], dtype=dtype),
          np.array([[1, 2], [3, 4]], dtype=dtype),
          expected=np.array([[1300, 2400], [13, 24]], dtype=dtype))

      self._testBinary(
          op,
          np.array([], dtype=dtype).reshape((2, 0)),
          np.array([], dtype=dtype).reshape((0, 3)),
          expected=np.array([[0, 0, 0], [0, 0, 0]], dtype=dtype))

  def testMatMul(self):
    self._testMatMul(math_ops.matmul)

  # TODO(phawkins): failing on GPU, no registered kernel.
  def DISABLED_testSparseMatMul(self):
    # Binary wrappers for sparse_matmul with different hints
    def SparseMatmulWrapperTF(a, b):
      return tf.sparse_matmul(a, b, a_is_sparse=True)

    def SparseMatmulWrapperFT(a, b):
      return tf.sparse_matmul(a, b, b_is_sparse=True)

    def SparseMatmulWrapperTT(a, b):
      return tf.sparse_matmul(a, b, a_is_sparse=True, b_is_sparse=True)

    self._testMatMul(tf.sparse_matmul)
    self._testMatMul(SparseMatmulWrapperTF)
    self._testMatMul(SparseMatmulWrapperFT)
    self._testMatMul(SparseMatmulWrapperTT)

  def testBatchMatMul(self):
    # Same tests as for tf.matmul above.
    self._testMatMul(math_ops.matmul)

    # Tests with batches of matrices.
    self._testBinary(
        math_ops.matmul,
        np.array([[[-0.25]]], dtype=np.float32),
        np.array([[[8]]], dtype=np.float32),
        expected=np.array([[[-2]]], dtype=np.float32))
    self._testBinary(
        math_ops.matmul,
        np.array([[[-0.25]], [[4]]], dtype=np.float32),
        np.array([[[8]], [[2]]], dtype=np.float32),
        expected=np.array([[[-2]], [[8]]], dtype=np.float32))
    self._testBinary(
        math_ops.matmul,
        np.array(
            [[[[7, 13], [10, 1]], [[2, 0.25], [20, 2]]],
             [[[3, 5], [30, 3]], [[0.75, 1], [40, 4]]]],
            dtype=np.float32),
        np.array(
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[11, 22], [33, 44]],
                                                    [[55, 66], [77, 88]]]],
            dtype=np.float32),
        expected=np.array(
            [[[[46, 66], [13, 24]], [[11.75, 14], [114, 136]]],
             [[[198, 286], [429, 792]], [[118.25, 137.5], [2508, 2992]]]],
            dtype=np.float32))

    self._testBinary(
        math_ops.matmul,
        np.array([], dtype=np.float32).reshape((2, 2, 0)),
        np.array([], dtype=np.float32).reshape((2, 0, 3)),
        expected=np.array(
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
            dtype=np.float32))
    self._testBinary(
        math_ops.matmul,
        np.array([], dtype=np.float32).reshape((0, 2, 4)),
        np.array([], dtype=np.float32).reshape((0, 4, 3)),
        expected=np.array([], dtype=np.float32).reshape(0, 2, 3))

    # Regression test for b/31472796.
    if hasattr(np, "matmul"):
      x = np.arange(0, 3 * 5 * 2 * 7, dtype=np.float32).reshape((3, 5, 2, 7))
      self._testBinary(
          lambda x, y: math_ops.matmul(x, y, adjoint_b=True),
          x, x,
          expected=np.matmul(x, x.transpose([0, 1, 3, 2])))

  def testExpandDims(self):
    for dtype in self.numeric_types:
      self._testBinary(
          array_ops.expand_dims,
          dtype(7),
          np.int32(0),
          expected=np.array([7], dtype=dtype))
      self._testBinary(
          array_ops.expand_dims,
          np.array([42], dtype=dtype),
          np.int32(0),
          expected=np.array([[42]], dtype=dtype))
      self._testBinary(
          array_ops.expand_dims,
          np.array([], dtype=dtype),
          np.int32(0),
          expected=np.array([[]], dtype=dtype))
      self._testBinary(
          array_ops.expand_dims,
          np.array([[[1, 2], [3, 4]]], dtype=dtype),
          np.int32(0),
          expected=np.array([[[[1, 2], [3, 4]]]], dtype=dtype))
      self._testBinary(
          array_ops.expand_dims,
          np.array([[[1, 2], [3, 4]]], dtype=dtype),
          np.int32(1),
          expected=np.array([[[[1, 2], [3, 4]]]], dtype=dtype))
      self._testBinary(
          array_ops.expand_dims,
          np.array([[[1, 2], [3, 4]]], dtype=dtype),
          np.int32(2),
          expected=np.array([[[[1, 2]], [[3, 4]]]], dtype=dtype))
      self._testBinary(
          array_ops.expand_dims,
          np.array([[[1, 2], [3, 4]]], dtype=dtype),
          np.int32(3),
          expected=np.array([[[[1], [2]], [[3], [4]]]], dtype=dtype))

  def testPad(self):
    for dtype in self.numeric_types:
      self._testBinary(
          array_ops.pad,
          np.array(
              [[1, 2, 3], [4, 5, 6]], dtype=dtype),
          np.array(
              [[1, 2], [2, 1]], dtype=np.int32),
          expected=np.array(
              [[0, 0, 0, 0, 0, 0],
               [0, 0, 1, 2, 3, 0],
               [0, 0, 4, 5, 6, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]],
              dtype=dtype))

      self._testBinary(
          lambda x, y: array_ops.pad(x, y, constant_values=7),
          np.array(
              [[1, 2, 3], [4, 5, 6]], dtype=dtype),
          np.array(
              [[0, 3], [2, 1]], dtype=np.int32),
          expected=np.array(
              [[7, 7, 1, 2, 3, 7],
               [7, 7, 4, 5, 6, 7],
               [7, 7, 7, 7, 7, 7],
               [7, 7, 7, 7, 7, 7],
               [7, 7, 7, 7, 7, 7]],
              dtype=dtype))

  def testMirrorPad(self):
    mirror_pad = lambda t, paddings: array_ops.pad(t, paddings, "REFLECT")
    for dtype in self.numeric_types:
      self._testBinary(
          mirror_pad,
          np.array(
              [
                  [1, 2, 3],  #
                  [4, 5, 6],  #
              ],
              dtype=dtype),
          np.array([[
              1,
              1,
          ], [2, 2]], dtype=np.int32),
          expected=np.array(
              [
                  [6, 5, 4, 5, 6, 5, 4],  #
                  [3, 2, 1, 2, 3, 2, 1],  #
                  [6, 5, 4, 5, 6, 5, 4],  #
                  [3, 2, 1, 2, 3, 2, 1]
              ],
              dtype=dtype))
      self._testBinary(
          mirror_pad,
          np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
          np.array([[0, 0], [0, 0]], dtype=np.int32),
          expected=np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype))
      self._testBinary(
          mirror_pad,
          np.array(
              [
                  [1, 2, 3],  #
                  [4, 5, 6],  #
                  [7, 8, 9]
              ],
              dtype=dtype),
          np.array([[2, 2], [0, 0]], dtype=np.int32),
          expected=np.array(
              [
                  [7, 8, 9],  #
                  [4, 5, 6],  #
                  [1, 2, 3],  #
                  [4, 5, 6],  #
                  [7, 8, 9],  #
                  [4, 5, 6],  #
                  [1, 2, 3]
              ],
              dtype=dtype))
      self._testBinary(
          mirror_pad,
          np.array(
              [
                  [[1, 2, 3], [4, 5, 6]],
                  [[7, 8, 9], [10, 11, 12]],
              ], dtype=dtype),
          np.array([[0, 0], [1, 1], [1, 1]], dtype=np.int32),
          expected=np.array(
              [
                  [
                      [5, 4, 5, 6, 5],  #
                      [2, 1, 2, 3, 2],  #
                      [5, 4, 5, 6, 5],  #
                      [2, 1, 2, 3, 2],  #
                  ],
                  [
                      [11, 10, 11, 12, 11],  #
                      [8, 7, 8, 9, 8],  #
                      [11, 10, 11, 12, 11],  #
                      [8, 7, 8, 9, 8],  #
                  ]
              ],
              dtype=dtype))

  def testReshape(self):
    for dtype in self.numeric_types:
      self._testBinary(
          array_ops.reshape,
          np.array([], dtype=dtype),
          np.array([0, 4], dtype=np.int32),
          expected=np.zeros(shape=[0, 4], dtype=dtype))
      self._testBinary(
          array_ops.reshape,
          np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
          np.array([2, 3], dtype=np.int32),
          expected=np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype))
      self._testBinary(
          array_ops.reshape,
          np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
          np.array([3, 2], dtype=np.int32),
          expected=np.array([[0, 1], [2, 3], [4, 5]], dtype=dtype))
      self._testBinary(
          array_ops.reshape,
          np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
          np.array([-1, 6], dtype=np.int32),
          expected=np.array([[0, 1, 2, 3, 4, 5]], dtype=dtype))
      self._testBinary(
          array_ops.reshape,
          np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
          np.array([6, -1], dtype=np.int32),
          expected=np.array([[0], [1], [2], [3], [4], [5]], dtype=dtype))
      self._testBinary(
          array_ops.reshape,
          np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
          np.array([2, -1], dtype=np.int32),
          expected=np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype))
      self._testBinary(
          array_ops.reshape,
          np.array([0, 1, 2, 3, 4, 5], dtype=dtype),
          np.array([-1, 3], dtype=np.int32),
          expected=np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype))

  def testSplit(self):
    for dtype in self.numeric_types:
      for axis in [0, -3]:
        self._testBinary(
            lambda x, y: array_ops.split(value=y, num_or_size_splits=3, axis=x),
            np.int32(axis),
            np.array([[[1], [2]], [[3], [4]], [[5], [6]]],
                     dtype=dtype),
            expected=[
                np.array([[[1], [2]]], dtype=dtype),
                np.array([[[3], [4]]], dtype=dtype),
                np.array([[[5], [6]]], dtype=dtype),
            ],
            equality_test=self.ListsAreClose)

      for axis in [1, -2]:
        self._testBinary(
            lambda x, y: array_ops.split(value=y, num_or_size_splits=2, axis=x),
            np.int32(axis),
            np.array([[[1], [2]], [[3], [4]], [[5], [6]]],
                     dtype=dtype),
            expected=[
                np.array([[[1]], [[3]], [[5]]], dtype=dtype),
                np.array([[[2]], [[4]], [[6]]], dtype=dtype),
            ],
            equality_test=self.ListsAreClose)

  def testTile(self):
    for dtype in self.numeric_types:
      self._testBinary(
          array_ops.tile,
          np.array([[6]], dtype=dtype),
          np.array([1, 2], dtype=np.int32),
          expected=np.array([[6, 6]], dtype=dtype))
      self._testBinary(
          array_ops.tile,
          np.array([[1], [2]], dtype=dtype),
          np.array([1, 2], dtype=np.int32),
          expected=np.array([[1, 1], [2, 2]], dtype=dtype))
      self._testBinary(
          array_ops.tile,
          np.array([[1, 2], [3, 4]], dtype=dtype),
          np.array([3, 2], dtype=np.int32),
          expected=np.array(
              [[1, 2, 1, 2],
               [3, 4, 3, 4],
               [1, 2, 1, 2],
               [3, 4, 3, 4],
               [1, 2, 1, 2],
               [3, 4, 3, 4]],
              dtype=dtype))
      self._testBinary(
          array_ops.tile,
          np.array([[1, 2], [3, 4]], dtype=dtype),
          np.array([1, 1], dtype=np.int32),
          expected=np.array(
              [[1, 2],
               [3, 4]],
              dtype=dtype))
      self._testBinary(
          array_ops.tile,
          np.array([[1, 2]], dtype=dtype),
          np.array([3, 1], dtype=np.int32),
          expected=np.array(
              [[1, 2],
               [1, 2],
               [1, 2]],
              dtype=dtype))

  def testTranspose(self):
    for dtype in self.numeric_types:
      self._testBinary(
          array_ops.transpose,
          np.zeros(shape=[1, 0, 4], dtype=dtype),
          np.array([1, 2, 0], dtype=np.int32),
          expected=np.zeros(shape=[0, 4, 1], dtype=dtype))
      self._testBinary(
          array_ops.transpose,
          np.array([[1, 2], [3, 4]], dtype=dtype),
          np.array([0, 1], dtype=np.int32),
          expected=np.array([[1, 2], [3, 4]], dtype=dtype))
      self._testBinary(
          array_ops.transpose,
          np.array([[1, 2], [3, 4]], dtype=dtype),
          np.array([1, 0], dtype=np.int32),
          expected=np.array([[1, 3], [2, 4]], dtype=dtype))

  def testCross(self):
    for dtype in self.float_types:
      self._testBinary(
          gen_math_ops.cross,
          np.zeros((4, 3), dtype=dtype),
          np.zeros((4, 3), dtype=dtype),
          expected=np.zeros((4, 3), dtype=dtype))
      self._testBinary(
          gen_math_ops.cross,
          np.array([1, 2, 3], dtype=dtype),
          np.array([4, 5, 6], dtype=dtype),
          expected=np.array([-3, 6, -3], dtype=dtype))
      self._testBinary(
          gen_math_ops.cross,
          np.array([[1, 2, 3], [10, 11, 12]], dtype=dtype),
          np.array([[4, 5, 6], [40, 50, 60]], dtype=dtype),
          expected=np.array([[-3, 6, -3], [60, -120, 60]], dtype=dtype))

  def testBroadcastArgs(self):
    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([2, 3, 5], dtype=np.int32),
                     np.array([1], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([1], dtype=np.int32),
                     np.array([2, 3, 5], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([2, 3, 5], dtype=np.int32),
                     np.array([5], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([5], dtype=np.int32),
                     np.array([2, 3, 5], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([2, 3, 5], dtype=np.int32),
                     np.array([3, 5], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([3, 5], dtype=np.int32),
                     np.array([2, 3, 5], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([2, 3, 5], dtype=np.int32),
                     np.array([3, 1], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([3, 1], dtype=np.int32),
                     np.array([2, 3, 5], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([2, 1, 5], dtype=np.int32),
                     np.array([3, 1], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    self._testBinary(array_ops.broadcast_dynamic_shape,
                     np.array([3, 1], dtype=np.int32),
                     np.array([2, 1, 5], dtype=np.int32),
                     expected=np.array([2, 3, 5], dtype=np.int32))

    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "Incompatible shapes"):
      self._testBinary(array_ops.broadcast_dynamic_shape,
                       np.array([1, 2, 3], dtype=np.int32),
                       np.array([4, 5, 6], dtype=np.int32),
                       expected=None)


if __name__ == "__main__":
  googletest.main()

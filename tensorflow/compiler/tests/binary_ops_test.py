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
from tensorflow.python.ops import array_ops
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

  def ListsAreClose(self, result, expected, rtol):
    """Tests closeness of two lists of floats."""
    self.assertEqual(len(result), len(expected))
    for i in range(len(result)):
      self.assertAllClose(result[i], expected[i], rtol)

  def testFloatOps(self):
    for dtype in self.float_types:
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
          gen_nn_ops._softplus_grad,
          np.array([4, 3, 2, 1], dtype=dtype),
          np.array([5, 6, 7, 8], dtype=dtype),
          expected=np.array(
              [3.97322869, 2.99258232, 1.99817801, 0.99966466], dtype=dtype))

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

    self._testBinary(
        gen_math_ops._floor_div,
        np.array([3, 3, -1, -9, -8], dtype=dtype),
        np.array([2, -2, 7, 2, -4], dtype=dtype),
        expected=np.array([1, -2, -1, -5, 2], dtype=dtype))

  def testIntDivision(self):
    for dtype in self.int_types:
      self._testDivision(dtype)

  def testFloatDivision(self):
    for dtype in self.float_types:
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
            [[[[1000, 100], [10, 1]], [[2000, 200], [20, 2]]],
             [[[3000, 300], [30, 3]], [[4000, 400], [40, 4]]]],
            dtype=np.float32),
        np.array(
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[11, 22], [33, 44]],
                                                    [[55, 66], [77, 88]]]],
            dtype=np.float32),
        expected=np.array(
            [[[[1300, 2400], [13, 24]], [[11400, 13600], [114, 136]]],
             [[[42900, 79200], [429, 792]], [[250800, 299200], [2508, 2992]]]],
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
      x = np.arange(0, 3 * 5 * 16 * 7, dtype=np.float32).reshape((3, 5, 16, 7))
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
      self._testBinary(
          lambda x, y: array_ops.split(value=y, num_or_size_splits=3, axis=x),
          np.int32(0),
          np.array([[[1], [2]], [[3], [4]], [[5], [6]]],
                   dtype=dtype),
          expected=[
              np.array([[[1], [2]]], dtype=dtype),
              np.array([[[3], [4]]], dtype=dtype),
              np.array([[[5], [6]]], dtype=dtype),
          ],
          equality_test=self.ListsAreClose)

      self._testBinary(
          lambda x, y: array_ops.split(value=y, num_or_size_splits=2, axis=x),
          np.int32(1),
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


if __name__ == "__main__":
  googletest.main()

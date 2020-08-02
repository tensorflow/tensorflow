# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tridiagonal solve ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import test

_sample_diags = np.array([[2, 1, 4, 0], [1, 3, 2, 2], [0, 1, -1, 1]],
                         dtype=np.float32)
_sample_rhs = np.array([1, 2, 3, 4], dtype=np.float32)
_sample_result = np.array([-9, 5, -4, 4], dtype=np.float32)


def _tfconst(array):
  return constant_op.constant(array, dtype=dtypes.float32)


def _tf_ones(shape):
  return array_ops.ones(shape, dtype=dtypes.float64)


class TridiagonalSolveOpsTest(xla_test.XLATestCase):
  """Test for tri-diagonal matrix related ops."""

  def testTridiagonalSolverSolves1Rhs(self):
    np.random.seed(19)

    batch_size = 8
    num_dims = 11

    diagonals_np = np.random.normal(size=(batch_size, 3,
                                          num_dims)).astype(np.float32)
    rhs_np = np.random.normal(size=(batch_size, num_dims, 1)).astype(np.float32)

    with self.session() as sess, self.test_scope():
      diags = array_ops.placeholder(
          shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
      rhs = array_ops.placeholder(
          shape=(batch_size, num_dims, 1), dtype=dtypes.float32)
      x_np = sess.run(
          linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=False),
          feed_dict={
              diags: diagonals_np,
              rhs: rhs_np
          })[:, :, 0]

    superdiag_np = diagonals_np[:, 0]
    diag_np = diagonals_np[:, 1]
    subdiag_np = diagonals_np[:, 2]

    y = np.zeros((batch_size, num_dims), dtype=np.float32)

    for i in range(num_dims):
      if i == 0:
        y[:, i] = (
            diag_np[:, i] * x_np[:, i] + superdiag_np[:, i] * x_np[:, i + 1])
      elif i == num_dims - 1:
        y[:, i] = (
            subdiag_np[:, i] * x_np[:, i - 1] + diag_np[:, i] * x_np[:, i])
      else:
        y[:, i] = (
            subdiag_np[:, i] * x_np[:, i - 1] + diag_np[:, i] * x_np[:, i] +
            superdiag_np[:, i] * x_np[:, i + 1])

    self.assertAllClose(y, rhs_np[:, :, 0], rtol=1e-4, atol=1e-4)

  def testTridiagonalSolverSolvesKRhs(self):
    np.random.seed(19)

    batch_size = 8
    num_dims = 11
    num_rhs = 5

    diagonals_np = np.random.normal(size=(batch_size, 3,
                                          num_dims)).astype(np.float32)
    rhs_np = np.random.normal(size=(batch_size, num_dims,
                                    num_rhs)).astype(np.float32)

    with self.session() as sess, self.test_scope():
      diags = array_ops.placeholder(
          shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
      rhs = array_ops.placeholder(
          shape=(batch_size, num_dims, num_rhs), dtype=dtypes.float32)
      x_np = sess.run(
          linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=False),
          feed_dict={
              diags: diagonals_np,
              rhs: rhs_np
          })

    superdiag_np = diagonals_np[:, 0]
    diag_np = diagonals_np[:, 1]
    subdiag_np = diagonals_np[:, 2]

    for eq in range(num_rhs):
      y = np.zeros((batch_size, num_dims), dtype=np.float32)
      for i in range(num_dims):
        if i == 0:
          y[:, i] = (
              diag_np[:, i] * x_np[:, i, eq] +
              superdiag_np[:, i] * x_np[:, i + 1, eq])
        elif i == num_dims - 1:
          y[:, i] = (
              subdiag_np[:, i] * x_np[:, i - 1, eq] +
              diag_np[:, i] * x_np[:, i, eq])
        else:
          y[:, i] = (
              subdiag_np[:, i] * x_np[:, i - 1, eq] +
              diag_np[:, i] * x_np[:, i, eq] +
              superdiag_np[:, i] * x_np[:, i + 1, eq])

      self.assertAllClose(y, rhs_np[:, :, eq], rtol=1e-4, atol=1e-4)

  # All the following is adapted from tridiagonal_solve_op_test.py
  def _test(self,
            diags,
            rhs,
            expected,
            diags_format="compact",
            transpose_rhs=False):
    with self.session() as sess, self.test_scope():
      self.assertAllClose(
          sess.run(
              linalg_impl.tridiagonal_solve(
                  _tfconst(diags),
                  _tfconst(rhs),
                  diags_format,
                  transpose_rhs,
                  conjugate_rhs=False,
                  partial_pivoting=False)),
          np.asarray(expected, dtype=np.float32))

  def _testWithDiagonalLists(self,
                             diags,
                             rhs,
                             expected,
                             diags_format="compact",
                             transpose_rhs=False):
    with self.session() as sess, self.test_scope():
      self.assertAllClose(
          sess.run(
              linalg_impl.tridiagonal_solve([_tfconst(x) for x in diags],
                                            _tfconst(rhs),
                                            diags_format,
                                            transpose_rhs,
                                            conjugate_rhs=False,
                                            partial_pivoting=False)),
          sess.run(_tfconst(expected)))

  def testReal(self):
    self._test(diags=_sample_diags, rhs=_sample_rhs, expected=_sample_result)

  # testComplex is skipped as complex type is not yet supported.

  def test3x3(self):
    self._test(
        diags=[[2.0, -1.0, 0.0], [1.0, 3.0, 1.0], [0.0, -1.0, -2.0]],
        rhs=[1.0, 2.0, 3.0],
        expected=[-3.0, 2.0, 7.0])

  def test2x2(self):
    self._test(
        diags=[[2.0, 0.0], [1.0, 3.0], [0.0, 1.0]],
        rhs=[1.0, 4.0],
        expected=[-5.0, 3.0])

  def test1x1(self):
    self._test(diags=[[0], [3], [0]], rhs=[6], expected=[2])

  def test0x0(self):
    self._test(
        diags=np.zeros(shape=(3, 0), dtype=np.float32),
        rhs=np.zeros(shape=(0, 1), dtype=np.float32),
        expected=np.zeros(shape=(0, 1), dtype=np.float32))

  def test2x2WithMultipleRhs(self):
    self._test(
        diags=[[2, 0], [1, 3], [0, 1]],
        rhs=[[1, 2, 3], [4, 8, 12]],
        expected=[[-5, -10, -15], [3, 6, 9]])

  def test1x1WithMultipleRhs(self):
    self._test(diags=[[0], [3], [0]], rhs=[[6, 9, 12]], expected=[[2, 3, 4]])

  # test1x1NotInvertible is skipped as runtime error not raised for now.

  # test2x2NotInvertible is skipped as runtime error not raised for now.

  def testPartialPivotingRaises(self):
    np.random.seed(0)
    batch_size = 8
    num_dims = 11
    num_rhs = 5

    diagonals_np = np.random.normal(size=(batch_size, 3,
                                          num_dims)).astype(np.float32)
    rhs_np = np.random.normal(size=(batch_size, num_dims,
                                    num_rhs)).astype(np.float32)

    with self.session() as sess, self.test_scope():
      with self.assertRaisesRegex(
          errors_impl.UnimplementedError,
          "Current implementation does not yet support pivoting."):
        diags = array_ops.placeholder(
            shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
        rhs = array_ops.placeholder(
            shape=(batch_size, num_dims, num_rhs), dtype=dtypes.float32)
        sess.run(
            linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=True),
            feed_dict={
                diags: diagonals_np,
                rhs: rhs_np
            })

  # testCaseRequiringPivotingLastRows is skipped as pivoting is not supported
  # for now.

  # testNotInvertible is skipped as runtime error not raised for now.

  def testDiagonal(self):
    self._test(
        diags=[[0, 0, 0, 0], [1, 2, -1, -2], [0, 0, 0, 0]],
        rhs=[1, 2, 3, 4],
        expected=[1, 1, -3, -2])

  def testUpperTriangular(self):
    self._test(
        diags=[[2, 4, -1, 0], [1, 3, 1, 2], [0, 0, 0, 0]],
        rhs=[1, 6, 4, 4],
        expected=[13, -6, 6, 2])

  def testLowerTriangular(self):
    self._test(
        diags=[[0, 0, 0, 0], [2, -1, 3, 1], [0, 1, 4, 2]],
        rhs=[4, 5, 6, 1],
        expected=[2, -3, 6, -11])

  def testWithTwoRightHandSides(self):
    self._test(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs, 2 * _sample_rhs]),
        expected=np.transpose([_sample_result, 2 * _sample_result]))

  def testBatching(self):
    self._test(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, -2 * _sample_result]))

  def testWithTwoBatchingDimensions(self):
    self._test(
        diags=np.array([[_sample_diags, -_sample_diags, _sample_diags],
                        [-_sample_diags, _sample_diags, -_sample_diags]]),
        rhs=np.array([[_sample_rhs, 2 * _sample_rhs, 3 * _sample_rhs],
                      [4 * _sample_rhs, 5 * _sample_rhs, 6 * _sample_rhs]]),
        expected=np.array(
            [[_sample_result, -2 * _sample_result, 3 * _sample_result],
             [-4 * _sample_result, 5 * _sample_result, -6 * _sample_result]]))

  def testBatchingAndTwoRightHandSides(self):
    rhs = np.transpose([_sample_rhs, 2 * _sample_rhs])
    expected_result = np.transpose([_sample_result, 2 * _sample_result])
    self._test(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([rhs, 2 * rhs]),
        expected=np.array([expected_result, -2 * expected_result]))

  def testSequenceFormat(self):
    self._testWithDiagonalLists(
        diags=[[2, 1, 4], [1, 3, 2, 2], [1, -1, 1]],
        rhs=[1, 2, 3, 4],
        expected=[-9, 5, -4, 4],
        diags_format="sequence")

  def testSequenceFormatWithDummyElements(self):
    dummy = 20  # Should be ignored by the solver.
    self._testWithDiagonalLists(
        diags=[
            [2, 1, 4, dummy],
            [1, 3, 2, 2],
            [dummy, 1, -1, 1],
        ],
        rhs=[1, 2, 3, 4],
        expected=[-9, 5, -4, 4],
        diags_format="sequence")

  def testSequenceFormatWithBatching(self):
    self._testWithDiagonalLists(
        diags=[[[2, 1, 4], [-2, -1, -4]], [[1, 3, 2, 2], [-1, -3, -2, -2]],
               [[1, -1, 1], [-1, 1, -1]]],
        rhs=[[1, 2, 3, 4], [1, 2, 3, 4]],
        expected=[[-9, 5, -4, 4], [9, -5, 4, -4]],
        diags_format="sequence")

  def testMatrixFormat(self):
    self._test(
        diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
        rhs=[1, 2, 3, 4],
        expected=[-9, 5, -4, 4],
        diags_format="matrix")

  def testMatrixFormatWithMultipleRightHandSides(self):
    self._test(
        diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
        rhs=[[1, -1], [2, -2], [3, -3], [4, -4]],
        expected=[[-9, 9], [5, -5], [-4, 4], [4, -4]],
        diags_format="matrix")

  def testMatrixFormatWithBatching(self):
    self._test(
        diags=[[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
               [[-1, -2, 0, 0], [-1, -3, -1, 0], [0, 1, -2, -4], [0, 0, -1,
                                                                  -2]]],
        rhs=[[1, 2, 3, 4], [1, 2, 3, 4]],
        expected=[[-9, 5, -4, 4], [9, -5, 4, -4]],
        diags_format="matrix")

  def testRightHandSideAsColumn(self):
    self._test(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs]),
        expected=np.transpose([_sample_result]),
        diags_format="compact")

  def testTransposeRhs(self):
    self._test(
        diags=_sample_diags,
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, 2 * _sample_result]).T,
        transpose_rhs=True)

  # testConjugateRhs is skipped as complex type is not yet supported.

  # testAjointRhs is skipped as complex type is not yet supported

  def testTransposeRhsWithRhsAsVector(self):
    self._test(
        diags=_sample_diags,
        rhs=_sample_rhs,
        expected=_sample_result,
        transpose_rhs=True)

  # testConjugateRhsWithRhsAsVector is skipped as complex type is not yet
  # supported.

  def testTransposeRhsWithRhsAsVectorAndBatching(self):
    self._test(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, -2 * _sample_result]),
        transpose_rhs=True)

  # Gradient tests

  def _gradientTest(
      self,
      diags,
      rhs,
      y,  # output = reduce_sum(y * tridiag_solve(diags, rhs))
      expected_grad_diags,  # expected gradient of output w.r.t. diags
      expected_grad_rhs,  # expected gradient of output w.r.t. rhs
      diags_format="compact",
      transpose_rhs=False,
      feed_dict=None):
    expected_grad_diags = np.array(expected_grad_diags).astype(np.float32)
    expected_grad_rhs = np.array(expected_grad_rhs).astype(np.float32)
    with self.session() as sess, self.test_scope():
      diags = _tfconst(diags)
      rhs = _tfconst(rhs)
      y = _tfconst(y)

      x = linalg_impl.tridiagonal_solve(
          diags,
          rhs,
          diagonals_format=diags_format,
          transpose_rhs=transpose_rhs,
          conjugate_rhs=False,
          partial_pivoting=False)

      res = math_ops.reduce_sum(x * y)
      actual_grad_diags = sess.run(
          gradient_ops.gradients(res, diags)[0], feed_dict=feed_dict)
      actual_rhs_diags = sess.run(
          gradient_ops.gradients(res, rhs)[0], feed_dict=feed_dict)
    self.assertAllClose(expected_grad_diags, actual_grad_diags)
    self.assertAllClose(expected_grad_rhs, actual_rhs_diags)

  def testGradientSimple(self):
    self._gradientTest(
        diags=_sample_diags,
        rhs=_sample_rhs,
        y=[1, 3, 2, 4],
        expected_grad_diags=[[-5, 0, 4, 0], [9, 0, -4, -16], [0, 0, 5, 16]],
        expected_grad_rhs=[1, 0, -1, 4])

  def testGradientWithMultipleRhs(self):
    self._gradientTest(
        diags=_sample_diags,
        rhs=[[1, 2], [2, 4], [3, 6], [4, 8]],
        y=[[1, 5], [2, 6], [3, 7], [4, 8]],
        expected_grad_diags=[[-20, 28, -60, 0], [36, -35, 60, 80],
                             [0, 63, -75, -80]],
        expected_grad_rhs=[[0, 2], [1, 3], [1, 7], [0, -10]])

  def _makeDataForGradientWithBatching(self):
    y = np.array([1, 3, 2, 4]).astype(np.float32)
    grad_diags = np.array([[-5, 0, 4, 0], [9, 0, -4, -16],
                           [0, 0, 5, 16]]).astype(np.float32)
    grad_rhs = np.array([1, 0, -1, 4]).astype(np.float32)

    diags_batched = np.array(
        [[_sample_diags, 2 * _sample_diags, 3 * _sample_diags],
         [4 * _sample_diags, 5 * _sample_diags,
          6 * _sample_diags]]).astype(np.float32)
    rhs_batched = np.array([[_sample_rhs, -_sample_rhs, _sample_rhs],
                            [-_sample_rhs, _sample_rhs,
                             -_sample_rhs]]).astype(np.float32)
    y_batched = np.array([[y, y, y], [y, y, y]]).astype(np.float32)
    expected_grad_diags_batched = np.array(
        [[grad_diags, -grad_diags / 4, grad_diags / 9],
         [-grad_diags / 16, grad_diags / 25,
          -grad_diags / 36]]).astype(np.float32)
    expected_grad_rhs_batched = np.array(
        [[grad_rhs, grad_rhs / 2, grad_rhs / 3],
         [grad_rhs / 4, grad_rhs / 5, grad_rhs / 6]]).astype(np.float32)

    return (y_batched, diags_batched, rhs_batched, expected_grad_diags_batched,
            expected_grad_rhs_batched)

  def testGradientWithBatchDims(self):
    y, diags, rhs, expected_grad_diags, expected_grad_rhs = (
        self._makeDataForGradientWithBatching())

    self._gradientTest(
        diags=diags,
        rhs=rhs,
        y=y,
        expected_grad_diags=expected_grad_diags,
        expected_grad_rhs=expected_grad_rhs)

  # testGradientWithUnknownShapes is skipped as shapes should be fully known.

  def _assertRaises(self, diags, rhs, diags_format="compact"):
    with self.assertRaises(ValueError):
      linalg_impl.tridiagonal_solve(diags, rhs, diags_format)

  # Invalid input shapes
  def testInvalidShapesCompactFormat(self):

    def test_raises(diags_shape, rhs_shape):
      self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), "compact")

    test_raises((5, 4, 4), (5, 4))
    test_raises((5, 3, 4), (4, 5))
    test_raises((5, 3, 4), (5))
    test_raises((5), (5, 4))

  def testInvalidShapesSequenceFormat(self):

    def test_raises(diags_tuple_shapes, rhs_shape):
      diagonals = tuple(_tf_ones(shape) for shape in diags_tuple_shapes)
      self._assertRaises(diagonals, _tf_ones(rhs_shape), "sequence")

    test_raises(((5, 4), (5, 4)), (5, 4))
    test_raises(((5, 4), (5, 4), (5, 6)), (5, 4))
    test_raises(((5, 3), (5, 4), (5, 6)), (5, 4))
    test_raises(((5, 6), (5, 4), (5, 3)), (5, 4))
    test_raises(((5, 4), (7, 4), (5, 4)), (5, 4))
    test_raises(((5, 4), (7, 4), (5, 4)), (3, 4))

  def testInvalidShapesMatrixFormat(self):

    def test_raises(diags_shape, rhs_shape):
      self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), "matrix")

    test_raises((5, 4, 7), (5, 4))
    test_raises((5, 4, 4), (3, 4))
    test_raises((5, 4, 4), (5, 3))

  # Tests involving placeholder with an unknown dimension are all skipped.
  # Dimensions have to be all known statically.


if __name__ == "__main__":
  test.main()

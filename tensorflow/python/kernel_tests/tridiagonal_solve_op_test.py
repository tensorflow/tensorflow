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
"""Tests for tensorflow.ops.linalg.linalg_impl.tridiagonal_solve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.client import session
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

_sample_diags = np.array([[2, 1, 4, 0], [1, 3, 2, 2], [0, 1, -1, 1]])
_sample_rhs = np.array([1, 2, 3, 4])
_sample_result = np.array([-9, 5, -4, 4])

FORWARD_COMPATIBLE_DATE = (2019, 10, 18)

# Flag, indicating that test should be run only with partial_pivoting=True
FLAG_REQUIRES_PIVOTING = "FLAG_REQUIRES_PIVOT"

# Flag, indicating that test shouldn't be parameterized by different values of
# partial_pivoting, etc.
FLAG_NO_PARAMETERIZATION = "FLAG_NO_PARAMETERIZATION"


def flags(*args):

  def decorator(f):
    for flag in args:
      setattr(f, flag, True)
    return f

  return decorator


def _tfconst(array):
  return constant_op.constant(array, dtypes.float64)


def _tf_ones(shape):
  return array_ops.ones(shape, dtype=dtypes.float64)


class TridiagonalSolveOpTest(test.TestCase):

  def _test(self,
            diags,
            rhs,
            expected,
            diags_format="compact",
            transpose_rhs=False,
            conjugate_rhs=False):
    with self.cached_session(use_gpu=True):
      result = linalg_impl.tridiagonal_solve(diags, rhs, diags_format,
                                             transpose_rhs, conjugate_rhs)
      self.assertAllClose(self.evaluate(result), expected)

  def _testWithLists(self,
                     diags,
                     rhs,
                     expected,
                     diags_format="compact",
                     transpose_rhs=False,
                     conjugate_rhs=False):
    self._test(
        _tfconst(diags), _tfconst(rhs), _tfconst(expected), diags_format,
        transpose_rhs, conjugate_rhs)

  def _assertRaises(self, diags, rhs, diags_format="compact"):
    with self.assertRaises(ValueError):
      linalg_impl.tridiagonal_solve(diags, rhs, diags_format)

  # Tests with various dtypes

  def testReal(self):
    for dtype in dtypes.float32, dtypes.float64:
      self._test(
          diags=constant_op.constant(_sample_diags, dtype),
          rhs=constant_op.constant(_sample_rhs, dtype),
          expected=constant_op.constant(_sample_result, dtype))

  def testComplex(self):
    for dtype in dtypes.complex64, dtypes.complex128:
      self._test(
          diags=constant_op.constant(_sample_diags, dtype) * (1 + 1j),
          rhs=constant_op.constant(_sample_rhs, dtype) * (1 - 1j),
          expected=constant_op.constant(_sample_result, dtype) * (1 - 1j) /
          (1 + 1j))

  # Tests with small matrix sizes

  def test3x3(self):
    self._testWithLists(
        diags=[[2, -1, 0], [1, 3, 1], [0, -1, -2]],
        rhs=[1, 2, 3],
        expected=[-3, 2, 7])

  def test2x2(self):
    self._testWithLists(
        diags=[[2, 0], [1, 3], [0, 1]], rhs=[1, 4], expected=[-5, 3])

  def test2x2Complex(self):
    for dtype in dtypes.complex64, dtypes.complex128:
      self._test(
          diags=constant_op.constant([[2j, 0j], [1j, 3j], [0j, 1j]], dtype),
          rhs=constant_op.constant([1 - 1j, 4 - 4j], dtype),
          expected=constant_op.constant([5 + 5j, -3 - 3j], dtype))

  def test1x1(self):
    self._testWithLists(diags=[[0], [3], [0]], rhs=[6], expected=[2])

  def test0x0(self):
    self._test(
        diags=constant_op.constant(0, shape=(3, 0), dtype=dtypes.float32),
        rhs=constant_op.constant(0, shape=(0, 1), dtype=dtypes.float32),
        expected=constant_op.constant(0, shape=(0, 1), dtype=dtypes.float32))

  def test2x2WithMultipleRhs(self):
    self._testWithLists(
        diags=[[2, 0], [1, 3], [0, 1]],
        rhs=[[1, 2, 3], [4, 8, 12]],
        expected=[[-5, -10, -15], [3, 6, 9]])

  def test1x1WithMultipleRhs(self):
    self._testWithLists(
        diags=[[0], [3], [0]], rhs=[[6, 9, 12]], expected=[[2, 3, 4]])

  def test1x1NotInvertible(self):
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self._testWithLists(diags=[[0], [0], [0]], rhs=[[6, 9, 12]], expected=[])

  def test2x2NotInvertible(self):
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self._testWithLists(
          diags=[[3, 0], [1, 3], [0, 1]], rhs=[1, 4], expected=[])

  # Other edge cases

  @flags(FLAG_REQUIRES_PIVOTING)
  def testCaseRequiringPivoting(self):
    # Without partial pivoting (e.g. Thomas algorithm) this would fail.
    self._testWithLists(
        diags=[[2, -1, 1, 0], [1, 4, 1, -1], [0, 2, -2, 3]],
        rhs=[1, 2, 3, 4],
        expected=[8, -3.5, 0, -4])

  @flags(FLAG_REQUIRES_PIVOTING)
  def testCaseRequiringPivotingLastRows(self):
    self._testWithLists(
        diags=[[2, 1, -1, 0], [1, -1, 2, 1], [0, 1, -6, 1]],
        rhs=[1, 2, -1, -2],
        expected=[5, -2, -5, 3])

  def testNotInvertible(self):
    if test.is_gpu_available(cuda_only=True):
      # CuSparse gtsv routines don't raise errors for non-invertible
      # matrices.
      return
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self._testWithLists(
          diags=[[2, -1, 1, 0], [1, 4, 1, -1], [0, 2, 0, 3]],
          rhs=[1, 2, 3, 4],
          expected=[8, -3.5, 0, -4])

  def testDiagonal(self):
    self._testWithLists(
        diags=[[0, 0, 0, 0], [1, 2, -1, -2], [0, 0, 0, 0]],
        rhs=[1, 2, 3, 4],
        expected=[1, 1, -3, -2])

  def testUpperTriangular(self):
    self._testWithLists(
        diags=[[2, 4, -1, 0], [1, 3, 1, 2], [0, 0, 0, 0]],
        rhs=[1, 6, 4, 4],
        expected=[13, -6, 6, 2])

  def testLowerTriangular(self):
    self._testWithLists(
        diags=[[0, 0, 0, 0], [2, -1, 3, 1], [0, 1, 4, 2]],
        rhs=[4, 5, 6, 1],
        expected=[2, -3, 6, -11])

  # Multiple right-hand sides and batching

  def testWithTwoRightHandSides(self):
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs, 2 * _sample_rhs]),
        expected=np.transpose([_sample_result, 2 * _sample_result]))

  def testBatching(self):
    self._testWithLists(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, -2 * _sample_result]))

  def testWithTwoBatchingDimensions(self):
    self._testWithLists(
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
    self._testWithLists(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([rhs, 2 * rhs]),
        expected=np.array([expected_result, -2 * expected_result]))

  # Various input formats

  def testSequenceFormat(self):
    self._test(
        diags=(_tfconst([2, 1, 4]), _tfconst([1, 3, 2, 2]), _tfconst([1, -1,
                                                                      1])),
        rhs=_tfconst([1, 2, 3, 4]),
        expected=_tfconst([-9, 5, -4, 4]),
        diags_format="sequence")

  def testSequenceFormatWithDummyElements(self):
    dummy = 20
    self._test(
        diags=(_tfconst([2, 1, 4, dummy]), _tfconst([1, 3, 2, 2]),
               _tfconst([dummy, 1, -1, 1])),
        rhs=_tfconst([1, 2, 3, 4]),
        expected=_tfconst([-9, 5, -4, 4]),
        diags_format="sequence")

  def testSequenceFormatWithBatching(self):
    self._test(
        diags=(_tfconst([[2, 1, 4], [-2, -1, -4]]),
               _tfconst([[1, 3, 2, 2], [-1, -3, -2, -2]]),
               _tfconst([[1, -1, 1], [-1, 1, -1]])),
        rhs=_tfconst([[1, 2, 3, 4], [1, 2, 3, 4]]),
        expected=_tfconst([[-9, 5, -4, 4], [9, -5, 4, -4]]),
        diags_format="sequence")

  def testMatrixFormat(self):
    self._testWithLists(
        diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
        rhs=[1, 2, 3, 4],
        expected=[-9, 5, -4, 4],
        diags_format="matrix")

  def testMatrixFormatWithMultipleRightHandSides(self):
    self._testWithLists(
        diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
        rhs=[[1, -1], [2, -2], [3, -3], [4, -4]],
        expected=[[-9, 9], [5, -5], [-4, 4], [4, -4]],
        diags_format="matrix")

  def testMatrixFormatWithBatching(self):
    self._testWithLists(
        diags=[[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
               [[-1, -2, 0, 0], [-1, -3, -1, 0], [0, 1, -2, -4], [0, 0, -1,
                                                                  -2]]],
        rhs=[[1, 2, 3, 4], [1, 2, 3, 4]],
        expected=[[-9, 5, -4, 4], [9, -5, 4, -4]],
        diags_format="matrix")

  def testRightHandSideAsColumn(self):
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs]),
        expected=np.transpose([_sample_result]),
        diags_format="compact")

  # Tests with transpose and adjoint

  def testTransposeRhs(self):
    expected = np.array([_sample_result, 2 * _sample_result])
    if compat.forward_compatible(*FORWARD_COMPATIBLE_DATE):
      expected = expected.T
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=expected,
        transpose_rhs=True)

  def testConjugateRhs(self):
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs * (1 + 1j), _sample_rhs * (1 - 2j)]),
        expected=np.transpose(
            [_sample_result * (1 - 1j), _sample_result * (1 + 2j)]),
        conjugate_rhs=True)

  def testAdjointRhs(self):
    expected = np.array(
        [_sample_result * (1 - 1j), _sample_result * (1 + 2j)])
    if compat.forward_compatible(*FORWARD_COMPATIBLE_DATE):
      expected = expected.T
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.array([_sample_rhs * (1 + 1j), _sample_rhs * (1 - 2j)]),
        expected=expected,
        transpose_rhs=True,
        conjugate_rhs=True)

  def testTransposeRhsWithBatching(self):
    expected = np.array(
        [[_sample_result, 2 * _sample_result],
         [-3 * _sample_result, -4 * _sample_result]])
    if compat.forward_compatible(*FORWARD_COMPATIBLE_DATE):
      expected = expected.transpose(0, 2, 1)
    self._testWithLists(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([[_sample_rhs, 2 * _sample_rhs],
                      [3 * _sample_rhs, 4 * _sample_rhs]]),
        expected=expected,
        transpose_rhs=True)

  def testTransposeRhsWithRhsAsVector(self):
    self._testWithLists(
        diags=_sample_diags,
        rhs=_sample_rhs,
        expected=_sample_result,
        transpose_rhs=True)

  def testConjugateRhsWithRhsAsVector(self):
    self._testWithLists(
        diags=_sample_diags,
        rhs=_sample_rhs * (1 + 1j),
        expected=_sample_result * (1 - 1j),
        conjugate_rhs=True)

  def testTransposeRhsWithRhsAsVectorAndBatching(self):
    self._testWithLists(
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
      conjugate_rhs=False,
      feed_dict=None):
    expected_grad_diags = _tfconst(expected_grad_diags)
    expected_grad_rhs = _tfconst(expected_grad_rhs)
    with backprop.GradientTape() as tape_diags:
      with backprop.GradientTape() as tape_rhs:
        tape_diags.watch(diags)
        tape_rhs.watch(rhs)
        x = linalg_impl.tridiagonal_solve(
            diags,
            rhs,
            diagonals_format=diags_format,
            transpose_rhs=transpose_rhs,
            conjugate_rhs=conjugate_rhs)
        res = math_ops.reduce_sum(x * y)
    with self.cached_session(use_gpu=True) as sess:
      actual_grad_diags = sess.run(
          tape_diags.gradient(res, diags), feed_dict=feed_dict)
      actual_rhs_diags = sess.run(
          tape_rhs.gradient(res, rhs), feed_dict=feed_dict)
    self.assertAllClose(expected_grad_diags, actual_grad_diags)
    self.assertAllClose(expected_grad_rhs, actual_rhs_diags)

  def _gradientTestWithLists(self,
                             diags,
                             rhs,
                             y,
                             expected_grad_diags,
                             expected_grad_rhs,
                             diags_format="compact",
                             transpose_rhs=False,
                             conjugate_rhs=False):
    self._gradientTest(
        _tfconst(diags), _tfconst(rhs), _tfconst(y), expected_grad_diags,
        expected_grad_rhs, diags_format, transpose_rhs, conjugate_rhs)

  def testGradientSimple(self):
    self._gradientTestWithLists(
        diags=_sample_diags,
        rhs=_sample_rhs,
        y=[1, 3, 2, 4],
        expected_grad_diags=[[-5, 0, 4, 0], [9, 0, -4, -16], [0, 0, 5, 16]],
        expected_grad_rhs=[1, 0, -1, 4])

  def testGradientWithMultipleRhs(self):
    self._gradientTestWithLists(
        diags=_sample_diags,
        rhs=[[1, 2], [2, 4], [3, 6], [4, 8]],
        y=[[1, 5], [2, 6], [3, 7], [4, 8]],
        expected_grad_diags=([[-20, 28, -60, 0], [36, -35, 60, 80],
                              [0, 63, -75, -80]]),
        expected_grad_rhs=[[0, 2], [1, 3], [1, 7], [0, -10]])

  def _makeDataForGradientWithBatching(self):
    y = np.array([1, 3, 2, 4])
    grad_diags = np.array([[-5, 0, 4, 0], [9, 0, -4, -16], [0, 0, 5, 16]])
    grad_rhs = np.array([1, 0, -1, 4])

    diags_batched = np.array(
        [[_sample_diags, 2 * _sample_diags, 3 * _sample_diags],
         [4 * _sample_diags, 5 * _sample_diags, 6 * _sample_diags]])
    rhs_batched = np.array([[_sample_rhs, -_sample_rhs, _sample_rhs],
                            [-_sample_rhs, _sample_rhs, -_sample_rhs]])
    y_batched = np.array([[y, y, y], [y, y, y]])
    expected_grad_diags_batched = np.array(
        [[grad_diags, -grad_diags / 4, grad_diags / 9],
         [-grad_diags / 16, grad_diags / 25, -grad_diags / 36]])
    expected_grad_rhs_batched = np.array(
        [[grad_rhs, grad_rhs / 2, grad_rhs / 3],
         [grad_rhs / 4, grad_rhs / 5, grad_rhs / 6]])

    return (y_batched, diags_batched, rhs_batched, expected_grad_diags_batched,
            expected_grad_rhs_batched)

  def testGradientWithBatchDims(self):
    y, diags, rhs, expected_grad_diags, expected_grad_rhs = \
      self._makeDataForGradientWithBatching()

    self._gradientTestWithLists(
        diags=diags,
        rhs=rhs,
        y=y,
        expected_grad_diags=expected_grad_diags,
        expected_grad_rhs=expected_grad_rhs)

  @test_util.run_deprecated_v1
  def testGradientWithUnknownShapes(self):

    def placeholder(rank):
      return array_ops.placeholder(
          dtypes.float64, shape=(None for _ in range(rank)))

    y, diags, rhs, expected_grad_diags, expected_grad_rhs = \
      self._makeDataForGradientWithBatching()

    diags_placeholder = placeholder(rank=4)
    rhs_placeholder = placeholder(rank=3)
    y_placeholder = placeholder(rank=3)

    self._gradientTest(
        diags=diags_placeholder,
        rhs=rhs_placeholder,
        y=y_placeholder,
        expected_grad_diags=expected_grad_diags,
        expected_grad_rhs=expected_grad_rhs,
        feed_dict={
            diags_placeholder: diags,
            rhs_placeholder: rhs,
            y_placeholder: y
        })

  # Invalid input shapes

  @flags(FLAG_NO_PARAMETERIZATION)
  def testInvalidShapesCompactFormat(self):

    def test_raises(diags_shape, rhs_shape):
      self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), "compact")

    test_raises((5, 4, 4), (5, 4))
    test_raises((5, 3, 4), (4, 5))
    test_raises((5, 3, 4), (5))
    test_raises((5), (5, 4))

  @flags(FLAG_NO_PARAMETERIZATION)
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

  @flags(FLAG_NO_PARAMETERIZATION)
  def testInvalidShapesMatrixFormat(self):

    def test_raises(diags_shape, rhs_shape):
      self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), "matrix")

    test_raises((5, 4, 7), (5, 4))
    test_raises((5, 4, 4), (3, 4))
    test_raises((5, 4, 4), (5, 3))

  # Tests with placeholders

  def _testWithPlaceholders(self,
                            diags_shape,
                            rhs_shape,
                            diags_feed,
                            rhs_feed,
                            expected,
                            diags_format="compact"):
    if context.executing_eagerly():
      return
    diags = array_ops.placeholder(dtypes.float64, shape=diags_shape)
    rhs = array_ops.placeholder(dtypes.float64, shape=rhs_shape)
    x = linalg_impl.tridiagonal_solve(
        diags, rhs, diags_format, partial_pivoting=self.pivoting)
    with self.cached_session(use_gpu=True) as sess:
      result = sess.run(x, feed_dict={diags: diags_feed, rhs: rhs_feed})
      self.assertAllClose(result, expected)

  @test_util.run_deprecated_v1
  def testCompactFormatAllDimsUnknown(self):
    self._testWithPlaceholders(
        diags_shape=[None, None],
        rhs_shape=[None],
        diags_feed=_sample_diags,
        rhs_feed=_sample_rhs,
        expected=_sample_result)

  @test_util.run_deprecated_v1
  def testCompactFormatUnknownMatrixSize(self):
    self._testWithPlaceholders(
        diags_shape=[3, None],
        rhs_shape=[4],
        diags_feed=_sample_diags,
        rhs_feed=_sample_rhs,
        expected=_sample_result)

  @test_util.run_deprecated_v1
  def testCompactFormatUnknownRhsCount(self):
    self._testWithPlaceholders(
        diags_shape=[3, 4],
        rhs_shape=[4, None],
        diags_feed=_sample_diags,
        rhs_feed=np.transpose([_sample_rhs, 2 * _sample_rhs]),
        expected=np.transpose([_sample_result, 2 * _sample_result]))

  @test_util.run_deprecated_v1
  def testCompactFormatUnknownBatchSize(self):
    self._testWithPlaceholders(
        diags_shape=[None, 3, 4],
        rhs_shape=[None, 4],
        diags_feed=np.array([_sample_diags, -_sample_diags]),
        rhs_feed=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, -2 * _sample_result]))

  @test_util.run_deprecated_v1
  def testMatrixFormatWithUnknownDims(self):
    if context.executing_eagerly():
      return

    def test_with_matrix_shapes(matrix_shape, rhs_shape=None):
      matrix = np.array([[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4],
                         [0, 0, 1, 2]])
      rhs = np.array([1, 2, 3, 4])
      x = np.array([-9, 5, -4, 4])
      self._testWithPlaceholders(
          diags_shape=matrix_shape,
          rhs_shape=rhs_shape,
          diags_feed=matrix,
          rhs_feed=np.transpose([rhs, 2 * rhs]),
          expected=np.transpose([x, 2 * x]),
          diags_format="matrix")

    test_with_matrix_shapes(matrix_shape=[4, 4], rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=[None, 4], rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=[4, None], rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=[None, None], rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=[4, 4])
    test_with_matrix_shapes(matrix_shape=[None, 4])
    test_with_matrix_shapes(matrix_shape=[4, None])
    test_with_matrix_shapes(matrix_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=None, rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=None)

  @test_util.run_deprecated_v1
  def testSequenceFormatWithUnknownDims(self):
    if context.executing_eagerly():
      return
    superdiag = array_ops.placeholder(dtypes.float64, shape=[None])
    diag = array_ops.placeholder(dtypes.float64, shape=[None])
    subdiag = array_ops.placeholder(dtypes.float64, shape=[None])
    rhs = array_ops.placeholder(dtypes.float64, shape=[None])

    x = linalg_impl.tridiagonal_solve((superdiag, diag, subdiag),
                                      rhs,
                                      diagonals_format="sequence",
                                      partial_pivoting=self.pivoting)
    with self.cached_session(use_gpu=True) as sess:
      result = sess.run(
          x,
          feed_dict={
              subdiag: [20, 1, -1, 1],
              diag: [1, 3, 2, 2],
              superdiag: [2, 1, 4, 20],
              rhs: [1, 2, 3, 4]
          })
      self.assertAllClose(result, [-9, 5, -4, 4])

  # Benchmark

  class TridiagonalSolveBenchmark(test.Benchmark):
    sizes = [(100000, 1, 1), (1000000, 1, 1), (10000000, 1, 1), (100000, 10, 1),
             (100000, 100, 1), (10000, 1, 10), (10000, 1, 100)]

    pivoting_options = [(True, "pivoting"), (False, "no_pivoting")]

    def _generateData(self, matrix_size, batch_size, num_rhs, seed=42):
      np.random.seed(seed)
      data = np.random.normal(size=(batch_size, matrix_size, 3 + num_rhs))
      diags = np.stack([data[:, :, 0], data[:, :, 1], data[:, :, 2]], axis=-2)
      rhs = data[:, :, 3:]
      return (variables.Variable(diags, dtype=dtypes.float64),
              variables.Variable(rhs, dtype=dtypes.float64))

    def _generateMatrixData(self, matrix_size, batch_size, num_rhs, seed=42):
      np.random.seed(seed)
      import scipy.sparse as sparse  # pylint:disable=g-import-not-at-top
      # By being strictly diagonally dominant, we guarantee invertibility.d
      diag = 2* np.abs(np.random.randn(matrix_size)) + 4.1
      subdiag = 2* np.abs(np.random.randn(matrix_size-1))
      superdiag = 2* np.abs(np.random.randn(matrix_size-1))
      matrix = sparse.diags([superdiag, diag, subdiag], [1, 0, -1]).toarray()
      vector = np.random.randn(batch_size, matrix_size, num_rhs)
      return (variables.Variable(np.tile(matrix, (batch_size, 1, 1))),
              variables.Variable(vector))

    def _benchmark(self, generate_data_fn, test_name_format_string):
      devices = [("/cpu:0", "cpu")]
      if test.is_gpu_available(cuda_only=True):
        devices += [("/gpu:0", "gpu")]

      for device_option, pivoting_option, size_option in \
          itertools.product(devices, self.pivoting_options, self.sizes):

        device_id, device_name = device_option
        pivoting, pivoting_name = pivoting_option
        matrix_size, batch_size, num_rhs = size_option

        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device(device_id):
          diags, rhs = generate_data_fn(matrix_size, batch_size, num_rhs)
          x = linalg_impl.tridiagonal_solve(
              diags, rhs, partial_pivoting=pivoting)
          variables.global_variables_initializer().run()
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(x),
              min_iters=10,
              store_memory_usage=False,
              name=test_name_format_string.format(
                  device_name, matrix_size, batch_size, num_rhs,
                  pivoting_name))

    def benchmarkTridiagonalSolveOp_WithMatrixInput(self):
      self._benchmark(
          self._generateMatrixData,
          test_name_format_string=(
              "tridiagonal_solve_matrix_format_{}_matrix_size_{}_"
              "batch_size_{}_num_rhs_{}_{}"))

    def benchmarkTridiagonalSolveOp(self):
      self._benchmark(
          self._generateMatrixData,
          test_name_format_string=(
              "tridiagonal_solve_{}_matrix_size_{}_"
              "batch_size_{}_num_rhs_{}_{}"))


if __name__ == "__main__":
  for name, fun in dict(TridiagonalSolveOpTest.__dict__).items():
    if not name.startswith("test"):
      continue
    if hasattr(fun, FLAG_NO_PARAMETERIZATION):
      continue

    # Replace testFoo with testFoo_pivoting and testFoo_noPivoting, setting
    # self.pivoting to corresponding value.
    delattr(TridiagonalSolveOpTest, name)

    def decor(test_fun, pivoting):

      def wrapped(instance):
        instance.pivoting = pivoting
        test_fun(instance)

      return wrapped

    setattr(TridiagonalSolveOpTest, name + "_pivoting",
            decor(fun, pivoting=True))
    if not hasattr(fun, FLAG_REQUIRES_PIVOTING):
      setattr(TridiagonalSolveOpTest, name + "_noPivoting",
              decor(fun, pivoting=False))

  test.main()

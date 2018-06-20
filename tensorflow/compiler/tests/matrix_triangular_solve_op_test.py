# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.MatrixTriangularSolve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def MakePlaceholder(x):
  return array_ops.placeholder(dtypes.as_dtype(x.dtype), shape=x.shape)


class MatrixTriangularSolveOpTest(XLATestCase):

  #  MatrixTriangularSolve defined for float64, float32, complex64, complex128
  # (https://www.tensorflow.org/api_docs/python/tf/matrix_triangular_solve)
  @property
  def float_types(self):
    return set(super(MatrixTriangularSolveOpTest,
                     self).float_types).intersection(
                         (np.float64, np.float32, np.complex64, np.complex128))

  def _VerifyTriangularSolveBase(self, sess, placeholder_a, placeholder_ca,
                                 placeholder_b, a, clean_a, b, verification,
                                 atol):
    feed_dict = {placeholder_a: a, placeholder_ca: clean_a, placeholder_b: b}
    verification_np = sess.run(verification, feed_dict)
    self.assertAllClose(b, verification_np, atol=atol)

  def _VerifyTriangularSolve(self, a, b, lower, adjoint, atol):
    clean_a = np.tril(a) if lower else np.triu(a)
    with self.test_session() as sess:
      placeholder_a = MakePlaceholder(a)
      placeholder_ca = MakePlaceholder(clean_a)
      placeholder_b = MakePlaceholder(b)
      with self.test_scope():
        x = linalg_ops.matrix_triangular_solve(
            placeholder_a, placeholder_b, lower=lower, adjoint=adjoint)
      verification = math_ops.matmul(placeholder_ca, x, adjoint_a=adjoint)
      self._VerifyTriangularSolveBase(sess, placeholder_a, placeholder_ca,
                                      placeholder_b, a, clean_a, b,
                                      verification, atol)

  def _VerifyTriangularSolveCombo(self, a, b, atol=1e-4):
    transp = lambda x: np.swapaxes(x, -1, -2)
    for lower, adjoint in itertools.product([True, False], repeat=2):
      self._VerifyTriangularSolve(
          a if lower else transp(a), b, lower, adjoint, atol)

  def testBasic(self):
    rng = np.random.RandomState(0)
    a = np.tril(rng.randn(5, 5))
    b = rng.randn(5, 7)
    for dtype in self.float_types:
      self._VerifyTriangularSolveCombo(a.astype(dtype), b.astype(dtype))

  def testBasicNotActuallyTriangular(self):
    rng = np.random.RandomState(0)
    a = rng.randn(5, 5)  # the `a` matrix is not lower-triangular
    b = rng.randn(5, 7)
    for dtype in self.float_types:
      self._VerifyTriangularSolveCombo(a.astype(dtype), b.astype(dtype))

  def testBasicComplexDtypes(self):
    rng = np.random.RandomState(0)
    a = np.tril(rng.randn(5, 5) + rng.randn(5, 5) * 1j)
    b = rng.randn(5, 7) + rng.randn(5, 7) * 1j
    for dtype in self.complex_types:
      self._VerifyTriangularSolveCombo(a.astype(dtype), b.astype(dtype))

  def testBatch(self):
    rng = np.random.RandomState(0)
    shapes = [((4, 3, 3), (4, 3, 5)), ((1, 2, 2), (1, 2, 1)),
              ((1, 1, 1), (1, 1, 2)), ((2, 3, 4, 4), (2, 3, 4, 1))]
    tuples = itertools.product(self.float_types, shapes)
    for dtype, (a_shape, b_shape) in tuples:
      n = a_shape[-1]
      a = np.tril(rng.rand(*a_shape) - 0.5) / (2.0 * n) + np.eye(n)
      b = rng.randn(*b_shape)
      self._VerifyTriangularSolveCombo(
          a.astype(dtype), b.astype(dtype), atol=1e-3)

  def testLarge(self):
    n = 1024
    rng = np.random.RandomState(0)
    a = np.tril(rng.rand(n, n) - 0.5) / (2.0 * n) + np.eye(n)
    b = rng.randn(n, n)
    self._VerifyTriangularSolve(
        a.astype(np.float32), b.astype(np.float32), True, False, 1e-4)

  def testNonSquareCoefficientMatrix(self):
    rng = np.random.RandomState(0)
    for dtype in self.float_types:
      a = rng.randn(3, 4).astype(dtype)
      b = rng.randn(4, 4).astype(dtype)
      with self.assertRaises(ValueError):
        linalg_ops.matrix_triangular_solve(a, b)
      with self.assertRaises(ValueError):
        linalg_ops.matrix_triangular_solve(a, b)

  def testWrongDimensions(self):
    randn = np.random.RandomState(0).randn
    for dtype in self.float_types:
      lhs = constant_op.constant(randn(3, 3), dtype=dtype)
      rhs = constant_op.constant(randn(4, 3), dtype=dtype)
      with self.assertRaises(ValueError):
        linalg_ops.matrix_triangular_solve(lhs, rhs)
      with self.assertRaises(ValueError):
        linalg_ops.matrix_triangular_solve(lhs, rhs)


if __name__ == "__main__":
  test.main()

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
"""Utilities for testing `OperatorPDBase` and related classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)  # pylint: disable=no-init
class OperatorPDDerivedClassTest(tf.test.TestCase):
  """Tests for derived classes.

  Subclasses should implement every abstractmethod, and this will enable all
  test methods to work.
  """

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _compare_results(
      self, expected, actual, static_shapes=True, atol=1e-5):
    """Compare expected value (array) to the actual value (Tensor)."""
    if static_shapes:
      self.assertEqual(expected.shape, actual.get_shape())
    self.assertAllClose(expected, actual.eval(), atol=atol)

  @abc.abstractmethod
  def _build_operator_and_mat(self, batch_shape, k, dtype=np.float64):
    """Build a batch matrix and an Operator that should have similar behavior.

    Every operator represents a (batch) matrix.  This method returns both
    together, and is used e.g. by tests.

    Args:
      batch_shape:  List-like of Python integers giving batch shape of operator.
      k: Python integer, the event size.
      dtype:  Numpy dtype.  Data type of returned array/operator.

    Returns:
      operator:  `OperatorPDBase` subclass.
      mat:  numpy array representing a (batch) matrix.
    """
    # Create a matrix as a numpy array.  Shape = batch_shape + [k, k].
    # Create an OperatorPDDiag that should have the same behavior as the matrix.
    # All arguments are convertable to numpy arrays.
    #
    batch_shape = list(batch_shape)
    mat_shape = batch_shape + [k, k]
    # return operator, mat
    raise NotImplementedError("Not implemented yet.")

  def testToDense(self):
    with self.test_session():
      for batch_shape in [(), (2, 3,)]:
        for k in [1, 4]:
          for dtype in [np.float32, np.float64]:
            operator, mat = self._build_operator_and_mat(
                batch_shape, k, dtype=dtype)
            self._compare_results(
                expected=mat,
                actual=operator.to_dense())

  def testSqrtToDense(self):
    with self.test_session():
      for batch_shape in [(), (2, 3,)]:
        for k in [1, 4]:
          operator, mat = self._build_operator_and_mat(batch_shape, k)
          sqrt = operator.sqrt_to_dense()
          self.assertEqual(mat.shape, sqrt.get_shape())
          # Square roots are not unique, but SS^T should equal mat.  In this
          # case however, we should have S = S^T.
          self._compare_results(expected=mat, actual=tf.matmul(sqrt, sqrt))

  def testDeterminants(self):
    with self.test_session():
      for batch_shape in [(), (2, 3,)]:
        for k in [1, 4]:
          operator, mat = self._build_operator_and_mat(batch_shape, k)
          expected_det = tf.matrix_determinant(mat).eval()

          self._compare_results(expected_det, operator.det())
          self._compare_results(np.log(expected_det), operator.log_det())

  def testMatmul(self):
    with self.test_session():
      for batch_shape in [(), (2, 3,)]:
        for k in [1, 4]:
          operator, mat = self._build_operator_and_mat(batch_shape, k)

          # Work with 5 simultaneous systems.  5 is arbitrary.
          x = self._rng.randn(*(batch_shape + (k, 5)))

          self._compare_results(
              expected=tf.matmul(mat, x).eval(), actual=operator.matmul(x))

  def testSqrtMatmul(self):
    # Square roots are not unique, but we should have SS^T x = Ax, and in our
    # case, we should have S = S^T, so SSx = Ax.
    with self.test_session():
      for batch_shape in [(), (2, 3,)]:
        for k in [1, 4]:
          operator, mat = self._build_operator_and_mat(batch_shape, k)

          # Work with 5 simultaneous systems.  5 is arbitrary.
          x = self._rng.randn(*(batch_shape + (k, 5)))

          self._compare_results(
              expected=tf.matmul(mat, x).eval(),
              actual=operator.sqrt_matmul(operator.sqrt_matmul(x)))

  def testSolve(self):
    with self.test_session():
      for batch_shape in [(), (2, 3,)]:
        for k in [1, 4]:
          operator, mat = self._build_operator_and_mat(batch_shape, k)

          # Work with 5 simultaneous systems.  5 is arbitrary.
          x = self._rng.randn(*(batch_shape + (k, 5)))

          self._compare_results(
              expected=tf.matrix_solve(mat, x).eval(), actual=operator.solve(x))

  def testSqrtSolve(self):
    # Square roots are not unique, but we should still have
    # S^{-T} S^{-1} x = A^{-1} x.
    # In our case, we should have S = S^T, so then S^{-1} S^{-1} x = A^{-1} x.
    with self.test_session():
      for batch_shape in [(), (2, 3,)]:
        for k in [1, 4]:
          operator, mat = self._build_operator_and_mat(batch_shape, k)

          # Work with 5 simultaneous systems.  5 is arbitrary.
          x = self._rng.randn(*(batch_shape + (k, 5)))

          self._compare_results(
              expected=tf.matrix_solve(mat, x).eval(),
              actual=operator.sqrt_solve(operator.sqrt_solve(x)))

  def testAddToTensor(self):
    with self.test_session():
      for batch_shape in [(), (2, 3,)]:
        for k in [1, 4]:
          operator, mat = self._build_operator_and_mat(batch_shape, k)
          tensor = tf.ones_like(mat)

          self._compare_results(
              expected=(mat + tensor).eval(),
              actual=operator.add_to_tensor(tensor))

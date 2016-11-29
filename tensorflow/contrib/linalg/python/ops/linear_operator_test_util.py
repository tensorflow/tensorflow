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
"""Utilities for testing `LinearOperator` and sub-classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)  # pylint: disable=no-init
class LinearOperatorDerivedClassTest(tf.test.TestCase):
  """Tests for derived classes.

  Subclasses should implement every abstractmethod, and this will enable all
  test methods to work.
  """

  @abc.abstractproperty
  def _dtypes_to_test(self):
    """Returns list of numpy or tensorflow dtypes.  Each will be tested."""
    raise NotImplementedError("dtypes_to_test has not been implemented.")

  @abc.abstractproperty
  def _shapes_to_test(self):
    """Returns list of tuples, each is one shape that will be tested."""
    raise NotImplementedError("shapes_to_test has not been implemented.")

  @abc.abstractmethod
  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    """Build a batch matrix and an Operator that should have similar behavior.

    Every operator acts like a (batch) matrix.  This method returns both
    together, and is used by tests.

    Args:
      shape:  List-like of Python integers giving full shape of operator.
      dtype:  Numpy dtype.  Data type of returned array/operator.
      use_placeholder:  Python bool.  If True, initialize the operator with a
        placeholder of undefined shape and correct dtype.

    Returns:
      operator:  `LinearOperator` subclass instance.
      mat:  `Tensor` representing operator.
      feed_dict:  Dictionary.  If placholder is True, this must be fed to
        sess.run calls at runtime to make the operator work.
    """
    # Create a matrix as a numpy array with desired shape/dtype.
    # Create a LinearOperator that should have the same behavior as the matrix.
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def _make_rhs(self, operator):
    """Make a rhs appropriate for calling operator.solve(rhs)."""
    raise NotImplementedError("_make_rhs is not defined.")

  @abc.abstractmethod
  def _make_x(self, operator):
    """Make a rhs appropriate for calling operator.apply(rhs)."""
    raise NotImplementedError("_make_x is not defined.")

  def test_to_dense(self):
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          operator, mat, _ = self._operator_and_mat_and_feed_dict(
              shape, dtype, use_placeholder=False)
          op_dense = operator.to_dense()
          self.assertAllEqual(shape, op_dense.get_shape())
          op_dense_v, mat_v = sess.run([op_dense, mat])
          self.assertAllClose(op_dense_v, mat_v)

  def test_to_dense_dynamic(self):
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
              shape, dtype, use_placeholder=True)
          op_dense_v, mat_v = sess.run(
              [operator.to_dense(), mat], feed_dict=feed_dict)
          self.assertAllClose(op_dense_v, mat_v)

  def test_det(self):
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          operator, mat, _ = self._operator_and_mat_and_feed_dict(
              shape, dtype, use_placeholder=False)
          op_det = operator.determinant()
          self.assertAllEqual(shape[:-2], op_det.get_shape())
          op_det_v, mat_det_v = sess.run([op_det, tf.matrix_determinant(mat)])
          self.assertAllClose(op_det_v, mat_det_v)

  def test_det_dynamic(self):
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
              shape, dtype, use_placeholder=True)
          op_det_v, mat_det_v = sess.run(
              [operator.determinant(), tf.matrix_determinant(mat)],
              feed_dict=feed_dict)
          self.assertAllClose(op_det_v, mat_det_v)

  def test_apply(self):
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          operator, mat, _ = self._operator_and_mat_and_feed_dict(
              shape, dtype, use_placeholder=False)
          for adjoint in [False, True]:
            if adjoint and operator.is_self_adjoint:
              continue
            x = self._make_x(operator)
            op_apply = operator.apply(x, adjoint=adjoint)
            mat_apply = tf.matmul(mat, x, adjoint_a=adjoint)
            self.assertAllEqual(op_apply.get_shape(), mat_apply.get_shape())
            op_apply_v, mat_apply_v = sess.run([op_apply, mat_apply])
            self.assertAllClose(op_apply_v, mat_apply_v)

  def test_apply_dynamic(self):
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
              shape, dtype, use_placeholder=True)
          x = self._make_x(operator)
          op_apply_v, mat_apply_v = sess.run(
              [operator.apply(x), tf.matmul(mat, x)],
              feed_dict=feed_dict)
          self.assertAllClose(op_apply_v, mat_apply_v)

  def test_solve(self):
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          operator, mat, _ = self._operator_and_mat_and_feed_dict(
              shape, dtype, use_placeholder=False)
          for adjoint in [False, True]:
            if adjoint and operator.is_self_adjoint:
              continue
            rhs = self._make_rhs(operator)
            op_solve = operator.solve(rhs, adjoint=adjoint)
            mat_solve = tf.matrix_solve(mat, rhs, adjoint=adjoint)
            self.assertAllEqual(op_solve.get_shape(), mat_solve.get_shape())
            op_solve_v, mat_solve_v = sess.run([op_solve, mat_solve])
            self.assertAllClose(op_solve_v, mat_solve_v)

  def test_solve_dynamic(self):
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
              shape, dtype, use_placeholder=True)
          rhs = self._make_rhs(operator)
          op_solve_v, mat_solve_v = sess.run(
              [operator.solve(rhs), tf.matrix_solve(mat, rhs)],
              feed_dict=feed_dict)
          self.assertAllClose(op_solve_v, mat_solve_v)

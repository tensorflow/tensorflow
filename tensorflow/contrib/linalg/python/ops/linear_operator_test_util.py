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

    # Absolute/relative tolerance for tests.
  _atol = {
      tf.float16: 1e-3, tf.float32: 1e-6, tf.float64: 1e-12, tf.complex64: 1e-6,
      tf.complex128: 1e-12}
  _rtol = {
      tf.float16: 1e-3, tf.float32: 1e-6, tf.float64: 1e-12, tf.complex64: 1e-6,
      tf.complex128: 1e-12}

  def assertAC(self, x, y):
    """Derived classes can set _atol, _rtol to get different tolerance."""
    dtype = tf.as_dtype(x.dtype)
    atol = self._atol[dtype]
    rtol = self._rtol[dtype]
    self.assertAllClose(x, y, atol=atol, rtol=rtol)

  @property
  def _dtypes_to_test(self):
    # TODO(langmore) Test tf.float16 once tf.matrix_diag works in 16bit.
    return [tf.float32, tf.float64, tf.complex64, tf.complex128]

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
      feed_dict:  Dictionary.
        If placholder is True, this must contains everything needed to be fed
          to sess.run calls at runtime to make the operator work.
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

  @property
  def _tests_to_skip(self):
    """List of test names to skip."""
    # Subclasses should over-ride if they want to skip some tests.
    # To skip "test_foo", add "foo" to this list.
    return []

  def _maybe_skip(self, test_name):
    if test_name in self._tests_to_skip:
      self.skipTest("%s skipped because it was added to self._tests_to_skip.")

  def test_to_dense(self):
    self._maybe_skip("to_dense")
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          for use_placeholder in False, True:
            operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                shape, dtype, use_placeholder=use_placeholder)
            op_dense = operator.to_dense()
            if not use_placeholder:
              self.assertAllEqual(shape, op_dense.get_shape())
            op_dense_v, mat_v = sess.run([op_dense, mat], feed_dict=feed_dict)
            self.assertAC(op_dense_v, mat_v)

  def test_det(self):
    self._maybe_skip("det")
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          if dtype.is_complex:
            self.skipTest(
                "tf.matrix_determinant does not work with complex, so this test"
                " is being skipped.")
          for use_placeholder in False, True:
            operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                shape, dtype, use_placeholder=use_placeholder)
            op_det = operator.determinant()
            if not use_placeholder:
              self.assertAllEqual(shape[:-2], op_det.get_shape())
            op_det_v, mat_det_v = sess.run(
                [op_det, tf.matrix_determinant(mat)], feed_dict=feed_dict)
            self.assertAC(op_det_v, mat_det_v)

  def test_apply(self):
    self._maybe_skip("apply")
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          for use_placeholder in False, True:
            for adjoint in [False, True]:
              operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                  shape, dtype, use_placeholder=use_placeholder)
              x = self._make_x(operator)
              op_apply = operator.apply(x, adjoint=adjoint)
              mat_apply = tf.matmul(mat, x, adjoint_a=adjoint)
              if not use_placeholder:
                self.assertAllEqual(op_apply.get_shape(), mat_apply.get_shape())
              op_apply_v, mat_apply_v = sess.run(
                  [op_apply, mat_apply], feed_dict=feed_dict)
              self.assertAC(op_apply_v, mat_apply_v)

  def test_solve(self):
    self._maybe_skip("solve")
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          for use_placeholder in False, True:
            for adjoint in [False, True]:
              operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                  shape, dtype, use_placeholder=use_placeholder)
              rhs = self._make_rhs(operator)
              op_solve = operator.solve(rhs, adjoint=adjoint)
              mat_solve = tf.matrix_solve(mat, rhs, adjoint=adjoint)
              if not use_placeholder:
                self.assertAllEqual(op_solve.get_shape(), mat_solve.get_shape())
              op_solve_v, mat_solve_v = sess.run(
                  [op_solve, mat_solve], feed_dict=feed_dict)
              self.assertAC(op_solve_v, mat_solve_v)

  def test_add_to_tensor(self):
    self._maybe_skip("add_to_tensor")
    with self.test_session() as sess:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          for use_placeholder in False, True:
            operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                shape, dtype, use_placeholder=use_placeholder)
            op_plus_2mat = operator.add_to_tensor(2 * mat)

            if not use_placeholder:
              self.assertAllEqual(shape, op_plus_2mat.get_shape())

            op_plus_2mat_v, mat_v = sess.run(
                [op_plus_2mat, mat], feed_dict=feed_dict)

            self.assertAC(op_plus_2mat_v, 3 * mat_v)


@six.add_metaclass(abc.ABCMeta)
class SquareLinearOperatorDerivedClassTest(LinearOperatorDerivedClassTest):
  """Base test class appropriate for square operators.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

  @property
  def _shapes_to_test(self):
    # non-batch operators (n, n) and batch operators.
    return [(0, 0), (1, 1), (1, 3, 3), (3, 4, 4), (2, 1, 4, 4)]

  def _make_rhs(self, operator):
    # This operator is square, so rhs and x will have same shape.
    return self._make_x(operator)

  def _make_x(self, operator):
    # Return the number of systems to solve, R, equal to 1 or 2.
    r = self._get_num_systems(operator)
    # If operator.shape = [B1,...,Bb, N, N] this returns a random matrix of
    # shape [B1,...,Bb, N, R], R = 1 or 2.
    if operator.shape.is_fully_defined():
      batch_shape = operator.batch_shape.as_list()
      n = operator.domain_dimension.value
      rhs_shape = batch_shape + [n, r]
    else:
      batch_shape = operator.batch_shape_dynamic()
      n = operator.domain_dimension_dynamic()
      rhs_shape = tf.concat(0, (batch_shape, [n, r]))

    x = tf.random_normal(shape=rhs_shape, dtype=operator.dtype.real_dtype)
    if operator.dtype.is_complex:
      x = tf.complex(
          x, tf.random_normal(shape=rhs_shape, dtype=operator.dtype.real_dtype))
    return x

  def _get_num_systems(self, operator):
    """Get some number, either 1 or 2, depending on operator."""
    if operator.tensor_rank is None or operator.tensor_rank % 2:
      return 1
    else:
      return 2

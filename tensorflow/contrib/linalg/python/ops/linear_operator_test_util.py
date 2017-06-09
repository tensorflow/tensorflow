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
import numpy as np
import six

from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.contrib.linalg.python.ops import linear_operator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


@six.add_metaclass(abc.ABCMeta)  # pylint: disable=no-init
class LinearOperatorDerivedClassTest(test.TestCase):
  """Tests for derived classes.

  Subclasses should implement every abstractmethod, and this will enable all
  test methods to work.
  """

  # Absolute/relative tolerance for tests.
  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-12,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-12
  }
  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-12,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-12
  }

  def assertAC(self, x, y):
    """Derived classes can set _atol, _rtol to get different tolerance."""
    dtype = dtypes.as_dtype(x.dtype)
    atol = self._atol[dtype]
    rtol = self._rtol[dtype]
    self.assertAllClose(x, y, atol=atol, rtol=rtol)

  @property
  def _dtypes_to_test(self):
    # TODO(langmore) Test tf.float16 once tf.matrix_solve works in 16bit.
    return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

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
  def _make_rhs(self, operator, adjoint):
    """Make a rhs appropriate for calling operator.solve(rhs).

    Args:
      operator:  A `LinearOperator`
      adjoint:  Python `bool`.  If `True`, we are making a 'rhs' value for the
        adjoint operator.

    Returns:
      A `Tensor`
    """
    raise NotImplementedError("_make_rhs is not defined.")

  @abc.abstractmethod
  def _make_x(self, operator, adjoint):
    """Make an 'x' appropriate for calling operator.matmul(x).

    Args:
      operator:  A `LinearOperator`
      adjoint:  Python `bool`.  If `True`, we are making an 'x' value for the
        adjoint operator.

    Returns:
      A `Tensor`
    """
    raise NotImplementedError("_make_x is not defined.")

  @property
  def _tests_to_skip(self):
    """List of test names to skip."""
    # Subclasses should over-ride if they want to skip some tests.
    # To skip "test_foo", add "foo" to this list.
    return []

  def _skip_if_tests_to_skip_contains(self, test_name):
    """If self._tests_to_skip contains test_name, raise SkipTest exception.

    See tests below for usage.

    Args:
      test_name:  String name corresponding to a test.

    Raises:
      SkipTest Exception, if test_name is in self._tests_to_skip.
    """
    if test_name in self._tests_to_skip:
      self.skipTest("%s skipped because it was added to self._tests_to_skip.")

  def test_to_dense(self):
    self._skip_if_tests_to_skip_contains("to_dense")
    for use_placeholder in False, True:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          with self.test_session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                shape, dtype, use_placeholder=use_placeholder)
            op_dense = operator.to_dense()
            if not use_placeholder:
              self.assertAllEqual(shape, op_dense.get_shape())
            op_dense_v, mat_v = sess.run([op_dense, mat], feed_dict=feed_dict)
            self.assertAC(op_dense_v, mat_v)

  def test_det(self):
    self._skip_if_tests_to_skip_contains("det")
    for use_placeholder in False, True:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          if dtype.is_complex:
            self.skipTest(
                "tf.matrix_determinant does not work with complex, so this "
                "test is being skipped.")
          with self.test_session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                shape, dtype, use_placeholder=use_placeholder)
            op_det = operator.determinant()
            if not use_placeholder:
              self.assertAllEqual(shape[:-2], op_det.get_shape())
            op_det_v, mat_det_v = sess.run(
                [op_det, linalg_ops.matrix_determinant(mat)],
                feed_dict=feed_dict)
            self.assertAC(op_det_v, mat_det_v)

  def test_log_abs_det(self):
    self._skip_if_tests_to_skip_contains("log_abs_det")
    for use_placeholder in False, True:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          if dtype.is_complex:
            self.skipTest(
                "tf.matrix_determinant does not work with complex, so this "
                "test is being skipped.")
          with self.test_session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                shape, dtype, use_placeholder=use_placeholder)
            op_log_abs_det = operator.log_abs_determinant()
            mat_log_abs_det = math_ops.log(
                math_ops.abs(linalg_ops.matrix_determinant(mat)))
            if not use_placeholder:
              self.assertAllEqual(shape[:-2], op_log_abs_det.get_shape())
            op_log_abs_det_v, mat_log_abs_det_v = sess.run(
                [op_log_abs_det, mat_log_abs_det],
                feed_dict=feed_dict)
            self.assertAC(op_log_abs_det_v, mat_log_abs_det_v)

  def test_matmul(self):
    self._skip_if_tests_to_skip_contains("matmul")
    for use_placeholder in False, True:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          for adjoint in False, True:
            for adjoint_arg in False, True:
              with self.test_session(graph=ops.Graph()) as sess:
                sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
                operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                    shape, dtype, use_placeholder=use_placeholder)
                x = self._make_x(operator, adjoint=adjoint)
                # If adjoint_arg, compute A X^H^H = A X.
                if adjoint_arg:
                  op_matmul = operator.matmul(
                      linear_operator_util.matrix_adjoint(x),
                      adjoint=adjoint, adjoint_arg=adjoint_arg)
                else:
                  op_matmul = operator.matmul(x, adjoint=adjoint)
                mat_matmul = math_ops.matmul(mat, x, adjoint_a=adjoint)
                if not use_placeholder:
                  self.assertAllEqual(
                      op_matmul.get_shape(), mat_matmul.get_shape())
                op_matmul_v, mat_matmul_v = sess.run(
                    [op_matmul, mat_matmul], feed_dict=feed_dict)
                self.assertAC(op_matmul_v, mat_matmul_v)

  def test_solve(self):
    self._skip_if_tests_to_skip_contains("solve")
    for use_placeholder in False, True:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          for adjoint in False, True:
            for adjoint_arg in False, True:
              with self.test_session(graph=ops.Graph()) as sess:
                sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
                operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                    shape, dtype, use_placeholder=use_placeholder)
                rhs = self._make_rhs(operator, adjoint=adjoint)
                # If adjoint_arg, solve A X = (rhs^H)^H = rhs.
                if adjoint_arg:
                  op_solve = operator.solve(
                      linear_operator_util.matrix_adjoint(rhs),
                      adjoint=adjoint, adjoint_arg=adjoint_arg)
                else:
                  op_solve = operator.solve(
                      rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
                mat_solve = linalg_ops.matrix_solve(mat, rhs, adjoint=adjoint)
                if not use_placeholder:
                  self.assertAllEqual(
                      op_solve.get_shape(), mat_solve.get_shape())
                op_solve_v, mat_solve_v = sess.run([op_solve, mat_solve],
                                                   feed_dict=feed_dict)
                self.assertAC(op_solve_v, mat_solve_v)

  def test_add_to_tensor(self):
    self._skip_if_tests_to_skip_contains("add_to_tensor")
    for use_placeholder in False, True:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          with self.test_session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                shape, dtype, use_placeholder=use_placeholder)
            op_plus_2mat = operator.add_to_tensor(2 * mat)

            if not use_placeholder:
              self.assertAllEqual(shape, op_plus_2mat.get_shape())

            op_plus_2mat_v, mat_v = sess.run([op_plus_2mat, mat],
                                             feed_dict=feed_dict)

            self.assertAC(op_plus_2mat_v, 3 * mat_v)

  def test_diag_part(self):
    self._skip_if_tests_to_skip_contains("diag_part")
    for use_placeholder in False, True:
      for shape in self._shapes_to_test:
        for dtype in self._dtypes_to_test:
          with self.test_session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            operator, mat, feed_dict = self._operator_and_mat_and_feed_dict(
                shape, dtype, use_placeholder=use_placeholder)
            op_diag_part = operator.diag_part()
            mat_diag_part = array_ops.matrix_diag_part(mat)

            if not use_placeholder:
              self.assertAllEqual(
                  mat_diag_part.get_shape(), op_diag_part.get_shape())

            op_diag_part_, mat_diag_part_ = sess.run(
                [op_diag_part, mat_diag_part], feed_dict=feed_dict)

            self.assertAC(op_diag_part_, mat_diag_part_)


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

  def _make_rhs(self, operator, adjoint):
    # This operator is square, so rhs and x will have same shape.
    # adjoint value makes no difference because the operator shape doesn't
    # change since it is square, but be pedantic.
    return self._make_x(operator, adjoint=not adjoint)

  def _make_x(self, operator, adjoint):
    # Value of adjoint makes no difference because the operator is square.
    # Return the number of systems to solve, R, equal to 1 or 2.
    r = self._get_num_systems(operator)
    # If operator.shape = [B1,...,Bb, N, N] this returns a random matrix of
    # shape [B1,...,Bb, N, R], R = 1 or 2.
    if operator.shape.is_fully_defined():
      batch_shape = operator.batch_shape.as_list()
      n = operator.domain_dimension.value
      x_shape = batch_shape + [n, r]
    else:
      batch_shape = operator.batch_shape_tensor()
      n = operator.domain_dimension_tensor()
      x_shape = array_ops.concat((batch_shape, [n, r]), 0)

    return random_normal(x_shape, dtype=operator.dtype)

  def _get_num_systems(self, operator):
    """Get some number, either 1 or 2, depending on operator."""
    if operator.tensor_rank is None or operator.tensor_rank % 2:
      return 1
    else:
      return 2


@six.add_metaclass(abc.ABCMeta)
class NonSquareLinearOperatorDerivedClassTest(LinearOperatorDerivedClassTest):
  """Base test class appropriate for generic rectangular operators.

  Square shapes are never tested by this class, so if you want to test your
  operator with a square shape, create two test classes, the other subclassing
  SquareLinearOperatorFullMatrixTest.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

  @property
  def _tests_to_skip(self):
    """List of test names to skip."""
    return ["solve", "det", "log_abs_det"]

  @property
  def _shapes_to_test(self):
    # non-batch operators (n, n) and batch operators.
    return [(2, 1), (1, 2), (1, 3, 2), (3, 3, 4), (2, 1, 2, 4)]

  def _make_rhs(self, operator, adjoint):
    # TODO(langmore) Add once we're testing solve_ls.
    raise NotImplementedError(
        "_make_rhs not implemented because we don't test solve")

  def _make_x(self, operator, adjoint):
    # Return the number of systems for the argument 'x' for .matmul(x)
    r = self._get_num_systems(operator)
    # If operator.shape = [B1,...,Bb, M, N] this returns a random matrix of
    # shape [B1,...,Bb, N, R], R = 1 or 2.
    if operator.shape.is_fully_defined():
      batch_shape = operator.batch_shape.as_list()
      if adjoint:
        n = operator.range_dimension.value
      else:
        n = operator.domain_dimension.value
      x_shape = batch_shape + [n, r]
    else:
      batch_shape = operator.batch_shape_tensor()
      if adjoint:
        n = operator.range_dimension_tensor()
      else:
        n = operator.domain_dimension_tensor()
      x_shape = array_ops.concat((batch_shape, [n, r]), 0)

    return random_normal(x_shape, dtype=operator.dtype)

  def _get_num_systems(self, operator):
    """Get some number, either 1 or 2, depending on operator."""
    if operator.tensor_rank is None or operator.tensor_rank % 2:
      return 1
    else:
      return 2


def random_positive_definite_matrix(shape, dtype, force_well_conditioned=False):
  """[batch] positive definite matrix.

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned matrix.
    dtype:  `TensorFlow` `dtype` or Python dtype.
    force_well_conditioned:  Python bool.  If `True`, returned matrix has
      eigenvalues with modulus in `(1, 4)`.  Otherwise, eigenvalues are
      chi-squared random variables.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)
  if not contrib_tensor_util.is_tensor(shape):
    shape = tensor_shape.TensorShape(shape)
    # Matrix must be square.
    shape[-1].assert_is_compatible_with(shape[-2])

  with ops.name_scope("random_positive_definite_matrix"):
    tril = random_tril_matrix(
        shape, dtype, force_well_conditioned=force_well_conditioned)
    return math_ops.matmul(tril, tril, adjoint_b=True)


def random_tril_matrix(shape,
                       dtype,
                       force_well_conditioned=False,
                       remove_upper=True):
  """[batch] lower triangular matrix.

  Args:
    shape:  `TensorShape` or Python `list`.  Shape of the returned matrix.
    dtype:  `TensorFlow` `dtype` or Python dtype
    force_well_conditioned:  Python `bool`. If `True`, returned matrix will have
      eigenvalues with modulus in `(1, 2)`.  Otherwise, eigenvalues are unit
      normal random variables.
    remove_upper:  Python `bool`.
      If `True`, zero out the strictly upper triangle.
      If `False`, the lower triangle of returned matrix will have desired
      properties, but will not not have the strictly upper triangle zero'd out.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  with ops.name_scope("random_tril_matrix"):
    # Totally random matrix.  Has no nice properties.
    tril = random_normal(shape, dtype=dtype)
    if remove_upper:
      tril = array_ops.matrix_band_part(tril, -1, 0)

    # Create a diagonal with entries having modulus in [1, 2].
    if force_well_conditioned:
      maxval = ops.convert_to_tensor(np.sqrt(2.), dtype=dtype.real_dtype)
      diag = random_sign_uniform(
          shape[:-1], dtype=dtype, minval=1., maxval=maxval)
      tril = array_ops.matrix_set_diag(tril, diag)

    return tril


def random_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None):
  """Tensor with (possibly complex) Gaussian entries.

  Samples are distributed like

  ```
  N(mean, stddev^2), if dtype is real,
  X + iY,  where X, Y ~ N(mean, stddev^2) if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    mean:  `Tensor` giving mean of normal to sample from.
    stddev:  `Tensor` giving stdev of normal to sample from.
    dtype:  `TensorFlow` `dtype` or numpy dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_normal"):
    samples = random_ops.random_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype.real_dtype, seed=seed)
    if dtype.is_complex:
      if seed is not None:
        seed += 1234
      more_samples = random_ops.random_normal(
          shape, mean=mean, stddev=stddev, dtype=dtype.real_dtype, seed=seed)
      samples = math_ops.complex(samples, more_samples)
    return samples


def random_uniform(shape,
                   minval=None,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None):
  """Tensor with (possibly complex) Uniform entries.

  Samples are distributed like

  ```
  Uniform[minval, maxval], if dtype is real,
  X + iY,  where X, Y ~ Uniform[minval, maxval], if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    minval:  `0-D` `Tensor` giving the minimum values.
    maxval:  `0-D` `Tensor` giving the maximum values.
    dtype:  `TensorFlow` `dtype` or Python dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_uniform"):
    samples = random_ops.random_uniform(
        shape, dtype=dtype.real_dtype, minval=minval, maxval=maxval, seed=seed)
    if dtype.is_complex:
      if seed is not None:
        seed += 12345
      more_samples = random_ops.random_uniform(
          shape,
          dtype=dtype.real_dtype,
          minval=minval,
          maxval=maxval,
          seed=seed)
      samples = math_ops.complex(samples, more_samples)
    return samples


def random_sign_uniform(shape,
                        minval=None,
                        maxval=None,
                        dtype=dtypes.float32,
                        seed=None):
  """Tensor with (possibly complex) random entries from a "sign Uniform".

  Letting `Z` be a random variable equal to `-1` and `1` with equal probability,
  Samples from this `Op` are distributed like

  ```
  Z * X, where X ~ Uniform[minval, maxval], if dtype is real,
  Z * (X + iY),  where X, Y ~ Uniform[minval, maxval], if dtype is complex.
  ```

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned tensor.
    minval:  `0-D` `Tensor` giving the minimum values.
    maxval:  `0-D` `Tensor` giving the maximum values.
    dtype:  `TensorFlow` `dtype` or Python dtype
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.
  """
  dtype = dtypes.as_dtype(dtype)

  with ops.name_scope("random_sign_uniform"):
    unsigned_samples = random_uniform(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)
    if seed is not None:
      seed += 12
    signs = math_ops.sign(
        random_ops.random_uniform(
            shape, minval=-1., maxval=1., seed=seed))
    return unsigned_samples * math_ops.cast(signs, unsigned_samples.dtype)


def random_normal_correlated_columns(
    shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, eps=1e-4, seed=None):
  """Batch matrix with (possibly complex) Gaussian entries and correlated cols.

  Returns random batch matrix `A` with specified element-wise `mean`, `stddev`,
  living close to an embedded hyperplane.

  Suppose `shape[-2:] = (M, N)`.

  If `M < N`, `A` is a random `M x N` [batch] matrix with iid Gaussian entries.

  If `M >= N`, then the colums of `A` will be made almost dependent as follows:

  ```
  L = random normal N x N-1 matrix, mean = 0, stddev = 1 / sqrt(N - 1)
  B = random normal M x N-1 matrix, mean = 0, stddev = stddev.

  G = (L B^H)^H, a random normal M x N matrix, living on N-1 dim hyperplane
  E = a random normal M x N matrix, mean = 0, stddev = eps
  mu = a constant M x N matrix, equal to the argument "mean"

  A = G + E + mu
  ```

  Args:
    shape:  Python list of integers.
      Shape of the returned tensor.  Must be at least length two.
    mean:  `Tensor` giving mean of normal to sample from.
    stddev:  `Tensor` giving stdev of normal to sample from.
    dtype:  `TensorFlow` `dtype` or numpy dtype
    eps:  Distance each column is perturbed from the low-dimensional subspace.
    seed:  Python integer seed for the RNG.

  Returns:
    `Tensor` with desired shape and dtype.

  Raises:
    ValueError:  If `shape` is not at least length 2.
  """
  dtype = dtypes.as_dtype(dtype)

  if len(shape) < 2:
    raise ValueError(
        "Argument shape must be at least length 2.  Found: %s" % shape)

  # Shape is the final shape, e.g. [..., M, N]
  shape = list(shape)
  batch_shape = shape[:-2]
  m, n = shape[-2:]

  # If there is only one column, "they" are by definition correlated.
  if n < 2 or n < m:
    return random_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)

  # Shape of the matrix with only n - 1 columns that we will embed in higher
  # dimensional space.
  smaller_shape = batch_shape + [m, n - 1]

  # Shape of the embedding matrix, mapping batch matrices
  # from [..., N-1, M] to [..., N, M]
  embedding_mat_shape = batch_shape + [n, n - 1]

  # This stddev for the embedding_mat ensures final result has correct stddev.
  stddev_mat = 1 / np.sqrt(n - 1)

  with ops.name_scope("random_normal_correlated_columns"):
    smaller_mat = random_normal(
        smaller_shape, mean=0.0, stddev=stddev_mat, dtype=dtype, seed=seed)

    if seed is not None:
      seed += 1287

    embedding_mat = random_normal(embedding_mat_shape, dtype=dtype, seed=seed)

    embedded_t = math_ops.matmul(embedding_mat, smaller_mat, transpose_b=True)
    embedded = array_ops.matrix_transpose(embedded_t)

    mean_mat = array_ops.ones_like(embedded) * mean

    return embedded + random_normal(shape, stddev=eps, dtype=dtype) + mean_mat

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
"""Base class for linear operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperator"]


# TODO(langmore) Use matrix_solve_ls for singular or non-square matrices.
@tf_export("linalg.LinearOperator")
class LinearOperator(object):
  """Base class defining a [batch of] linear operator[s].

  Subclasses of `LinearOperator` provide access to common methods on a
  (batch) matrix, without the need to materialize the matrix.  This allows:

  * Matrix free computations
  * Operators that take advantage of special structure, while providing a
    consistent API to users.

  #### Subclassing

  To enable a public method, subclasses should implement the leading-underscore
  version of the method.  The argument signature should be identical except for
  the omission of `name="..."`.  For example, to enable
  `matmul(x, adjoint=False, name="matmul")` a subclass should implement
  `_matmul(x, adjoint=False)`.

  #### Performance contract

  Subclasses should only implement the assert methods
  (e.g. `assert_non_singular`) if they can be done in less than `O(N^3)`
  time.

  Class docstrings should contain an explanation of computational complexity.
  Since this is a high-performance library, attention should be paid to detail,
  and explanations can include constants as well as Big-O notation.

  #### Shape compatibility

  `LinearOperator` subclasses should operate on a [batch] matrix with
  compatible shape.  Class docstrings should define what is meant by compatible
  shape.  Some subclasses may not support batching.

  Examples:

  `x` is a batch matrix with compatible shape for `matmul` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  x.shape =   [B1,...,Bb] + [N, R]
  ```

  `rhs` is a batch matrix with compatible shape for `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  rhs.shape =   [B1,...,Bb] + [M, R]
  ```

  #### Example docstring for subclasses.

  This operator acts like a (batch) matrix `A` with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `m x n` matrix.  Again, this matrix `A` may not be materialized, but for
  purposes of identifying and working with compatible arguments the shape is
  relevant.

  Examples:

  ```python
  some_tensor = ... shape = ????
  operator = MyLinOp(some_tensor)

  operator.shape()
  ==> [2, 4, 4]

  operator.log_abs_determinant()
  ==> Shape [2] Tensor

  x = ... Shape [2, 4, 5] Tensor

  operator.matmul(x)
  ==> Shape [2, 4, 5] Tensor
  ```

  #### Shape compatibility

  This operator acts on batch matrices with compatible shape.
  FILL IN WHAT IS MEANT BY COMPATIBLE SHAPE

  #### Performance

  FILL THIS IN

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               dtype,
               graph_parents=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
    r"""Initialize the `LinearOperator`.

    **This is a private method for subclass use.**
    **Subclasses should copy-paste this `__init__` documentation.**

    Args:
      dtype: The type of the this `LinearOperator`.  Arguments to `matmul` and
        `solve` will have to be this type.
      graph_parents: Python list of graph prerequisites of this `LinearOperator`
        Typically tensors that are passed during initialization.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `dtype` is real, this is equivalent to being symmetric.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      ValueError:  If any member of graph_parents is `None` or not a `Tensor`.
      ValueError:  If hints are set incorrectly.
    """
    # Check and auto-set flags.
    if is_positive_definite:
      if is_non_singular is False:
        raise ValueError("A positive definite matrix is always non-singular.")
      is_non_singular = True

    if is_non_singular:
      if is_square is False:
        raise ValueError("A non-singular matrix is always square.")
      is_square = True

    if is_self_adjoint:
      if is_square is False:
        raise ValueError("A self-adjoint matrix is always square.")
      is_square = True

    self._is_square_set_or_implied_by_hints = is_square

    graph_parents = [] if graph_parents is None else graph_parents
    for i, t in enumerate(graph_parents):
      if t is None or not tensor_util.is_tensor(t):
        raise ValueError("Graph parent item %d is not a Tensor; %s." % (i, t))
    self._dtype = dtype
    self._graph_parents = graph_parents
    self._is_non_singular = is_non_singular
    self._is_self_adjoint = is_self_adjoint
    self._is_positive_definite = is_positive_definite
    self._name = name or type(self).__name__

  @contextlib.contextmanager
  def _name_scope(self, name=None, values=None):
    """Helper function to standardize op scope."""
    with ops.name_scope(self.name):
      with ops.name_scope(
          name, values=((values or []) + self._graph_parents)) as scope:
        yield scope

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `LinearOperator`."""
    return self._dtype

  @property
  def name(self):
    """Name prepended to all ops created by this `LinearOperator`."""
    return self._name

  @property
  def graph_parents(self):
    """List of graph dependencies of this `LinearOperator`."""
    return self._graph_parents

  @property
  def is_non_singular(self):
    return self._is_non_singular

  @property
  def is_self_adjoint(self):
    return self._is_self_adjoint

  @property
  def is_positive_definite(self):
    return self._is_positive_definite

  @property
  def is_square(self):
    """Return `True/False` depending on if this operator is square."""
    # Static checks done after __init__.  Why?  Because domain/range dimension
    # sometimes requires lots of work done in the derived class after init.
    auto_square_check = self.domain_dimension == self.range_dimension
    if self._is_square_set_or_implied_by_hints is False and auto_square_check:
      raise ValueError(
          "User set is_square hint to False, but the operator was square.")
    if self._is_square_set_or_implied_by_hints is None:
      return auto_square_check

    return self._is_square_set_or_implied_by_hints

  @abc.abstractmethod
  def _shape(self):
    # Write this in derived class to enable all static shape methods.
    raise NotImplementedError("_shape is not implemented.")

  @property
  def shape(self):
    """`TensorShape` of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    return self._shape()

  @abc.abstractmethod
  def _shape_tensor(self):
    raise NotImplementedError("_shape_tensor is not implemented.")

  def shape_tensor(self, name="shape_tensor"):
    """Shape of this `LinearOperator`, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

    Args:
      name:  A name for this `Op.

    Returns:
      `int32` `Tensor`
    """
    with self._name_scope(name):
      # Prefer to use statically defined shape if available.
      if self.shape.is_fully_defined():
        return linear_operator_util.shape_tensor(self.shape.as_list())
      else:
        return self._shape_tensor()

  @property
  def batch_shape(self):
    """`TensorShape` of batch dimensions of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[:-2]

  def batch_shape_tensor(self, name="batch_shape_tensor"):
    """Shape of batch dimensions of this operator, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb]`.

    Args:
      name:  A name for this `Op.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      # Prefer to use statically defined shape if available.
      if self.batch_shape.is_fully_defined():
        return linear_operator_util.shape_tensor(
            self.batch_shape.as_list(), name="batch_shape")
      else:
        return self.shape_tensor()[:-2]

  @property
  def tensor_rank(self, name="tensor_rank"):
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op.

    Returns:
      Python integer, or None if the tensor rank is undefined.
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      return self.shape.ndims

  def tensor_rank_tensor(self, name="tensor_rank_tensor"):
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op.

    Returns:
      `int32` `Tensor`, determined at runtime.
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      # Prefer to use statically defined shape if available.
      if self.tensor_rank is not None:
        return ops.convert_to_tensor(self.tensor_rank)
      else:
        return array_ops.size(self.shape_tensor())

  @property
  def domain_dimension(self):
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Returns:
      `Dimension` object.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[-1]

  def domain_dimension_tensor(self, name="domain_dimension_tensor"):
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      # Prefer to use statically defined shape if available.
      if self.domain_dimension.value is not None:
        return ops.convert_to_tensor(self.domain_dimension.value)
      else:
        return self.shape_tensor()[-1]

  @property
  def range_dimension(self):
    """Dimension (in the sense of vector spaces) of the range of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Returns:
      `Dimension` object.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[-2]

  def range_dimension_tensor(self, name="range_dimension_tensor"):
    """Dimension (in the sense of vector spaces) of the range of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      # Prefer to use statically defined shape if available.
      if self.range_dimension.value is not None:
        return ops.convert_to_tensor(self.range_dimension.value)
      else:
        return self.shape_tensor()[-2]

  def _assert_non_singular(self):
    """Private default implementation of _assert_non_singular."""
    logging.warn(
        "Using (possibly slow) default implementation of assert_non_singular."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      return self.assert_positive_definite()
    else:
      singular_values = linalg_ops.svd(self.to_dense(), compute_uv=False)
      # TODO(langmore) Add .eig and .cond as methods.
      cond = (math_ops.reduce_max(singular_values, axis=-1) /
              math_ops.reduce_min(singular_values, axis=-1))
      return check_ops.assert_less(
          cond,
          self._max_condition_number_to_be_non_singular(),
          message="Singular matrix up to precision epsilon.")

  def _max_condition_number_to_be_non_singular(self):
    """Return the maximum condition number that we consider nonsingular."""
    with ops.name_scope("max_nonsingular_condition_number"):
      dtype_eps = np.finfo(self.dtype.as_numpy_dtype).eps
      eps = math_ops.cast(
          math_ops.reduce_max([
              100.,
              math_ops.cast(self.range_dimension_tensor(), self.dtype),
              math_ops.cast(self.domain_dimension_tensor(), self.dtype)
          ]), self.dtype) * dtype_eps
      return 1. / eps

  def assert_non_singular(self, name="assert_non_singular"):
    """Returns an `Op` that asserts this operator is non singular.

    This operator is considered non-singular if

    ```
    ConditionNumber < max{100, range_dimension, domain_dimension} * eps,
    eps := np.finfo(self.dtype.as_numpy_dtype).eps
    ```

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is singular.
    """
    with self._name_scope(name):
      return self._assert_non_singular()

  def _assert_positive_definite(self):
    """Default implementation of _assert_positive_definite."""
    logging.warn(
        "Using (possibly slow) default implementation of "
        "assert_positive_definite."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    # If the operator is self-adjoint, then checking that
    # Cholesky decomposition succeeds + results in positive diag is necessary
    # and sufficient.
    if self.is_self_adjoint:
      return check_ops.assert_positive(
          array_ops.matrix_diag_part(linalg_ops.cholesky(self.to_dense())),
          message="Matrix was not positive definite.")
    # We have no generic check for positive definite.
    raise NotImplementedError("assert_positive_definite is not implemented.")

  def assert_positive_definite(self, name="assert_positive_definite"):
    """Returns an `Op` that asserts this operator is positive definite.

    Here, positive definite means that the quadratic form `x^H A x` has positive
    real part for all nonzero `x`.  Note that we do not require the operator to
    be self-adjoint to be positive definite.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not positive definite.
    """
    with self._name_scope(name):
      return self._assert_positive_definite()

  def _assert_self_adjoint(self):
    dense = self.to_dense()
    logging.warn(
        "Using (possibly slow) default implementation of assert_self_adjoint."
        "  Requires conversion to a dense matrix.")
    return check_ops.assert_equal(
        dense,
        linalg.adjoint(dense),
        message="Matrix was not equal to its adjoint.")

  def assert_self_adjoint(self, name="assert_self_adjoint"):
    """Returns an `Op` that asserts this operator is self-adjoint.

    Here we check that this operator is *exactly* equal to its hermitian
    transpose.

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not self-adjoint.
    """
    with self._name_scope(name):
      return self._assert_self_adjoint()

  def _check_input_dtype(self, arg):
    """Check that arg.dtype == self.dtype."""
    if arg.dtype != self.dtype:
      raise TypeError(
          "Expected argument to have dtype %s.  Found: %s in tensor %s" %
          (self.dtype, arg.dtype, arg))

  @abc.abstractmethod
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    raise NotImplementedError("_matmul is not implemented.")

  def matmul(self, x, adjoint=False, adjoint_arg=False, name="matmul"):
    """Transform [batch] matrix `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    X = ... # shape [..., N, R], batch matrix, R > 0.

    Y = operator.matmul(X)
    Y.shape
    ==> [..., M, R]

    Y[..., :, r] = sum_j A[..., :, j] X[j, r]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is
        the hermitian transpose (transposition and complex conjugation).
      name:  A name for this `Op.

    Returns:
      A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.
    """
    with self._name_scope(name, values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)

      self_dim = -2 if adjoint else -1
      arg_dim = -1 if adjoint_arg else -2
      self.shape[self_dim].assert_is_compatible_with(x.get_shape()[arg_dim])

      return self._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _matvec(self, x, adjoint=False):
    x_mat = array_ops.expand_dims(x, axis=-1)
    y_mat = self.matmul(x_mat, adjoint=adjoint)
    return array_ops.squeeze(y_mat, axis=-1)

  def matvec(self, x, adjoint=False, name="matvec"):
    """Transform [batch] vector `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matric A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)

    X = ... # shape [..., N], batch vector

    Y = operator.matvec(X)
    Y.shape
    ==> [..., M]

    Y[..., :] = sum_j A[..., :, j] X[..., j]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`.
        `x` is treated as a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      name:  A name for this `Op.

    Returns:
      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.
    """
    with self._name_scope(name, values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)
      self_dim = -2 if adjoint else -1
      self.shape[self_dim].assert_is_compatible_with(x.get_shape()[-1])
      return self._matvec(x, adjoint=adjoint)

  def _determinant(self):
    logging.warn(
        "Using (possibly slow) default implementation of determinant."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      return math_ops.exp(self.log_abs_determinant())
    return linalg_ops.matrix_determinant(self._matrix)

  def determinant(self, name="det"):
    """Determinant for every batch member.

    Args:
      name:  A name for this `Op.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    if self.is_square is False:
      raise NotImplementedError(
          "Determinant not implemented for an operator that is expected to "
          "not be square.")
    with self._name_scope(name):
      return self._determinant()

  def _log_abs_determinant(self):
    logging.warn(
        "Using (possibly slow) default implementation of determinant."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      diag = array_ops.matrix_diag_part(linalg_ops.cholesky(self.to_dense()))
      return 2 * math_ops.reduce_sum(math_ops.log(diag), reduction_indices=[-1])
    _, log_abs_det = linalg.slogdet(self._matrix)
    return log_abs_det

  def log_abs_determinant(self, name="log_abs_det"):
    """Log absolute value of determinant for every batch member.

    Args:
      name:  A name for this `Op.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    if self.is_square is False:
      raise NotImplementedError(
          "Determinant not implemented for an operator that is expected to "
          "not be square.")
    with self._name_scope(name):
      return self._log_abs_determinant()

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    """Default implementation of _solve."""
    if self.is_square is False:
      raise NotImplementedError(
          "Solve is not yet implemented for non-square operators.")
    logging.warn(
        "Using (possibly slow) default implementation of solve."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    if self._can_use_cholesky():
      return linear_operator_util.cholesky_solve_with_broadcast(
          linalg_ops.cholesky(self.to_dense()), rhs)
    return linear_operator_util.matrix_solve_with_broadcast(
        self.to_dense(), rhs, adjoint=adjoint)

  def solve(self, rhs, adjoint=False, adjoint_arg=False, name="solve"):
    """Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve R > 0 linear systems for every member of the batch.
    RHS = ... # shape [..., M, R]

    X = operator.solve(RHS)
    # X[..., :, r] is the solution to the r'th linear system
    # sum_j A[..., :, j] X[..., j, r] = RHS[..., :, r]

    operator.matmul(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape.
        `rhs` is treated like a [batch] matrix meaning for every set of leading
        dimensions, the last two dimensions defines a matrix.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      adjoint_arg:  Python `bool`.  If `True`, solve `A X = rhs^H` where `rhs^H`
        is the hermitian transpose (transposition and complex conjugation).
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    if self.is_non_singular is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "be singular.")
    if self.is_square is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "not be square.")
    with self._name_scope(name, values=[rhs]):
      rhs = ops.convert_to_tensor(rhs, name="rhs")
      self._check_input_dtype(rhs)

      self_dim = -1 if adjoint else -2
      arg_dim = -1 if adjoint_arg else -2
      self.shape[self_dim].assert_is_compatible_with(rhs.get_shape()[arg_dim])

      return self._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _solvevec(self, rhs, adjoint=False):
    """Default implementation of _solvevec."""
    rhs_mat = array_ops.expand_dims(rhs, axis=-1)
    solution_mat = self.solve(rhs_mat, adjoint=adjoint)
    return array_ops.squeeze(solution_mat, axis=-1)

  def solvevec(self, rhs, adjoint=False, name="solve"):
    """Solve single equation with best effort: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve one linear system for every member of the batch.
    RHS = ... # shape [..., M]

    X = operator.solvevec(RHS)
    # X is the solution to the linear system
    # sum_j A[..., :, j] X[..., j] = RHS[..., :]

    operator.matvec(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator.
        `rhs` is treated like a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.  See class docstring
        for definition of compatibility regarding batch dimensions.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    with self._name_scope(name, values=[rhs]):
      rhs = ops.convert_to_tensor(rhs, name="rhs")
      self._check_input_dtype(rhs)
      self_dim = -1 if adjoint else -2
      self.shape[self_dim].assert_is_compatible_with(rhs.get_shape()[-1])

      return self._solvevec(rhs, adjoint=adjoint)

  def _to_dense(self):
    """Generic and often inefficient implementation.  Override often."""
    logging.warn("Using (possibly slow) default implementation of to_dense."
                 "  Converts by self.matmul(identity).")
    if self.batch_shape.is_fully_defined():
      batch_shape = self.batch_shape
    else:
      batch_shape = self.batch_shape_tensor()

    if self.domain_dimension.value is not None:
      n = self.domain_dimension.value
    else:
      n = self.domain_dimension_tensor()

    eye = linalg_ops.eye(num_rows=n, batch_shape=batch_shape, dtype=self.dtype)
    return self.matmul(eye)

  def to_dense(self, name="to_dense"):
    """Return a dense (batch) matrix representing this operator."""
    with self._name_scope(name):
      return self._to_dense()

  def _diag_part(self):
    """Generic and often inefficient implementation.  Override often."""
    return array_ops.matrix_diag_part(self.to_dense())

  def diag_part(self, name="diag_part"):
    """Efficiently get the [batch] diagonal part of this operator.

    If this operator has shape `[B1,...,Bb, M, N]`, this returns a
    `Tensor` `diagonal`, of shape `[B1,...,Bb, min(M, N)]`, where
    `diagonal[b1,...,bb, i] = self.to_dense()[b1,...,bb, i, i]`.

    ```
    my_operator = LinearOperatorDiag([1., 2.])

    # Efficiently get the diagonal
    my_operator.diag_part()
    ==> [1., 2.]

    # Equivalent, but inefficient method
    tf.matrix_diag_part(my_operator.to_dense())
    ==> [1., 2.]
    ```

    Args:
      name:  A name for this `Op`.

    Returns:
      diag_part:  A `Tensor` of same `dtype` as self.
    """
    with self._name_scope(name):
      return self._diag_part()

  def _trace(self):
    return math_ops.reduce_sum(self.diag_part(), axis=-1)

  def trace(self, name="trace"):
    """Trace of the linear operator, equal to sum of `self.diag_part()`.

    If the operator is square, this is also the sum of the eigenvalues.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    """
    with self._name_scope(name):
      return self._trace()

  def _add_to_tensor(self, x):
    # Override if a more efficient implementation is available.
    return self.to_dense() + x

  def add_to_tensor(self, x, name="add_to_tensor"):
    """Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

    Args:
      x:  `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name, values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)
      return self._add_to_tensor(x)

  def _can_use_cholesky(self):
    # TODO(langmore) Add complex types when tf.cholesky can use them.
    return (not self.dtype.is_complex and self.is_self_adjoint and
            self.is_positive_definite)

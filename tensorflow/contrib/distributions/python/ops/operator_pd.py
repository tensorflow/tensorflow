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
"""Base class for symmetric positive definite operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


@six.add_metaclass(abc.ABCMeta)
class OperatorPDBase(object):
  """Class representing a (batch) of positive definite matrices `A`.

  This class provides access to functions of a (batch) symmetric positive
  definite (PD) matrix, without the need to materialize them.  In other words,
  this provides means to do "matrix free" computations.

  ### Basics

  For example, `my_operator.matmul(x)` computes the result of matrix
  multiplication, and this class is free to do this computation with or without
  ever materializing a matrix.

  In practice, this operator represents a (batch) matrix `A` with shape
  `[N1,...,Nn, k, k]` for some `n >= 0`.  The first `n` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,in, : :]` is
  a `k x k` matrix.  Again, this matrix `A` may not be materialized, but for
  purposes of broadcasting this shape will be relevant.

  Since `A` is (batch) positive definite, it has a (or several) square roots `S`
  such that `A = SS^T`.

  For example, if `MyOperator` inherits from `OperatorPDBase`, the user can do

  ```python
  operator = MyOperator(...)  # Initialize with some tensors.
  operator.log_det()

  # Compute the quadratic form x^T A^{-1} x for vector x.
  x = ... # some shape [M1,...,Mm, N1,...,Nn, k] tensor
  operator.inv_quadratic_form_on_vectors(x)

  # Matrix multiplication by the square root, S w.
  # If w is iid normal, S w has covariance A.
  w = ... # some shape [N1,...,Nn, k, r] tensor, r >= 1
  operator.sqrt_matmul(w)
  ```

  The above three methods, `log_det`, `inv_quadratic_form_on_vectors`, and
  `sqrt_matmul` provide "all" that is necessary to use a covariance matrix
  in a multi-variate normal distribution.  See the class `MVNOperatorPD`.

  ### Details about shape requirements

  The `Operator` classes operate on batch vectors and batch matrices with
  compatible shapes.  `matrix` is a batch matrix with compatible shape if

  ```
  operator.shape = [N1,...,Nn] + [j, k]
  matrix.shape =   [N1,...,Nn] + [k, r]
  ```

  This is the same requirement as `tf.matmul`.  `vec` is a batch vector with
  compatible shape if

  ```
  operator.shape = [N1,...,Nn] + [j, k]
  vec.shape =   [N1,...,Nn] + [k]
  OR
  vec.shape = [M1,...,Mm] + [N1,...,Nn] + [k]
  ```

  We are strict with the matrix shape requirements since we do not want to
  require `Operator` broadcasting.  The `Operator` may be defined by large
  tensors (thus broadcasting is expensive), or the `Operator` may be matrix
  free, in which case there is no guarantee that the underlying implementation
  will broadcast.

  We are more flexible with vector shapes since extra leading dimensions can
  be "flipped" to the end to change the vector to a compatible matrix.

  """

  @abc.abstractproperty
  def name(self):
    """String name identifying this `Operator`."""
    return self._name

  @abc.abstractproperty
  def verify_pd(self):
    """Whether to verify that this `Operator` is positive definite."""
    # return self._verify_pd
    pass

  @abc.abstractproperty
  def dtype(self):
    """Data type of matrix elements of `A`."""
    pass

  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`.  Equiv to `A + mat`.

    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs + [mat]):
        mat = ops.convert_to_tensor(mat, name="mat")
        return self._add_to_tensor(mat)

  def _add_to_tensor(self, mat):
    # Re-implement in derived class if a more efficient method is available.
    return self.to_dense() + mat

  def _dispatch_based_on_batch(self, batch_method, singleton_method, **args):
    """Helper to automatically call batch or singleton operation."""
    if self.get_shape().ndims is not None:
      is_batch = self.get_shape().ndims > 2
      if is_batch:
        return batch_method(**args)
      else:
        return singleton_method(**args)
    else:
      is_batch = self.rank() > 2
      return control_flow_ops.cond(
          is_batch,
          lambda: batch_method(**args),
          lambda: singleton_method(**args)
      )

  def inv_quadratic_form_on_vectors(
      self, x, name="inv_quadratic_form_on_vectors"):
    """Compute the quadratic form: `x^T A^{-1} x` where `x` is a batch vector.

    `x` is a batch vector with compatible shape if

    ```
    self.shape = [N1,...,Nn] + [k, k]
    x.shape = [M1,...,Mm] + [N1,...,Nn] + [k]
    ```

    Args:
      x: `Tensor` with compatible batch vector shape and same `dtype` as self.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[M1,...,Mm] + [N1,...,Nn]` and same `dtype`
        as `self`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[x] + self.inputs):
        x = ops.convert_to_tensor(x, name="x")
        return self._inv_quadratic_form_on_vectors(x)

  def _inv_quadratic_form_on_vectors(self, x):
    # Implement in derived class to enable self.inv_quadratic_form_on_vectors().
    #
    # To implement,
    # Depending on which is more efficient, derived class should be a one-liner
    # calling either
    # return self._iqfov_via_sqrt_solve(x)
    # OR
    # return self._iqfov_via_solve(x)
    # both of which are written in this base class.
    raise NotImplementedError(
        "inv_quadratic_form_on_vectors not implemented")

  def _iqfov_via_sqrt_solve(self, x):
    """Get the inverse quadratic form on vectors via a sqrt_solve."""
    # x^{-1} A^{-1} x = || S^{-1}x ||^2,
    # where S is a square root of A (A = SS^T).
    # Steps:
    # 1. Convert x to a matrix, flipping all extra dimensions in `x` to the
    #    final dimension of x_matrix.
    x_matrix = flip_vector_to_matrix(
        x, self.batch_shape(), self.get_batch_shape())
    # 2. Get soln_matrix = S^{-1} x_matrix
    soln_matrix = self.sqrt_solve(x_matrix)
    # 3. Reshape back to a vector.
    soln = flip_matrix_to_vector(
        soln_matrix, extract_batch_shape(x, 1), x.get_shape()[:-1])
    # 4. L2 (batch) vector norm squared.
    result = math_ops.reduce_sum(
        math_ops.square(soln), reduction_indices=[-1])
    result.set_shape(x.get_shape()[:-1])
    return result

  def _iqfov_via_solve(self, x):
    """Get the inverse quadratic form on vectors via a solve."""
    # x^{-1} A^{-1} x
    # 1. Convert x to a matrix, flipping all extra dimensions in `x` to the
    #    final dimension of x_matrix.
    x_matrix = flip_vector_to_matrix(
        x, self.batch_shape(), self.get_batch_shape())
    # 2. Get x_whitened_matrix = A^{-1} x_matrix
    soln_matrix = self.solve(x_matrix)
    # 3. Reshape back to a vector.
    soln = flip_matrix_to_vector(
        soln_matrix, extract_batch_shape(x, 1), x.get_shape()[:-1])
    # 4. Compute the dot product: x^T soln
    result = math_ops.reduce_sum(x * soln, reduction_indices=[-1])
    result.set_shape(x.get_shape()[:-1])
    return result

  def det(self, name="det"):
    """Determinant for every batch member.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      Determinant for every batch member.
    """
    # Derived classes are encouraged to implement log_det() (since it is
    # usually more stable), and then det() comes for free.
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return self._det()

  def _det(self):
    return math_ops.exp(self.log_det())

  def log_det(self, name="log_det"):
    """Log of the determinant for every batch member.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      Logarithm of determinant for every batch member.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return self._dispatch_based_on_batch(self._batch_log_det, self._log_det)

  def _batch_log_det(self):
    # Implement in derived class to enable self.log_det(x).
    raise NotImplementedError("Log determinant (log_det) not implemented.")

  def _log_det(self):
    # As implemented here, this just calls the batch version.  If a more
    # efficient non-batch version is available, override in the derived class.
    return self._batch_log_det()

  def sqrt_log_abs_det(self, name="sqrt_log_det"):
    """Log absolute value determinant of the sqrt `S` for every batch member.

    In most cases, this will be the same as `sqrt_log_det`, but for certain
    operators defined by a square root, this might be implemented slightly
    differently.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      Logarithm of absolute value determinant of the square root `S` for
      every batch member.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return self._dispatch_based_on_batch(
            self._batch_sqrt_log_abs_det, self._sqrt_log_abs_det)

  def sqrt_log_det(self, name="sqrt_log_det"):
    """Log of the determinant of the sqrt `S` for every batch member.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      Logarithm of determinant of the square root `S` for every batch member.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return self._dispatch_based_on_batch(
            self._batch_sqrt_log_det, self._sqrt_log_det)

  def _batch_sqrt_log_det(self):
    # Over-ride in derived class if it can be done more efficiently.
    return 0.5 * self._log_det()

  def _sqrt_log_det(self):
    # As implemented here, this just calls the batch version.  If a more
    # efficient non-batch version is available, override in the derived class.
    return self._batch_sqrt_log_det()

  def _batch_sqrt_log_abs_det(self):
    # Over-ride in derived class if it can be done more efficiently.
    return self._sqrt_log_det()

  def _sqrt_log_abs_det(self):
    # As implemented here, this just calls the batch version.  If a more
    # efficient non-batch version is available, override in the derived class.
    return self._batch_sqrt_log_abs_det()

  @abc.abstractproperty
  def inputs(self):
    """List of tensors that were provided as initialization inputs."""
    pass

  @abc.abstractmethod
  def get_shape(self):
    """Static `TensorShape` of entire operator.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, then this returns
    `TensorShape([N1,...,Nn, k, k])`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    pass

  def get_batch_shape(self):
    """`TensorShape` with batch shape.  Statically determined if possible.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, then this returns `TensorShape([N1,...,Nn])`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    # Derived classes get this "for free" once .get_shape() is implemented.
    return self.get_shape()[:-2]

  def get_vector_shape(self):
    """`TensorShape` of vectors this operator will work with.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, then this returns
    `TensorShape([N1,...,Nn, k])`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    # Derived classes get this "for free" once .get_shape() is implemented.
    return self.get_shape()[:-1]

  def shape(self, name="shape"):
    """Equivalent to `tf.shape(A).`  Equal to `[N1,...,Nn, k, k]`, `n >= 0`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return self._shape()

  @abc.abstractmethod
  def _shape(self):
    # Implement in derived class to enable .shape().
    pass

  def rank(self, name="rank"):
    """Tensor rank.  Equivalent to `tf.rank(A)`.  Will equal `n + 2`.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, the `rank` is `n + 2`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return array_ops.size(self.shape())

  def batch_shape(self, name="batch_shape"):
    """Shape of batches associated with this operator.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, the `batch_shape` is `[N1,...,Nn]`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return array_ops.strided_slice(self.shape(), [0], [self.rank() - 2])

  def vector_shape(self, name="vector_shape"):
    """Shape of (batch) vectors that this (batch) matrix will multiply.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, the `vector_shape` is `[N1,...,Nn, k]`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return array_ops.concat(
            (self.batch_shape(), [self.vector_space_dimension()]), 0)

  def vector_space_dimension(self, name="vector_space_dimension"):
    """Dimension of vector space on which this acts.  The `k` in `R^k`.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, the `vector_space_dimension` is `k`.

    Args:
      name:  A name scope to use for ops added by this method.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return array_ops.gather(self.shape(), self.rank() - 1)

  def matmul(self, x, transpose_x=False, name="matmul"):
    """Left (batch) matmul `x` by this matrix:  `Ax`.

    `x` is a batch matrix with compatible shape if

    ```
    self.shape = [N1,...,Nn] + [k, k]
    x.shape = [N1,...,Nn] + [k, r]
    ```

    Args:
      x: `Tensor` with shape `self.batch_shape + [k, r]` and same `dtype` as
        this `Operator`.
      transpose_x: If `True`, `x` is transposed before multiplication.
      name:  A name to give this `Op`.

    Returns:
      A result equivalent to `tf.matmul(self.to_dense(), x)`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[x] + self.inputs):
        x = ops.convert_to_tensor(x, name="x")
        return self._dispatch_based_on_batch(
            self._batch_matmul, self._matmul, x=x, transpose_x=transpose_x)

  def _batch_matmul(self, x, transpose_x=False):
    # Implement in derived class to enable self.matmul(x).
    raise NotImplementedError("This operator has no batch matmul Op.")

  def _matmul(self, x, transpose_x=False):
    # As implemented here, this just calls the batch version.  If a more
    # efficient non-batch version is available, override in the derived class.
    return self._batch_matmul(x, transpose_x=transpose_x)

  def sqrt_matmul(self, x, transpose_x=False, name="sqrt_matmul"):
    """Left (batch) matmul `x` by a sqrt of this matrix: `Sx` where `A = S S^T`.

    `x` is a batch matrix with compatible shape if

    ```
    self.shape = [N1,...,Nn] + [k, k]
    x.shape = [N1,...,Nn] + [k, r]
    ```

    Args:
      x: `Tensor` with shape `self.batch_shape + [k, r]` and same `dtype` as
        this `Operator`.
      transpose_x: If `True`, `x` is transposed before multiplication.
      name:  A name scope to use for ops added by this method.

    Returns:
      A result equivalent to `tf.matmul(self.sqrt_to_dense(), x)`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[x] + self.inputs):
        x = ops.convert_to_tensor(x, name="x")
        return self._dispatch_based_on_batch(
            self._batch_sqrt_matmul, self._sqrt_matmul, x=x,
            transpose_x=transpose_x)

  def _batch_sqrt_matmul(self, x, transpose_x=False):
    # Implement in derived class to enable self.sqrt_matmul(x).
    raise NotImplementedError("This operator has no batch_sqrt_matmul Op.")

  def _sqrt_matmul(self, x, transpose_x=False):
    # As implemented here, this just calls the batch version.  If a more
    # efficient non-batch version is available, override in the derived class.
    return self._batch_sqrt_matmul(x, transpose_x=transpose_x)

  def solve(self, rhs, name="solve"):
    """Solve `r` batch systems: `A X = rhs`.

    `rhs` is a batch matrix with compatible shape if

    ```python
    self.shape = [N1,...,Nn] + [k, k]
    rhs.shape = [N1,...,Nn] + [k, r]
    ```

    For every batch member, this is done in `O(r*k^2)` complexity using back
    substitution.

    ```python
    # Solve one linear system (r = 1) for every member of the length 10 batch.
    A = ... # shape 10 x 2 x 2
    RHS = ... # shape 10 x 2 x 1
    operator.shape # = 10 x 2 x 2
    X = operator.squrt_solve(RHS)  # shape 10 x 2 x 1
    # operator.squrt_matmul(X) ~ RHS
    X[3, :, 0]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 0]

    # Solve five linear systems (r = 5) for every member of the length 10 batch.
    operator.shape # = 10 x 2 x 2
    RHS = ... # shape 10 x 2 x 5
    ...
    X[3, :, 2]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 2]
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape,
        `rhs.shape = self.shape[:-1] + [r]` for `r >= 1`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with same `dtype` and shape as `x`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[rhs] + self.inputs):
        rhs = ops.convert_to_tensor(rhs, name="rhs")
        return self._dispatch_based_on_batch(
            self._batch_solve, self._solve, rhs=rhs)

  def _solve(self, rhs):
    # As implemented here, this just calls the batch version.  If a more
    # efficient non-batch version is available, override in the derived class.
    return self._batch_solve(rhs)

  def _batch_solve(self, rhs):
    # Implement in derived class to enable self.solve().
    raise NotImplementedError("batch_solve not implemented for this Operator.")

  def sqrt_solve(self, rhs, name="sqrt_solve"):
    """Solve `r` batch systems involving sqrt: `S X = rhs` where `A = SS^T`.

    `rhs` is a batch matrix with compatible shape if

    ```python
    self.shape = [N1,...,Nn] + [k, k]
    rhs.shape = [N1,...,Nn] + [k, r]
    ```

    For every batch member, this is done in `O(r*k^2)` complexity using back
    substitution.

    ```python
    # Solve one linear system (r = 1) for every member of the length 10 batch.
    A = ... # shape 10 x 2 x 2
    RHS = ... # shape 10 x 2 x 1
    operator.shape # = 10 x 2 x 2
    X = operator.squrt_solve(RHS)  # shape 10 x 2 x 1
    # operator.squrt_matmul(X) ~ RHS
    X[3, :, 0]  # Solution to the linear system S[3, :, :] x = RHS[3, :, 0]

    # Solve five linear systems (r = 5) for every member of the length 10 batch.
    operator.shape # = 10 x 2 x 2
    RHS = ... # shape 10 x 2 x 5
    ...
    X[3, :, 2]  # Solution to the linear system S[3, :, :] x = RHS[3, :, 2]
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape,
        `rhs.shape = self.shape[:-1] + [r]` for `r >= 1`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with same `dtype` and shape as `x`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[rhs] + self.inputs):
        rhs = ops.convert_to_tensor(rhs, name="rhs")
        return self._dispatch_based_on_batch(
            self._batch_sqrt_solve, self._sqrt_solve, rhs=rhs)

  def _sqrt_solve(self, rhs):
    # As implemented here, this just calls the batch version.  If a more
    # efficient non-batch version is available, override in the derived class.
    return self._batch_sqrt_solve(rhs)

  def _batch_sqrt_solve(self, rhs):
    # Implement in derived class to enable self.sqrt_solve()
    raise NotImplementedError(
        "batch sqrt_solve not implemented for this Operator.")

  def to_dense(self, name="to_dense"):
    """Return a dense (batch) matrix representing this operator."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return self._to_dense()

  def _to_dense(self):
    # Implement in derived class to enable self.to_dense().
    raise NotImplementedError("This operator has no dense representation.")

  def sqrt_to_dense(self, name="sqrt_to_dense"):
    """Return a dense (batch) matrix representing sqrt of this operator."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.inputs):
        return self._sqrt_to_dense()

  def _sqrt_to_dense(self):
    # Implement in derived class to enable self.sqrt_to_dense().
    raise NotImplementedError("This operator has no dense sqrt representation.")


def flip_matrix_to_vector(mat, batch_shape, static_batch_shape):
  """Flip dims to reshape batch matrix `mat` to a vector with given batch shape.

  ```python
  mat = tf.random_normal(2, 3, 4, 6)

  # Flip the trailing dimension around to the front.
  flip_matrix_to_vector(mat, [6, 2, 3], [6, 3, 2])  # Shape [6, 2, 3, 4]

  # Flip the trailing dimension around then reshape batch indices to batch_shape
  flip_matrix_to_vector(mat, [6, 3, 2], [6, 3, 2])  # Shape [6, 3, 2, 4]
  flip_matrix_to_vector(mat, [2, 3, 2, 3], [2,3,2,3])  # Shape [2, 3, 2, 3, 4]
  ```

  Assume `mat.shape = matrix_batch_shape + [k, M]`.  The returned vector is
  generated in two steps:

  1. Flip the final dimension to the front, giving a shape
    `[M] + matrix_batch_shape + [k]`.
  2. Reshape the leading dimensions, giving final shape = `batch_shape + [k]`.

  The reshape in step 2 will fail if the number of elements is not equal, i.e.
  `M*prod(matrix_batch_shape) != prod(batch_shape)`.

  See also:  flip_vector_to_matrix.

  Args:
    mat:  `Tensor` with rank `>= 2`.
    batch_shape:  `int32` `Tensor` giving leading "batch" shape of result.
    static_batch_shape:  `TensorShape` object giving batch shape of result.

  Returns:
    `Tensor` with same elements as `mat` but with shape `batch_shape + [k]`.
  """
  mat = ops.convert_to_tensor(mat, name="mat")
  if (static_batch_shape.is_fully_defined()
      and mat.get_shape().is_fully_defined()):
    return _flip_matrix_to_vector_static(mat, static_batch_shape)
  else:
    return _flip_matrix_to_vector_dynamic(mat, batch_shape)


def _flip_matrix_to_vector_static(mat, static_batch_shape):
  """Flip matrix to vector with static shapes."""
  mat_rank = mat.get_shape().ndims
  k = mat.get_shape()[-2]
  final_shape = static_batch_shape.concatenate(k)

  # mat.shape = matrix_batch_shape + [k, M]
  # Permutation corresponding to [M] + matrix_batch_shape + [k]
  perm = [mat_rank - 1] + list(range(0, mat_rank - 1))
  mat_with_end_at_beginning = array_ops.transpose(mat, perm=perm)
  vector = array_ops.reshape(mat_with_end_at_beginning, final_shape)
  return vector


def _flip_matrix_to_vector_dynamic(mat, batch_shape):
  """Flip matrix to vector with dynamic shapes."""
  mat_rank = array_ops.rank(mat)
  k = array_ops.gather(array_ops.shape(mat), mat_rank - 2)
  final_shape = array_ops.concat((batch_shape, [k]), 0)

  # mat.shape = matrix_batch_shape + [k, M]
  # Permutation corresponding to [M] + matrix_batch_shape + [k]
  perm = array_ops.concat(([mat_rank - 1], math_ops.range(0, mat_rank - 1)), 0)
  mat_with_end_at_beginning = array_ops.transpose(mat, perm=perm)
  vector = array_ops.reshape(mat_with_end_at_beginning, final_shape)
  return vector


def flip_vector_to_matrix(vec, batch_shape, static_batch_shape):
  """Flip dims to reshape batch vector `x` to a matrix with given batch shape.

  ```python
  vec = tf.random_normal(2, 3, 4, 5)

  # Flip the leading dimension to the end.
  flip_vector_to_matrix(vec, [3, 4], [3, 4])  # Shape [3, 4, 5, 2]

  # Flip nothing, just extend with a singleton dimension.
  flip_vector_to_matrix(vec, [2, 3, 4], [2, 3, 4])  # Shape [2, 3, 4, 5, 1]

  # Flip leading dimension to the end and reshape the batch indices to
  # batch_shape.
  flip_vector_to_matrix(vec, [4, 3], [4, 3])  # Shape [4, 3, 5, 2]
  ```

  Suppose `batch_shape` is length `n`.  Then...

  Given `vec.shape = [M1,...,Mm] + [N1,...,Nn] + [k]`, for some
  `m > 0` we reshape to a batch matrix with shape `batch_shape + [k, M]`
  where `M = M1*...*Mm`.  This is done by "flipping" the leading dimensions to
  the end and possibly reshaping `[N1,...,Nn]` to `batch_shape`.

  In the case `vec.shape = [N1,...,Nn] + [k]`, we reshape to
  `batch_shape + [k, 1]` by extending the tensor with a singleton dimension and
  possibly reshaping `[N1,...,Nn]` to `batch_shape`.

  See also: flip_matrix_to_vector.

  Args:
    vec:  `Tensor` with shape `[M1,...,Mm] + [N1,...,Nn] + [k]`
    batch_shape:  `int32` `Tensor`.
    static_batch_shape:  `TensorShape` with statically determined batch shape.

  Returns:
    `Tensor` with same `dtype` as `vec` and new shape.
  """
  vec = ops.convert_to_tensor(vec, name="vec")
  if (
      vec.get_shape().is_fully_defined()
      and static_batch_shape.is_fully_defined()):
    return _flip_vector_to_matrix_static(vec, static_batch_shape)
  else:
    return _flip_vector_to_matrix_dynamic(vec, batch_shape)


def _flip_vector_to_matrix_dynamic(vec, batch_shape):
  """flip_vector_to_matrix with dynamic shapes."""
  # Shapes associated with batch_shape
  batch_rank = array_ops.size(batch_shape)

  # Shapes associated with vec.
  vec = ops.convert_to_tensor(vec, name="vec")
  vec_shape = array_ops.shape(vec)
  vec_rank = array_ops.rank(vec)
  vec_batch_rank = vec_rank - 1

  m = vec_batch_rank - batch_rank
  # vec_shape_left = [M1,...,Mm] or [].
  vec_shape_left = array_ops.strided_slice(vec_shape, [0], [m])
  # If vec_shape_left = [], then condensed_shape = [1] since reduce_prod([]) = 1
  # If vec_shape_left = [M1,...,Mm], condensed_shape = [M1*...*Mm]
  condensed_shape = [math_ops.reduce_prod(vec_shape_left)]
  k = array_ops.gather(vec_shape, vec_rank - 1)
  new_shape = array_ops.concat((batch_shape, [k], condensed_shape), 0)

  def _flip_front_dims_to_back():
    # Permutation corresponding to [N1,...,Nn] + [k, M1,...,Mm]
    perm = array_ops.concat((math_ops.range(m, vec_rank), math_ops.range(0, m)),
                            0)
    return array_ops.transpose(vec, perm=perm)

  x_flipped = control_flow_ops.cond(
      math_ops.less(0, m),
      _flip_front_dims_to_back,
      lambda: array_ops.expand_dims(vec, -1))

  return array_ops.reshape(x_flipped, new_shape)


def _flip_vector_to_matrix_static(vec, batch_shape):
  """flip_vector_to_matrix with static shapes."""
  # Shapes associated with batch_shape
  batch_rank = batch_shape.ndims

  # Shapes associated with vec.
  vec = ops.convert_to_tensor(vec, name="vec")
  vec_shape = vec.get_shape()
  vec_rank = len(vec_shape)
  vec_batch_rank = vec_rank - 1

  m = vec_batch_rank - batch_rank
  # vec_shape_left = [M1,...,Mm] or [].
  vec_shape_left = vec_shape[:m]
  # If vec_shape_left = [], then condensed_shape = [1] since reduce_prod([]) = 1
  # If vec_shape_left = [M1,...,Mm], condensed_shape = [M1*...*Mm]
  condensed_shape = [np.prod(vec_shape_left)]
  k = vec_shape[-1]
  new_shape = batch_shape.concatenate(k).concatenate(condensed_shape)

  def _flip_front_dims_to_back():
    # Permutation corresponding to [N1,...,Nn] + [k, M1,...,Mm]
    perm = array_ops.concat((math_ops.range(m, vec_rank), math_ops.range(0, m)),
                            0)
    return array_ops.transpose(vec, perm=perm)

  if 0 < m:
    x_flipped = _flip_front_dims_to_back()
  else:
    x_flipped = array_ops.expand_dims(vec, -1)

  return array_ops.reshape(x_flipped, new_shape)


def extract_batch_shape(x, num_event_dims, name="extract_batch_shape"):
  """Extract the batch shape from `x`.

  Assuming `x.shape = batch_shape + event_shape`, when `event_shape` has
  `num_event_dims` dimensions.  This `Op` returns the batch shape `Tensor`.

  Args:
    x: `Tensor` with rank at least `num_event_dims`.  If rank is not high enough
      this `Op` will fail.
    num_event_dims:  `int32` scalar `Tensor`.  The number of trailing dimensions
      in `x` to be considered as part of `event_shape`.
    name:  A name to prepend to created `Ops`.

  Returns:
    batch_shape:  `1-D` `int32` `Tensor`
  """
  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name="x")
    return array_ops.strided_slice(
        array_ops.shape(x), [0], [array_ops.rank(x) - num_event_dims])

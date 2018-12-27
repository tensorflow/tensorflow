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
"""Internal utilities for `LinearOperator` classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg


def assert_no_entries_with_modulus_zero(
    x, message=None, name="assert_no_entries_with_modulus_zero"):
  """Returns `Op` that asserts Tensor `x` has no entries with modulus zero.

  Args:
    x:  Numeric `Tensor`, real, integer, or complex.
    message:  A string message to prepend to failure message.
    name:  A name to give this `Op`.

  Returns:
    An `Op` that asserts `x` has no entries with modulus zero.
  """
  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name="x")
    dtype = x.dtype.base_dtype
    should_be_nonzero = math_ops.abs(x)
    zero = ops.convert_to_tensor(0, dtype=dtype.real_dtype)
    return check_ops.assert_less(zero, should_be_nonzero, message=message)


def assert_zero_imag_part(x, message=None, name="assert_zero_imag_part"):
  """Returns `Op` that asserts Tensor `x` has no non-zero imaginary parts.

  Args:
    x:  Numeric `Tensor`, real, integer, or complex.
    message:  A string message to prepend to failure message.
    name:  A name to give this `Op`.

  Returns:
    An `Op` that asserts `x` has no entries with modulus zero.
  """
  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name="x")
    dtype = x.dtype.base_dtype

    if dtype.is_floating:
      return control_flow_ops.no_op()

    zero = ops.convert_to_tensor(0, dtype=dtype.real_dtype)
    return check_ops.assert_equal(zero, math_ops.imag(x), message=message)


def assert_compatible_matrix_dimensions(operator, x):
  """Assert that an argument to solve/matmul has proper domain dimension.

  If `operator.shape[-2:] = [M, N]`, and `x.shape[-2:] = [Q, R]`, then
  `operator.matmul(x)` is defined only if `N = Q`.  This `Op` returns an
  `Assert` that "fires" if this is not the case.  Static checks are already
  done by the base class `LinearOperator`.

  Args:
    operator:  `LinearOperator`.
    x:  `Tensor`.

  Returns:
    `Assert` `Op`.
  """
  # Static checks are done in the base class.  Only tensor asserts here.
  assert_same_dd = check_ops.assert_equal(
      array_ops.shape(x)[-2],
      operator.domain_dimension_tensor(),
      message=("Incompatible matrix dimensions.  "
               "shape[-2] of argument to be the same as this operator"))

  return assert_same_dd


def assert_is_batch_matrix(tensor):
  """Static assert that `tensor` has rank `2` or higher."""
  sh = tensor.get_shape()
  if sh.ndims is not None and sh.ndims < 2:
    raise ValueError(
        "Expected [batch] matrix to have at least two dimensions.  Found: "
        "%s" % tensor)


def shape_tensor(shape, name=None):
  """Convert Tensor using default type, unless empty list or tuple."""
  # Works just like random_ops._ShapeTensor.
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int32
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name=name)


################################################################################
# Broadcasting versions of common linear algebra functions.
# TODO(b/77519145) Do this more efficiently in some special cases.
################################################################################


def broadcast_matrix_batch_dims(batch_matrices, name=None):
  """Broadcast leading dimensions of zero or more [batch] matrices.

  Example broadcasting one batch dim of two simple matrices.

  ```python
  x = [[1, 2],
       [3, 4]]  # Shape [2, 2], no batch dims

  y = [[[1]]]   # Shape [1, 1, 1], 1 batch dim of shape [1]

  x_bc, y_bc = broadcast_matrix_batch_dims([x, y])

  x_bc
  ==> [[[1, 2],
        [3, 4]]]  # Shape [1, 2, 2], 1 batch dim of shape [1].

  y_bc
  ==> same as y
  ```

  Example broadcasting many batch dims

  ```python
  x = tf.random_normal(shape=(2, 3, 1, 4, 4))
  y = tf.random_normal(shape=(1, 3, 2, 5, 5))
  x_bc, y_bc = broadcast_matrix_batch_dims([x, y])

  x_bc.shape
  ==> (2, 3, 2, 4, 4)

  y_bc.shape
  ==> (2, 3, 2, 5, 5)
  ```

  Args:
    batch_matrices:  Iterable of `Tensor`s, each having two or more dimensions.
    name:  A string name to prepend to created ops.

  Returns:
    bcast_matrices: List of `Tensor`s, with `bcast_matricies[i]` containing
      the values from `batch_matrices[i]`, with possibly broadcast batch dims.

  Raises:
    ValueError:  If any input `Tensor` is statically determined to have less
      than two dimensions.
  """
  with ops.name_scope(
      name or "broadcast_matrix_batch_dims", values=batch_matrices):
    check_ops.assert_proper_iterable(batch_matrices)
    batch_matrices = list(batch_matrices)

    for i, mat in enumerate(batch_matrices):
      batch_matrices[i] = ops.convert_to_tensor(mat)
      assert_is_batch_matrix(batch_matrices[i])

    if len(batch_matrices) < 2:
      return batch_matrices

    # Try static broadcasting.
    # bcast_batch_shape is the broadcast batch shape of ALL matrices.
    # E.g. if batch_matrices = [x, y], with
    # x.shape =    [2, j, k]  (batch shape =    [2])
    # y.shape = [3, 1, l, m]  (batch shape = [3, 1])
    # ==> bcast_batch_shape = [3, 2]
    bcast_batch_shape = batch_matrices[0].get_shape()[:-2]
    for mat in batch_matrices[1:]:
      bcast_batch_shape = array_ops.broadcast_static_shape(
          bcast_batch_shape,
          mat.get_shape()[:-2])
    if bcast_batch_shape.is_fully_defined():
      # The [1, 1] at the end will broadcast with anything.
      bcast_shape = bcast_batch_shape.concatenate([1, 1])
      for i, mat in enumerate(batch_matrices):
        if mat.get_shape()[:-2] != bcast_batch_shape:
          batch_matrices[i] = _broadcast_to_shape(mat, bcast_shape)
      return batch_matrices

    # Since static didn't work, do dynamic, which always copies data.
    bcast_batch_shape = array_ops.shape(batch_matrices[0])[:-2]
    for mat in batch_matrices[1:]:
      bcast_batch_shape = array_ops.broadcast_dynamic_shape(
          bcast_batch_shape,
          array_ops.shape(mat)[:-2])
    bcast_shape = array_ops.concat([bcast_batch_shape, [1, 1]], axis=0)
    for i, mat in enumerate(batch_matrices):
      batch_matrices[i] = _broadcast_to_shape(mat, bcast_shape)

    return batch_matrices


def _broadcast_to_shape(x, shape):
  return x + array_ops.zeros(shape=shape, dtype=x.dtype)


def cholesky_solve_with_broadcast(chol, rhs, name=None):
  """Solve systems of linear equations."""
  with ops.name_scope(name, "CholeskySolveWithBroadcast", [chol, rhs]):
    chol, rhs = broadcast_matrix_batch_dims([chol, rhs])
    return linalg_ops.cholesky_solve(chol, rhs)


def matmul_with_broadcast(a,
                          b,
                          transpose_a=False,
                          transpose_b=False,
                          adjoint_a=False,
                          adjoint_b=False,
                          a_is_sparse=False,
                          b_is_sparse=False,
                          name=None):
  """Multiplies matrix `a` by matrix `b`, producing `a @ b`.

  Works identically to `tf.matmul`, but broadcasts batch dims
  of `a` and `b` if they are determined statically to be different, or if static
  shapes are not fully defined. Attempts are made to avoid unnecessary
  replication of data, but this is not always possible.

  The inputs must be matrices (or tensors of rank > 2, representing batches of
  matrices).

  Both matrices must be of the same type. The supported types are:
  `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

  Either matrix can be transposed or adjointed (conjugated and transposed) on
  the fly by setting one of the corresponding flag to `True`. These are `False`
  by default.

  If one or both of the matrices contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
  This optimization is only available for plain matrices (rank-2 tensors) with
  datatypes `bfloat16` or `float32`.

  For example:

  ```python
  # A 2-batch of 3x4 matrices
  a = tf.random_normal(shape=(2, 3, 4))

  # A single 4x5 matrix
  b = tf.random_normal(shape=(4, 5))

  result = matmul_with_broadcast(a, b)

  result.shape
  ==> (2, 3, 5)

  result[0,...]
  ==> tf.matmul(a[0,...], b)

  result[1,...]
  ==> tf.matmul(a[1,...], b)
  ```

  Args:
    a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
      `complex128` and `rank > 1`.
    b: `Tensor` with same type as `a` having compatible matrix dimensions and
      broadcastable batch dimensions.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    adjoint_a: If `True`, `a` is conjugated and transposed before
      multiplication.
    adjoint_b: If `True`, `b` is conjugated and transposed before
      multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
    the product of the corresponding matrices in `a` and `b`, e.g. if all
    transpose or adjoint attributes are `False`:

    The leading shape of `output` is the result of broadcasting the leading
    dimensions of `a` and `b`.

    `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
    for all indices i, j.

    Note: This is matrix product, not element-wise product.


  Raises:
    ValueError: If transpose_a and adjoint_a, or transpose_b and adjoint_b
      are both set to True.
  """
  with ops.name_scope(name, "MatMulWithBroadcast", [a, b]):
    a = ops.convert_to_tensor(a, name="a")
    b = ops.convert_to_tensor(b, name="b", dtype=a.dtype)

    # If either a or b has extra dims, we can reshape to get rid of them.
    a, b, reshape_inv, still_need_to_transpose = _reshape_for_efficiency(
        a,
        b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b)

    # This will broadcast by brute force if we still need to.
    a, b = broadcast_matrix_batch_dims([a, b])

    a_times_b = math_ops.matmul(
        a,
        b,
        transpose_a=transpose_a and still_need_to_transpose,
        transpose_b=transpose_b and still_need_to_transpose,
        adjoint_a=adjoint_a and still_need_to_transpose,
        adjoint_b=adjoint_b and still_need_to_transpose,
        a_is_sparse=a_is_sparse,
        b_is_sparse=b_is_sparse)

    return reshape_inv(a_times_b)


def matrix_solve_with_broadcast(matrix, rhs, adjoint=False, name=None):
  """Solve systems of linear equations."""
  with ops.name_scope(name, "MatrixSolveWithBroadcast", [matrix, rhs]):
    matrix = ops.convert_to_tensor(matrix, name="matrix")
    rhs = ops.convert_to_tensor(rhs, name="rhs", dtype=matrix.dtype)

    # If either matrix/rhs has extra dims, we can reshape to get rid of them.
    matrix, rhs, reshape_inv, still_need_to_transpose = _reshape_for_efficiency(
        matrix, rhs, adjoint_a=adjoint)

    # This will broadcast by brute force if we still need to.
    matrix, rhs = broadcast_matrix_batch_dims([matrix, rhs])

    solution = linalg_ops.matrix_solve(
        matrix, rhs, adjoint=adjoint and still_need_to_transpose)

    return reshape_inv(solution)


def matrix_triangular_solve_with_broadcast(matrix,
                                           rhs,
                                           lower=True,
                                           adjoint=False,
                                           name=None):
  """Solves triangular systems of linear equations with by backsubstitution.

  Works identically to `tf.matrix_triangular_solve`, but broadcasts batch dims
  of `matrix` and `rhs` (by replicating) if they are determined statically to be
  different, or if static shapes are not fully defined.  Thus, this may result
  in an inefficient replication of data.

  Args:
    matrix: A Tensor. Must be one of the following types:
      `float64`, `float32`, `complex64`, `complex128`. Shape is `[..., M, M]`.
    rhs: A `Tensor`. Must have the same `dtype` as `matrix`.
      Shape is `[..., M, K]`.
    lower: An optional `bool`. Defaults to `True`. Indicates whether the
      innermost matrices in `matrix` are lower or upper triangular.
    adjoint: An optional `bool`. Defaults to `False`. Indicates whether to solve
      with matrix or its (block-wise) adjoint.
    name: A name for the operation (optional).

  Returns:
    `Tensor` with same `dtype` as `matrix` and shape `[..., M, K]`.
  """
  with ops.name_scope(name, "MatrixTriangularSolve", [matrix, rhs]):
    matrix = ops.convert_to_tensor(matrix, name="matrix")
    rhs = ops.convert_to_tensor(rhs, name="rhs", dtype=matrix.dtype)

    # If either matrix/rhs has extra dims, we can reshape to get rid of them.
    matrix, rhs, reshape_inv, still_need_to_transpose = _reshape_for_efficiency(
        matrix, rhs, adjoint_a=adjoint)

    # lower indicates whether the matrix is lower triangular. If we have
    # manually taken adjoint inside _reshape_for_efficiency, it is now upper tri
    if not still_need_to_transpose and adjoint:
      lower = not lower

    # This will broadcast by brute force if we still need to.
    matrix, rhs = broadcast_matrix_batch_dims([matrix, rhs])

    solution = linalg_ops.matrix_triangular_solve(
        matrix,
        rhs,
        lower=lower,
        adjoint=adjoint and still_need_to_transpose)

    return reshape_inv(solution)


def _reshape_for_efficiency(a,
                            b,
                            transpose_a=False,
                            transpose_b=False,
                            adjoint_a=False,
                            adjoint_b=False):
  """Maybe reshape a, b, and return an inverse map.  For matmul/solve."""
  def identity(x):
    return x

  # At this point, we have not taken transpose/adjoint of a/b.
  still_need_to_transpose = True

  if a.shape.ndims is None or b.shape.ndims is None:
    return a, b, identity, still_need_to_transpose

  # This could be handled in the future, but seems less common.
  if a.shape.ndims >= b.shape.ndims:
    return a, b, identity, still_need_to_transpose

  # From now on, we might modify b, but will not modify a.

  # Suppose:
  #   a.shape =     C + [m, n], b.shape =
  #   b.shape = S + C + [n, r]
  b_extra_ndims = b.shape.ndims - a.shape.ndims

  # b_extra_sh = S, b_main_sh = C + [n, r]
  b_extra_sh = array_ops.shape(b)[:b_extra_ndims]
  b_main_sh = array_ops.shape(b)[b_extra_ndims:]

  # No reason to flip unless the extra dims of b are big enough.  Why?
  # Assume adjoint/transpose = False.  Then...
  # By not flipping, we have to replicate a to shape
  #   b_extra_sh + a.shape,
  # which could use extra memory.  But in all cases, the final output has shape
  #   b_extra_sh + a.shape[:-1] + [b.shape[-1]]
  # So we only end up creating a larger object if the end dim of b is smaller
  # than the end dim of a.  This often happens, e.g. if b was a vector that was
  # expanded to a matrix (by appending a singleton).

  # Since adjoint/transpose may not be False, we must make adjustments here.
  # The dim of b that holds the multiple equations.
  a_domain_sz_ = a.shape[-2 if adjoint_a or transpose_a else -1]
  b_eq_sz_ = b.shape[-2 if adjoint_b or transpose_b else -1]
  b_extra_sz_ = (
      np.prod(b.shape[:b_extra_ndims].as_list())
      if b.shape[:b_extra_ndims].is_fully_defined() else None)
  if (a_domain_sz_ is not None and b_eq_sz_ is not None and
      b_extra_sz_ is not None):
    if b_extra_sz_ < 2 or a_domain_sz_ <= b_eq_sz_:
      return a, b, identity, still_need_to_transpose

  # At this point, we're flipping for sure!
  # Any transposes/adjoints will happen here explicitly, rather than in calling
  # code.  Why?  To avoid having to write separate complex code for each case.
  if adjoint_a:
    a = linalg.adjoint(a)
  elif transpose_a:
    a = linalg.transpose(a)
  if adjoint_b:
    b = linalg.adjoint(b)
  elif transpose_b:
    b = linalg.transpose(b)
  still_need_to_transpose = False

  # Recompute shapes, since the transpose/adjoint may have changed them.
  b_extra_sh = array_ops.shape(b)[:b_extra_ndims]
  b_main_sh = array_ops.shape(b)[b_extra_ndims:]

  # Permutation to put the extra dims at the end.
  perm = (
      array_ops.concat(
          (math_ops.range(b_extra_ndims, b.shape.ndims),
           math_ops.range(0, b_extra_ndims)), 0))
  b_extra_on_end = array_ops.transpose(b, perm=perm)

  # Now squash this end into one long dim.
  b_squashed_end = array_ops.reshape(
      b_extra_on_end, array_ops.concat((b_main_sh[:-1], [-1]), 0))

  def reshape_inv(y):
    # Expand the extra dims hanging off the end, "b_extra_sh".
    # Note we use y_sh[:-1] + [b_main_sh[-1]] rather than b_main_sh, because y
    # Could have different batch dims than a and b, because of broadcasting.
    y_extra_shape = array_ops.concat(
        (array_ops.shape(y)[:-1], [b_main_sh[-1]], b_extra_sh), 0)
    y_extra_on_end = array_ops.reshape(y, y_extra_shape)
    return array_ops.transpose(
        y_extra_on_end, perm=array_ops.invert_permutation(perm))

  return a, b_squashed_end, reshape_inv, still_need_to_transpose

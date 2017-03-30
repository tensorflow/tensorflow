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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


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
  """Assert that an argument to solve/apply has proper domain dimension.

  If `operator.shape[-2:] = [M, N]`, and `x.shape[-2:] = [Q, R]`, then
  `operator.apply(x)` is defined only if `N = Q`.  This `Op` returns an
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
          bcast_batch_shape, mat.get_shape()[:-2])
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
          bcast_batch_shape, array_ops.shape(mat)[:-2])
    bcast_shape = array_ops.concat([bcast_batch_shape, [1, 1]], axis=0)
    for i, mat in enumerate(batch_matrices):
      batch_matrices[i] = _broadcast_to_shape(mat, bcast_shape)

    return batch_matrices


def _broadcast_to_shape(x, shape):
  return x + array_ops.zeros(shape=shape, dtype=x.dtype)


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
  with ops.name_scope(name, "MatMulWithBroadcast", [a, b]) as name:
    a, b = broadcast_matrix_batch_dims([a, b])
    return math_ops.matmul(
        a,
        b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b,
        a_is_sparse=a_is_sparse,
        b_is_sparse=b_is_sparse)


def shape_tensor(shape, name=None):
  """Convert Tensor using default type, unless empty list or tuple."""
  # Works just like random_ops._ShapeTensor.
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int32
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name=name)

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
"""`LinearOperator` acting like a zero matrix."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "LinearOperatorZeros",
]


@tf_export("linalg.LinearOperatorZeros")
@linear_operator.make_composite_tensor
class LinearOperatorZeros(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] zero matrix.

  This operator acts like a [batch] zero matrix `A` with shape
  `[B1,...,Bb, N, M]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x M` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorZeros` is initialized with `num_rows`, and optionally
  `num_columns, `batch_shape`, and `dtype` arguments.  If `num_columns` is
  `None`, then this operator will be initialized as a square matrix. If
  `batch_shape` is `None`, this operator efficiently passes through all
  arguments.  If `batch_shape` is provided, broadcasting may occur, which will
  require making copies.

  ```python
  # Create a 2 x 2 zero matrix.
  operator = LinearOperatorZero(num_rows=2, dtype=tf.float32)

  operator.to_dense()
  ==> [[0., 0.]
       [0., 0.]]

  operator.shape
  ==> [2, 2]

  operator.determinant()
  ==> 0.

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor, same as x.

  # Create a 2-batch of 2x2 zero matrices
  operator = LinearOperatorZeros(num_rows=2, batch_shape=[2])
  operator.to_dense()
  ==> [[[0., 0.]
        [0., 0.]],
       [[0., 0.]
        [0., 0.]]]

  # Here, even though the operator has a batch shape, the input is the same as
  # the output, so x can be passed through without a copy.  The operator is able
  # to detect that no broadcast is necessary because both x and the operator
  # have statically defined shape.
  x = ... Shape [2, 2, 3]
  operator.matmul(x)
  ==> Shape [2, 2, 3] Tensor, same as tf.zeros_like(x)

  # Here the operator and x have different batch_shape, and are broadcast.
  # This requires a copy, since the output is different size than the input.
  x = ... Shape [1, 2, 3]
  operator.matmul(x)
  ==> Shape [2, 2, 3] Tensor, equal to tf.zeros_like([x, x])
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, M],  with b >= 0
  x.shape =   [C1,...,Cc] + [M, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

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

  def __init__(self,
               num_rows,
               num_columns=None,
               batch_shape=None,
               dtype=None,
               is_non_singular=False,
               is_self_adjoint=True,
               is_positive_definite=False,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorZeros"):
    r"""Initialize a `LinearOperatorZeros`.

    The `LinearOperatorZeros` is initialized with arguments defining `dtype`
    and shape.

    This operator is able to broadcast the leading (batch) dimensions, which
    sometimes requires copying data.  If `batch_shape` is `None`, the operator
    can take arguments of any batch shape without copying.  See examples.

    Args:
      num_rows:  Scalar non-negative integer `Tensor`.  Number of rows in the
        corresponding zero matrix.
      num_columns:  Scalar non-negative integer `Tensor`.  Number of columns in
        the corresponding zero matrix. If `None`, defaults to the value of
        `num_rows`.
      batch_shape:  Optional `1-D` integer `Tensor`.  The shape of the leading
        dimensions.  If `None`, this operator has no leading dimensions.
      dtype:  Data type of the matrix that this operator represents.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      assert_proper_shapes:  Python `bool`.  If `False`, only perform static
        checks that initialization and method arguments have proper shape.
        If `True`, and static checks are inconclusive, add asserts to the graph.
      name: A name for this `LinearOperator`

    Raises:
      ValueError:  If `num_rows` is determined statically to be non-scalar, or
        negative.
      ValueError:  If `num_columns` is determined statically to be non-scalar,
        or negative.
      ValueError:  If `batch_shape` is determined statically to not be 1-D, or
        negative.
      ValueError:  If any of the following is not `True`:
        `{is_self_adjoint, is_non_singular, is_positive_definite}`.
    """
    parameters = dict(
        num_rows=num_rows,
        num_columns=num_columns,
        batch_shape=batch_shape,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name
    )

    dtype = dtype or dtypes.float32
    self._assert_proper_shapes = assert_proper_shapes

    with ops.name_scope(name):
      dtype = dtypes.as_dtype(dtype)
      if not is_self_adjoint and is_square:
        raise ValueError("A zero operator is always self adjoint.")
      if is_non_singular:
        raise ValueError("A zero operator is always singular.")
      if is_positive_definite:
        raise ValueError("A zero operator is always not positive-definite.")

      super(LinearOperatorZeros, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

      linear_operator_util.assert_not_ref_type(num_rows, "num_rows")
      linear_operator_util.assert_not_ref_type(num_columns, "num_columns")
      linear_operator_util.assert_not_ref_type(batch_shape, "batch_shape")

      self._num_rows = linear_operator_util.shape_tensor(
          num_rows, name="num_rows")
      self._num_rows_static = tensor_util.constant_value(self._num_rows)

      if num_columns is None:
        num_columns = num_rows

      self._num_columns = linear_operator_util.shape_tensor(
          num_columns, name="num_columns")
      self._num_columns_static = tensor_util.constant_value(self._num_columns)

      self._check_domain_range_possibly_add_asserts()

      if (self._num_rows_static is not None and
          self._num_columns_static is not None):
        if is_square and self._num_rows_static != self._num_columns_static:
          raise ValueError(
              "LinearOperatorZeros initialized as is_square=True, but got "
              "num_rows({}) != num_columns({})".format(
                  self._num_rows_static,
                  self._num_columns_static))

      if batch_shape is None:
        self._batch_shape_arg = None
      else:
        self._batch_shape_arg = linear_operator_util.shape_tensor(
            batch_shape, name="batch_shape_arg")
        self._batch_shape_static = tensor_util.constant_value(
            self._batch_shape_arg)
        self._check_batch_shape_possibly_add_asserts()

  def _shape(self):
    matrix_shape = tensor_shape.TensorShape((self._num_rows_static,
                                             self._num_columns_static))
    if self._batch_shape_arg is None:
      return matrix_shape

    batch_shape = tensor_shape.TensorShape(self._batch_shape_static)
    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    matrix_shape = array_ops.stack((self._num_rows, self._num_columns), axis=0)
    if self._batch_shape_arg is None:
      return matrix_shape

    return array_ops.concat((self._batch_shape_arg, matrix_shape), 0)

  def _assert_non_singular(self):
    raise errors.InvalidArgumentError(
        node_def=None, op=None, message="Zero operators are always "
        "non-invertible.")

  def _assert_positive_definite(self):
    raise errors.InvalidArgumentError(
        node_def=None, op=None, message="Zero operators are always "
        "non-positive definite.")

  def _assert_self_adjoint(self):
    return control_flow_ops.no_op("assert_self_adjoint")

  def _possibly_broadcast_batch_shape(self, x):
    """Return 'x', possibly after broadcasting the leading dimensions."""
    # If we have no batch shape, our batch shape broadcasts with everything!
    if self._batch_shape_arg is None:
      return x

    # Static attempt:
    #   If we determine that no broadcast is necessary, pass x through
    #   If we need a broadcast, add to an array of zeros.
    #
    # special_shape is the shape that, when broadcast with x's shape, will give
    # the correct broadcast_shape.  Note that
    #   We have already verified the second to last dimension of self.shape
    #   matches x's shape in assert_compatible_matrix_dimensions.
    #   Also, the final dimension of 'x' can have any shape.
    #   Therefore, the final two dimensions of special_shape are 1's.
    special_shape = self.batch_shape.concatenate([1, 1])
    bshape = array_ops.broadcast_static_shape(x.shape, special_shape)
    if special_shape.is_fully_defined():
      # bshape.is_fully_defined iff special_shape.is_fully_defined.
      if bshape == x.shape:
        return x
      # Use the built in broadcasting of addition.
      zeros = array_ops.zeros(shape=special_shape, dtype=self.dtype)
      return x + zeros

    # Dynamic broadcast:
    #   Always add to an array of zeros, rather than using a "cond", since a
    #   cond would require copying data from GPU --> CPU.
    special_shape = array_ops.concat((self.batch_shape_tensor(), [1, 1]), 0)
    zeros = array_ops.zeros(shape=special_shape, dtype=self.dtype)
    return x + zeros

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    if self._assert_proper_shapes:
      x = linalg.adjoint(x) if adjoint_arg else x
      aps = linear_operator_util.assert_compatible_matrix_dimensions(self, x)
      x = control_flow_ops.with_dependencies([aps], x)
    if self.is_square:
      # Note that adjoint has no effect since this matrix is self-adjoint.
      if adjoint_arg:
        output_shape = array_ops.concat([
            array_ops.shape(x)[:-2],
            [array_ops.shape(x)[-1], array_ops.shape(x)[-2]]], axis=0)
      else:
        output_shape = array_ops.shape(x)

      return self._possibly_broadcast_batch_shape(
          array_ops.zeros(shape=output_shape, dtype=x.dtype))

    x_shape = array_ops.shape(x)
    n = self._num_columns if adjoint else self._num_rows
    m = x_shape[-2] if adjoint_arg else x_shape[-1]

    output_shape = array_ops.concat([x_shape[:-2], [n, m]], axis=0)

    zeros = array_ops.zeros(shape=output_shape, dtype=x.dtype)
    return self._possibly_broadcast_batch_shape(zeros)

  def _determinant(self):
    if self.batch_shape.is_fully_defined():
      return array_ops.zeros(shape=self.batch_shape, dtype=self.dtype)
    else:
      return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _trace(self):
    # Get Tensor of all zeros of same shape as self.batch_shape.
    if self.batch_shape.is_fully_defined():
      return array_ops.zeros(shape=self.batch_shape, dtype=self.dtype)
    else:
      return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _diag_part(self):
    return self._zeros_diag()

  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    return self._possibly_broadcast_batch_shape(mat)

  def _check_domain_range_possibly_add_asserts(self):
    """Static check of init arg `num_rows`, possibly add asserts."""
    # Possibly add asserts.
    if self._assert_proper_shapes:
      self._num_rows = control_flow_ops.with_dependencies([
          check_ops.assert_rank(
              self._num_rows,
              0,
              message="Argument num_rows must be a 0-D Tensor."),
          check_ops.assert_non_negative(
              self._num_rows,
              message="Argument num_rows must be non-negative."),
      ], self._num_rows)
      self._num_columns = control_flow_ops.with_dependencies([
          check_ops.assert_rank(
              self._num_columns,
              0,
              message="Argument num_columns must be a 0-D Tensor."),
          check_ops.assert_non_negative(
              self._num_columns,
              message="Argument num_columns must be non-negative."),
      ], self._num_columns)

    # Static checks.
    if not self._num_rows.dtype.is_integer:
      raise TypeError("Argument num_rows must be integer type.  Found:"
                      " %s" % self._num_rows)

    if not self._num_columns.dtype.is_integer:
      raise TypeError("Argument num_columns must be integer type.  Found:"
                      " %s" % self._num_columns)

    num_rows_static = self._num_rows_static
    num_columns_static = self._num_columns_static

    if num_rows_static is not None:
      if num_rows_static.ndim != 0:
        raise ValueError("Argument num_rows must be a 0-D Tensor.  Found:"
                         " %s" % num_rows_static)

      if num_rows_static < 0:
        raise ValueError("Argument num_rows must be non-negative.  Found:"
                         " %s" % num_rows_static)
    if num_columns_static is not None:
      if num_columns_static.ndim != 0:
        raise ValueError("Argument num_columns must be a 0-D Tensor.  Found:"
                         " %s" % num_columns_static)

      if num_columns_static < 0:
        raise ValueError("Argument num_columns must be non-negative.  Found:"
                         " %s" % num_columns_static)

  def _check_batch_shape_possibly_add_asserts(self):
    """Static check of init arg `batch_shape`, possibly add asserts."""
    if self._batch_shape_arg is None:
      return

    # Possibly add asserts
    if self._assert_proper_shapes:
      self._batch_shape_arg = control_flow_ops.with_dependencies([
          check_ops.assert_rank(
              self._batch_shape_arg,
              1,
              message="Argument batch_shape must be a 1-D Tensor."),
          check_ops.assert_non_negative(
              self._batch_shape_arg,
              message="Argument batch_shape must be non-negative."),
      ], self._batch_shape_arg)

    # Static checks
    if not self._batch_shape_arg.dtype.is_integer:
      raise TypeError("Argument batch_shape must be integer type.  Found:"
                      " %s" % self._batch_shape_arg)

    if self._batch_shape_static is None:
      return  # Cannot do any other static checks.

    if self._batch_shape_static.ndim != 1:
      raise ValueError("Argument batch_shape must be a 1-D Tensor.  Found:"
                       " %s" % self._batch_shape_static)

    if np.any(self._batch_shape_static < 0):
      raise ValueError("Argument batch_shape must be non-negative.  Found:"
                       "%s" % self._batch_shape_static)

  def _min_matrix_dim(self):
    """Minimum of domain/range dimension, if statically available, else None."""
    domain_dim = self.domain_dimension.value
    range_dim = self.range_dimension.value
    if domain_dim is None or range_dim is None:
      return None
    return min(domain_dim, range_dim)

  def _min_matrix_dim_tensor(self):
    """Minimum of domain/range dimension, as a tensor."""
    return math_ops.reduce_min(self.shape_tensor()[-2:])

  def _zeros_diag(self):
    """Returns the diagonal of this operator as all zeros."""
    if self.shape.is_fully_defined():
      d_shape = self.batch_shape.concatenate([self._min_matrix_dim()])
    else:
      d_shape = array_ops.concat(
          [self.batch_shape_tensor(),
           [self._min_matrix_dim_tensor()]], axis=0)

    return array_ops.zeros(shape=d_shape, dtype=self.dtype)

  def _eigvals(self):
    return self._zeros_diag()

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("num_rows", "num_columns", "batch_shape")

  @property
  def _composite_tensor_fields(self):
    return ("num_rows", "num_columns", "batch_shape", "dtype",
            "assert_proper_shapes")

  def __getitem__(self, slices):
    # Slice the batch shape and return a new LinearOperatorIdentity.
    # Use a proxy shape and slice it. Use this as the new batch shape
    new_batch_shape = array_ops.shape(
        array_ops.ones(self._batch_shape_arg)[slices])
    parameters = dict(self.parameters, batch_shape=new_batch_shape)
    return LinearOperatorZeros(**parameters)


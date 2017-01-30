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
"""`LinearOperator` acting like the identity matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.linalg.python.ops import linear_operator
from tensorflow.contrib.linalg.python.ops import linear_operator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

__all__ = [
    "LinearOperatorIdentity",
    "LinearOperatorScaledIdentity",
]


class BaseLinearOperatorIdentity(linear_operator.LinearOperator):

  def _check_num_rows_possibly_add_asserts(self):
    """Static check of init arg `num_rows`, possibly add asserts."""
    # Possibly add asserts.
    if self._assert_proper_shapes:
      self._num_rows = control_flow_ops.with_dependencies(
          [
              check_ops.assert_rank(
                  self._num_rows,
                  0,
                  message="Argument num_rows must be a 0-D Tensor."),
              check_ops.assert_non_negative(
                  self._num_rows,
                  message="Argument num_rows must be non-negative."),
          ],
          self._num_rows)

    # Static checks.
    if not self._num_rows.dtype.is_integer:
      raise TypeError("Argument num_rows must be integer type.  Found:"
                      " %s" % self._num_rows)

    num_rows_static = self._num_rows_static

    if num_rows_static is None:
      return  # Cannot do any other static checks.

    if num_rows_static.ndim != 0:
      raise ValueError("Argument num_rows must be a 0-D Tensor.  Found:"
                       " %s" % num_rows_static)

    if num_rows_static < 0:
      raise ValueError("Argument num_rows must be non-negative.  Found:"
                       " %s" % num_rows_static)


class LinearOperatorIdentity(BaseLinearOperatorIdentity):
  """`LinearOperator` acting like a [batch] square identity matrix.

  This operator acts like a [batch] identity matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorIdentity` is initialized with `num_rows`, and optionally
  `batch_shape`, and `dtype` arguments.  If `batch_shape` is `None`, this
  operator efficiently passes through all arguments.  If `batch_shape` is
  provided, broadcasting may occur, which will require making copies.

  ```python
  # Create a 2 x 2 identity matrix.
  operator = LinearOperatorIdentity(num_rows=2, dtype=tf.float32)

  operator.to_dense()
  ==> [[1., 0.]
       [0., 1.]]

  operator.shape
  ==> [2, 2]

  operator.log_determinant()
  ==> 0.

  x = ... Shape [2, 4] Tensor
  operator.apply(x)
  ==> Shape [2, 4] Tensor, same as x.

  y = tf.random_normal(shape=[3, 2, 4])
  # Note that y.shape is compatible with operator.shape because operator.shape
  # is broadcast to [3, 2, 2].
  # This broadcast does NOT require copying data, since we can infer that y
  # will be passed through without changing shape.  We are always able to infer
  # this if the operator has no batch_shape.
  x = operator.solve(y)
  ==> Shape [3, 2, 4] Tensor, same as y.

  # Create a 2-batch of 2x2 identity matrices
  operator = LinearOperatorIdentity(num_rows=2, batch_shape=[2])
  operator.to_dense()
  ==> [[[1., 0.]
        [0., 1.]],
       [[1., 0.]
        [0., 1.]]]

  # Here, even though the operator has a batch shape, the input is the same as
  # the output, so x can be passed through without a copy.  The operator is able
  # to detect that no broadcast is necessary because both x and the operator
  # have statically defined shape.
  x = ... Shape [2, 2, 3]
  operator.apply(x)
  ==> Shape [2, 2, 3] Tensor, same as x

  # Here the operator and x have different batch_shape, and are broadcast.
  # This requires a copy, since the output is different size than the input.
  x = ... Shape [1, 2, 3]
  operator.apply(x)
  ==> Shape [2, 2, 3] Tensor, equal to [x, x]
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `apply` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  ### Performance

  If `batch_shape` initialization arg is `None`:

  * `operator.apply(x)` is `O(1)`
  * `operator.solve(x)` is `O(1)`
  * `operator.determinant()` is `O(1)`

  If `batch_shape` initialization arg is provided, and static checks cannot
  rule out the need to broadcast:

  * `operator.apply(x)` is `O(D1*...*Dd*N*R)`
  * `operator.solve(x)` is `O(D1*...*Dd*N*R)`
  * `operator.determinant()` is `O(B1*...*Bb)`

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite`.
  These have the following meaning
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
               batch_shape=None,
               dtype=None,
               is_non_singular=True,
               is_self_adjoint=True,
               is_positive_definite=True,
               assert_proper_shapes=False,
               name="LinearOperatorIdentity"):
    """Initialize a `LinearOperatorIdentity`.

    The `LinearOperatorIdentity` is initialized with arguments defining `dtype`
    and shape.

    This operator is able to broadcast the leading (batch) dimensions, which
    sometimes requires copying data.  If `batch_shape` is `None`, the operator
    can take arguments of any batch shape without copying.  See examples.

    Args:
      num_rows:  Scalar non-negative integer `Tensor`.  Number of rows in the
        corresponding identity matrix.
      batch_shape:  Optional `1-D` integer `Tensor`.  The shape of the leading
        dimensions.  If `None`, this operator has no leading dimensions.
      dtype:  Data type of the matrix that this operator represents.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite.
      assert_proper_shapes:  Python `bool`.  If `False`, only perform static
        checks that initialization and method arguments have proper shape.
        If `True`, and static checks are inconclusive, add asserts to the graph.
      name: A name for this `LinearOperator`

    Raises:
      ValueError:  If `num_rows` is determined statically to be non-scalar, or
        negative.
      ValueError:  If `batch_shape` is determined statically to not be 1-D, or
        negative.
      ValueError:  If any of the following is not `True`:
        `{is_self_adjoint, is_non_singular, is_positive_definite}`.
    """
    dtype = dtype or dtypes.float32
    self._assert_proper_shapes = assert_proper_shapes

    with ops.name_scope(name):
      dtype = dtypes.as_dtype(dtype)
      if not is_self_adjoint:
        raise ValueError("An identity operator is always self adjoint.")
      if not is_non_singular:
        raise ValueError("An identity operator is always non-singular.")
      if not is_positive_definite:
        raise ValueError("An identity operator is always positive-definite.")

      super(LinearOperatorIdentity, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          name=name)

      self._num_rows = linear_operator_util.shape_tensor(
          num_rows, name="num_rows")
      self._num_rows_static = tensor_util.constant_value(self._num_rows)
      self._check_num_rows_possibly_add_asserts()

      if batch_shape is None:
        self._batch_shape_arg = None
      else:
        self._batch_shape_arg = linear_operator_util.shape_tensor(
            batch_shape, name="batch_shape_arg")
        self._batch_shape_static = tensor_util.constant_value(
            self._batch_shape_arg)
        self._check_batch_shape_possibly_add_asserts()

  def _shape(self):
    matrix_shape = tensor_shape.TensorShape(
        (self._num_rows_static, self._num_rows_static))
    if self._batch_shape_arg is None:
      return matrix_shape

    batch_shape = tensor_shape.TensorShape(self._batch_shape_static)
    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    matrix_shape = array_ops.stack(
        (self._num_rows, self._num_rows), axis=0)
    if self._batch_shape_arg is None:
      return matrix_shape

    return array_ops.concat((self._batch_shape_arg, matrix_shape), 0)

  def _assert_non_singular(self):
    return control_flow_ops.no_op("assert_non_singular")

  def _assert_positive_definite(self):
    return control_flow_ops.no_op("assert_positive_definite")

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
    bshape = array_ops.broadcast_static_shape(x.get_shape(), special_shape)
    if special_shape.is_fully_defined():
      # bshape.is_fully_defined iff special_shape.is_fully_defined.
      if bshape == x.get_shape():
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

  def _apply(self, x, adjoint=False):
    # Note that adjoint has no effect since this matrix is self-adjoint.
    if self._assert_proper_shapes:
      aps = linear_operator_util.assert_compatible_matrix_dimensions(
          self, x)
      x = control_flow_ops.with_dependencies([aps], x)
    return self._possibly_broadcast_batch_shape(x)

  def _determinant(self):
    return array_ops.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _log_abs_determinant(self):
    return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _solve(self, rhs, adjoint=False):
    return self._apply(rhs)

  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name, values=[mat]):
      mat = ops.convert_to_tensor(mat, name="mat")
      mat_diag = array_ops.matrix_diag_part(mat)
      new_diag = 1 + mat_diag
      return array_ops.matrix_set_diag(mat, new_diag)

  def _check_num_rows_possibly_add_asserts(self):
    """Static check of init arg `num_rows`, possibly add asserts."""
    # Possibly add asserts.
    if self._assert_proper_shapes:
      self._num_rows = control_flow_ops.with_dependencies(
          [
              check_ops.assert_rank(
                  self._num_rows,
                  0,
                  message="Argument num_rows must be a 0-D Tensor."),
              check_ops.assert_non_negative(
                  self._num_rows,
                  message="Argument num_rows must be non-negative."),
          ],
          self._num_rows)

    # Static checks.
    if not self._num_rows.dtype.is_integer:
      raise TypeError("Argument num_rows must be integer type.  Found:"
                      " %s" % self._num_rows)

    num_rows_static = self._num_rows_static

    if num_rows_static is None:
      return  # Cannot do any other static checks.

    if num_rows_static.ndim != 0:
      raise ValueError("Argument num_rows must be a 0-D Tensor.  Found:"
                       " %s" % num_rows_static)

    if num_rows_static < 0:
      raise ValueError("Argument num_rows must be non-negative.  Found:"
                       " %s" % num_rows_static)

  def _check_batch_shape_possibly_add_asserts(self):
    """Static check of init arg `batch_shape`, possibly add asserts."""
    if self._batch_shape_arg is None:
      return

    # Possibly add asserts
    if self._assert_proper_shapes:
      self._batch_shape_arg = control_flow_ops.with_dependencies(
          [
              check_ops.assert_rank(
                  self._batch_shape_arg,
                  1,
                  message="Argument batch_shape must be a 1-D Tensor."),
              check_ops.assert_non_negative(
                  self._batch_shape_arg,
                  message="Argument batch_shape must be non-negative."),
          ],
          self._batch_shape_arg)

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


class LinearOperatorScaledIdentity(BaseLinearOperatorIdentity):
  """`LinearOperator` acting like a scaled [batch] identity matrix `A = c I`.

  This operator acts like a scaled [batch] identity matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  a scaled version of the `N x N` identity matrix.

  `LinearOperatorIdentity` is initialized with `num_rows`, and a `multiplier`
  (a `Tensor`) of shape `[B1,...,Bb]`.  `N` is set to `num_rows`, and the
  `multiplier` determines the scale for each batch member.

  ```python
  # Create a 2 x 2 scaled identity matrix.
  operator = LinearOperatorIdentity(num_rows=2, multiplier=3.)

  operator.to_dense()
  ==> [[3., 0.]
       [0., 3.]]

  operator.shape
  ==> [2, 2]

  operator.log_determinant()
  ==> 2 * Log[3]

  x = ... Shape [2, 4] Tensor
  operator.apply(x)
  ==> 3 * x

  y = tf.random_normal(shape=[3, 2, 4])
  # Note that y.shape is compatible with operator.shape because operator.shape
  # is broadcast to [3, 2, 2].
  x = operator.solve(y)
  ==> 3 * x

  # Create a 2-batch of 2x2 identity matrices
  operator = LinearOperatorIdentity(num_rows=2, multiplier=5.)
  operator.to_dense()
  ==> [[[5., 0.]
        [0., 5.]],
       [[5., 0.]
        [0., 5.]]]

  x = ... Shape [2, 2, 3]
  operator.apply(x)
  ==> 5 * x

  # Here the operator and x have different batch_shape, and are broadcast.
  x = ... Shape [1, 2, 3]
  operator.apply(x)
  ==> 5 * x
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `apply` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  ### Performance

  * `operator.apply(x)` is `O(D1*...*Dd*N*R)`
  * `operator.solve(x)` is `O(D1*...*Dd*N*R)`
  * `operator.determinant()` is `O(D1*...*Dd)`

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite`.
  These have the following meaning
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
               multiplier,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               assert_proper_shapes=False,
               name="LinearOperatorScaledIdentity"):
    """Initialize a `LinearOperatorScaledIdentity`.

    The `LinearOperatorScaledIdentity` is initialized with `num_rows`, which
    determines the size of each identity matrix, and a `multiplier`,
    which defines `dtype`, batch shape, and scale of each matrix.

    This operator is able to broadcast the leading (batch) dimensions.

    Args:
      num_rows:  Scalar non-negative integer `Tensor`.  Number of rows in the
        corresponding identity matrix.
      multiplier:  `Tensor` of shape `[B1,...,Bb]`, or `[]` (a scalar).
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite.
      assert_proper_shapes:  Python `bool`.  If `False`, only perform static
        checks that initialization and method arguments have proper shape.
        If `True`, and static checks are inconclusive, add asserts to the graph.
      name: A name for this `LinearOperator`

    Raises:
      ValueError:  If `num_rows` is determined statically to be non-scalar, or
        negative.
    """
    self._assert_proper_shapes = assert_proper_shapes

    with ops.name_scope(name, values=[multiplier, num_rows]):
      self._multiplier = ops.convert_to_tensor(multiplier, name="multiplier")

      super(LinearOperatorScaledIdentity, self).__init__(
          dtype=self._multiplier.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          name=name)

      # Shape [B1,...Bb, 1, 1]
      self._multiplier_matrix = array_ops.expand_dims(
          array_ops.expand_dims(self.multiplier, -1), -1)
      self._multiplier_matrix_conj = math_ops.conj(
          self._multiplier_matrix)
      self._abs_multiplier = math_ops.abs(self.multiplier)

      self._num_rows = linear_operator_util.shape_tensor(
          num_rows, name="num_rows")
      self._num_rows_static = tensor_util.constant_value(self._num_rows)
      self._check_num_rows_possibly_add_asserts()
      self._num_rows_cast_to_dtype = math_ops.cast(self._num_rows, self.dtype)
      self._num_rows_cast_to_real_dtype = math_ops.cast(
          self._num_rows, self.dtype.real_dtype)

  def _shape(self):
    matrix_shape = tensor_shape.TensorShape(
        (self._num_rows_static, self._num_rows_static))

    batch_shape = self.multiplier.get_shape()
    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    matrix_shape = array_ops.stack(
        (self._num_rows, self._num_rows), axis=0)

    batch_shape = array_ops.shape(self.multiplier)
    return array_ops.concat((batch_shape, matrix_shape), 0)

  def _assert_non_singular(self):
    return check_ops.assert_positive(
        math_ops.abs(self.multiplier),
        message="LinearOperator was singular")

  def _assert_positive_definite(self):
    return check_ops.assert_positive(
        math_ops.real(self.multiplier),
        message="LinearOperator was not positive definite.")

  def _assert_self_adjoint(self):
    imag_multiplier = math_ops.imag(self.multiplier)
    return check_ops.assert_equal(
        array_ops.zeros_like(imag_multiplier),
        imag_multiplier,
        message="LinearOperator was not self-adjoint")

  def _apply(self, x, adjoint=False):
    if adjoint:
      matrix = self._multiplier_matrix_conj
    else:
      matrix = self._multiplier_matrix
    if self._assert_proper_shapes:
      aps = linear_operator_util.assert_compatible_matrix_dimensions(
          self, x)
      x = control_flow_ops.with_dependencies([aps], x)
    return x * matrix

  def _determinant(self):
    return self.multiplier ** self._num_rows_cast_to_dtype

  def _log_abs_determinant(self):
    return self._num_rows_cast_to_real_dtype * math_ops.log(
        self._abs_multiplier)

  def _solve(self, rhs, adjoint=False):
    if adjoint:
      matrix = self._multiplier_matrix_conj
    else:
      matrix = self._multiplier_matrix
    if self._assert_proper_shapes:
      aps = linear_operator_util.assert_compatible_matrix_dimensions(
          self, rhs)
      rhs = control_flow_ops.with_dependencies([aps], rhs)
    return rhs / matrix

  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name, values=[mat]):
      # Shape [B1,...,Bb, 1]
      multiplier_vector = array_ops.expand_dims(self.multiplier, -1)

      # Shape [C1,...,Cc, M, M]
      mat = ops.convert_to_tensor(mat, name="mat")

      # Shape [C1,...,Cc, M]
      mat_diag = array_ops.matrix_diag_part(mat)

      # multiplier_vector broadcasts here.
      new_diag = multiplier_vector + mat_diag

      return array_ops.matrix_set_diag(mat, new_diag)

  @property
  def multiplier(self):
    """The [batch] scalar `Tensor`, `c` in `cI`."""
    return self._multiplier

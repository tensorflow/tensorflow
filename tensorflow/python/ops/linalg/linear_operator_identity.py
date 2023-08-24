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

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import property_hint_util
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "LinearOperatorIdentity",
    "LinearOperatorScaledIdentity",
]


class BaseLinearOperatorIdentity(linear_operator.LinearOperator):
  """Base class for Identity operators."""

  def _check_num_rows_possibly_add_asserts(self):
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

  def _min_matrix_dim(self):
    """Minimum of domain/range dimension, if statically available, else None."""
    domain_dim = tensor_shape.dimension_value(self.domain_dimension)
    range_dim = tensor_shape.dimension_value(self.range_dimension)
    if domain_dim is None or range_dim is None:
      return None
    return min(domain_dim, range_dim)

  def _min_matrix_dim_tensor(self):
    """Minimum of domain/range dimension, as a tensor."""
    return math_ops.reduce_min(self.shape_tensor()[-2:])

  def _ones_diag(self):
    """Returns the diagonal of this operator as all ones."""
    if self.shape.is_fully_defined():
      d_shape = self.batch_shape.concatenate([self._min_matrix_dim()])
    else:
      d_shape = array_ops.concat(
          [self.batch_shape_tensor(),
           [self._min_matrix_dim_tensor()]], axis=0)

    return array_ops.ones(shape=d_shape, dtype=self.dtype)


@tf_export("linalg.LinearOperatorIdentity")
@linear_operator.make_composite_tensor
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

  operator.log_abs_determinant()
  ==> 0.

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor, same as x.

  y = tf.random.normal(shape=[3, 2, 4])
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
  operator.matmul(x)
  ==> Shape [2, 2, 3] Tensor, same as x

  # Here the operator and x have different batch_shape, and are broadcast.
  # This requires a copy, since the output is different size than the input.
  x = ... Shape [1, 2, 3]
  operator.matmul(x)
  ==> Shape [2, 2, 3] Tensor, equal to [x, x]
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  ### Performance

  If `batch_shape` initialization arg is `None`:

  * `operator.matmul(x)` is `O(1)`
  * `operator.solve(x)` is `O(1)`
  * `operator.determinant()` is `O(1)`

  If `batch_shape` initialization arg is provided, and static checks cannot
  rule out the need to broadcast:

  * `operator.matmul(x)` is `O(D1*...*Dd*N*R)`
  * `operator.solve(x)` is `O(D1*...*Dd*N*R)`
  * `operator.determinant()` is `O(B1*...*Bb)`

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
               batch_shape=None,
               dtype=None,
               is_non_singular=True,
               is_self_adjoint=True,
               is_positive_definite=True,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorIdentity"):
    r"""Initialize a `LinearOperatorIdentity`.

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
      ValueError:  If `batch_shape` is determined statically to not be 1-D, or
        negative.
      ValueError:  If any of the following is not `True`:
        `{is_self_adjoint, is_non_singular, is_positive_definite}`.
      TypeError:  If `num_rows` or `batch_shape` is ref-type (e.g. Variable).
    """
    parameters = dict(
        num_rows=num_rows,
        batch_shape=batch_shape,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name)

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
      if not is_square:
        raise ValueError("An identity operator is always square.")

      super(LinearOperatorIdentity, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

      linear_operator_util.assert_not_ref_type(num_rows, "num_rows")
      linear_operator_util.assert_not_ref_type(batch_shape, "batch_shape")

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
    matrix_shape = tensor_shape.TensorShape((self._num_rows_static,
                                             self._num_rows_static))
    if self._batch_shape_arg is None:
      return matrix_shape

    batch_shape = tensor_shape.TensorShape(self._batch_shape_static)
    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    matrix_shape = array_ops_stack.stack(
        (self._num_rows, self._num_rows), axis=0)
    if self._batch_shape_arg is None:
      return matrix_shape

    return array_ops.concat((self._batch_shape_arg, matrix_shape), 0)

  def _linop_adjoint(self) -> "LinearOperatorIdentity":
    return self

  def _linop_cholesky(self) -> "LinearOperatorIdentity":
    return LinearOperatorIdentity(
        num_rows=self._num_rows,  # pylint: disable=protected-access
        batch_shape=self.batch_shape,
        dtype=self.dtype,
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=True,
        is_square=True)

  def _linop_inverse(self) -> "LinearOperatorIdentity":
    return self

  def _linop_matmul(
      self,
      left_operator: "LinearOperatorIdentity",
      right_operator: linear_operator.LinearOperator,
    ) -> "LinearOperatorIdentity":
    del left_operator
    return right_operator

  def _linop_solve(
      self,
      left_operator: "LinearOperatorIdentity",
      right_operator: linear_operator.LinearOperator,
  ) -> linear_operator.LinearOperator:
    del left_operator
    return right_operator

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
    # Note that adjoint has no effect since this matrix is self-adjoint.
    x = linalg.adjoint(x) if adjoint_arg else x
    if self._assert_proper_shapes:
      aps = linear_operator_util.assert_compatible_matrix_dimensions(self, x)
      x = control_flow_ops.with_dependencies([aps], x)
    return self._possibly_broadcast_batch_shape(x)

  def _determinant(self):
    return array_ops.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _log_abs_determinant(self):
    return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    return self._matmul(rhs, adjoint_arg=adjoint_arg)

  def _trace(self):
    # Get Tensor of all ones of same shape as self.batch_shape.
    if self.batch_shape.is_fully_defined():
      batch_of_ones = array_ops.ones(shape=self.batch_shape, dtype=self.dtype)
    else:
      batch_of_ones = array_ops.ones(
          shape=self.batch_shape_tensor(), dtype=self.dtype)

    if self._min_matrix_dim() is not None:
      return self._min_matrix_dim() * batch_of_ones
    else:
      return (math_ops.cast(self._min_matrix_dim_tensor(), self.dtype) *
              batch_of_ones)

  def _diag_part(self):
    return self._ones_diag()

  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      mat = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          mat, name="mat"
      )
      mat_diag = array_ops.matrix_diag_part(mat)
      new_diag = 1 + mat_diag
      return array_ops.matrix_set_diag(mat, new_diag)

  def _eigvals(self):
    return self._ones_diag()

  def _cond(self):
    return array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)

  def _check_num_rows_possibly_add_asserts(self):
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

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("num_rows", "batch_shape")

  @property
  def _composite_tensor_fields(self):
    return ("num_rows", "batch_shape", "dtype", "assert_proper_shapes")

  def __getitem__(self, slices):
    # Slice the batch shape and return a new LinearOperatorIdentity.
    # Use a proxy shape and slice it. Use this as the new batch shape
    new_batch_shape = array_ops.shape(
        array_ops.ones(self._batch_shape_arg)[slices])
    parameters = dict(self.parameters, batch_shape=new_batch_shape)
    return LinearOperatorIdentity(**parameters)


@tf_export("linalg.LinearOperatorScaledIdentity")
@linear_operator.make_composite_tensor
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

  operator.log_abs_determinant()
  ==> 2 * Log[3]

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> 3 * x

  y = tf.random.normal(shape=[3, 2, 4])
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
  operator.matmul(x)
  ==> 5 * x

  # Here the operator and x have different batch_shape, and are broadcast.
  x = ... Shape [1, 2, 3]
  operator.matmul(x)
  ==> 5 * x
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  ### Performance

  * `operator.matmul(x)` is `O(D1*...*Dd*N*R)`
  * `operator.solve(x)` is `O(D1*...*Dd*N*R)`
  * `operator.determinant()` is `O(D1*...*Dd)`

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
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
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorScaledIdentity"):
    r"""Initialize a `LinearOperatorScaledIdentity`.

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
    """
    parameters = dict(
        num_rows=num_rows,
        multiplier=multiplier,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name)

    self._assert_proper_shapes = assert_proper_shapes

    with ops.name_scope(name, values=[multiplier, num_rows]):
      self._multiplier = linear_operator_util.convert_nonref_to_tensor(
          multiplier, name="multiplier")

      # Check and auto-set hints.
      if not self._multiplier.dtype.is_complex:
        if is_self_adjoint is False:  # pylint: disable=g-bool-id-comparison
          raise ValueError("A real diagonal operator is always self adjoint.")
        else:
          is_self_adjoint = True

      if not is_square:
        raise ValueError("A ScaledIdentity operator is always square.")

      linear_operator_util.assert_not_ref_type(num_rows, "num_rows")

      super(LinearOperatorScaledIdentity, self).__init__(
          dtype=self._multiplier.dtype.base_dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

      self._num_rows = linear_operator_util.shape_tensor(
          num_rows, name="num_rows")
      self._num_rows_static = tensor_util.constant_value(self._num_rows)
      self._check_num_rows_possibly_add_asserts()
      self._num_rows_cast_to_dtype = math_ops.cast(self._num_rows, self.dtype)
      self._num_rows_cast_to_real_dtype = math_ops.cast(self._num_rows,
                                                        self.dtype.real_dtype)

  def _shape(self):
    matrix_shape = tensor_shape.TensorShape((self._num_rows_static,
                                             self._num_rows_static))

    batch_shape = self.multiplier.shape
    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    matrix_shape = array_ops_stack.stack(
        (self._num_rows, self._num_rows), axis=0)

    batch_shape = array_ops.shape(self.multiplier)
    return array_ops.concat((batch_shape, matrix_shape), 0)

  def _assert_non_singular(self):
    return check_ops.assert_positive(
        math_ops.abs(self.multiplier), message="LinearOperator was singular")

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

  def _make_multiplier_matrix(self, conjugate=False):
    # Shape [B1,...Bb, 1, 1]
    multiplier_matrix = array_ops.expand_dims(
        array_ops.expand_dims(self.multiplier, -1), -1)
    if conjugate:
      multiplier_matrix = math_ops.conj(multiplier_matrix)
    return multiplier_matrix

  def _linop_adjoint(self) -> "LinearOperatorScaledIdentity":
    multiplier = self.multiplier
    if multiplier.dtype.is_complex:
      multiplier = math_ops.conj(multiplier)

    return LinearOperatorScaledIdentity(
        num_rows=self._num_rows,
        multiplier=multiplier,
        is_non_singular=self.is_non_singular,
        is_self_adjoint=self.is_self_adjoint,
        is_positive_definite=self.is_positive_definite,
        is_square=True)

  def _linop_cholesky(self) -> "LinearOperatorScaledIdentity":
    return LinearOperatorScaledIdentity(
        num_rows=self._num_rows,
        multiplier=math_ops.sqrt(self.multiplier),
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=True,
        is_square=True)

  def _linop_inverse(self) -> "LinearOperatorScaledIdentity":
    return LinearOperatorScaledIdentity(
        num_rows=self._num_rows,
        multiplier=1. / self.multiplier,
        is_non_singular=self.is_non_singular,
        is_self_adjoint=True,
        is_positive_definite=self.is_positive_definite,
        is_square=True)

  def _linop_matmul(
      self,
      left_operator: "LinearOperatorScaledIdentity",
      right_operator: linear_operator.LinearOperator,
    ) -> "LinearOperatorScaledIdentity":
    is_non_singular = property_hint_util.combined_non_singular_hint(
        left_operator, right_operator)
    is_self_adjoint = property_hint_util.combined_commuting_self_adjoint_hint(
        left_operator, right_operator)
    is_positive_definite = (
        property_hint_util.combined_commuting_positive_definite_hint(
            left_operator, right_operator))
    if isinstance(right_operator, LinearOperatorScaledIdentity):
      return LinearOperatorScaledIdentity(
          num_rows=left_operator.domain_dimension_tensor(),
          multiplier=left_operator.multiplier * right_operator.multiplier,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True)
    elif isinstance(right_operator, linear_operator_diag.LinearOperatorDiag):
      return linear_operator_diag.LinearOperatorDiag(
          diag=right_operator.diag * left_operator.multiplier,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True)
    else:
      return super()._linop_matmul(left_operator, right_operator)

  def _linop_solve(
      self,
      left_operator: "LinearOperatorScaledIdentity",
      right_operator: linear_operator.LinearOperator,
  ) -> linear_operator.LinearOperator:
    is_non_singular = property_hint_util.combined_non_singular_hint(
        left_operator, right_operator)
    is_self_adjoint = property_hint_util.combined_commuting_self_adjoint_hint(
        left_operator, right_operator)
    is_positive_definite = (
        property_hint_util.combined_commuting_positive_definite_hint(
            left_operator, right_operator))
    if isinstance(right_operator, LinearOperatorScaledIdentity):
      return LinearOperatorScaledIdentity(
          num_rows=left_operator.domain_dimension_tensor(),
          multiplier=right_operator.multiplier / left_operator.multiplier,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True)
    elif isinstance(right_operator, linear_operator_diag.LinearOperatorDiag):
      return linear_operator_diag.LinearOperatorDiag(
          diag=right_operator.diag / left_operator.multiplier,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True)
    else:
      return super()._linop_solve(left_operator, right_operator)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x = linalg.adjoint(x) if adjoint_arg else x
    if self._assert_proper_shapes:
      aps = linear_operator_util.assert_compatible_matrix_dimensions(self, x)
      x = control_flow_ops.with_dependencies([aps], x)
    return x * self._make_multiplier_matrix(conjugate=adjoint)

  def _determinant(self):
    return self.multiplier**self._num_rows_cast_to_dtype

  def _log_abs_determinant(self):
    return self._num_rows_cast_to_real_dtype * math_ops.log(
        math_ops.abs(self.multiplier))

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    if self._assert_proper_shapes:
      aps = linear_operator_util.assert_compatible_matrix_dimensions(self, rhs)
      rhs = control_flow_ops.with_dependencies([aps], rhs)
    return rhs / self._make_multiplier_matrix(conjugate=adjoint)

  def _trace(self):
    # Get Tensor of all ones of same shape as self.batch_shape.
    if self.batch_shape.is_fully_defined():
      batch_of_ones = array_ops.ones(shape=self.batch_shape, dtype=self.dtype)
    else:
      batch_of_ones = array_ops.ones(
          shape=self.batch_shape_tensor(), dtype=self.dtype)

    if self._min_matrix_dim() is not None:
      return self.multiplier * self._min_matrix_dim() * batch_of_ones
    else:
      return (self.multiplier * math_ops.cast(self._min_matrix_dim_tensor(),
                                              self.dtype) * batch_of_ones)

  def _diag_part(self):
    return self._ones_diag() * self.multiplier[..., array_ops.newaxis]

  def add_to_tensor(self, mat, name="add_to_tensor"):
    """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      # Shape [B1,...,Bb, 1]
      multiplier_vector = array_ops.expand_dims(self.multiplier, -1)

      # Shape [C1,...,Cc, M, M]
      mat = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          mat, name="mat"
      )

      # Shape [C1,...,Cc, M]
      mat_diag = array_ops.matrix_diag_part(mat)

      # multiplier_vector broadcasts here.
      new_diag = multiplier_vector + mat_diag

      return array_ops.matrix_set_diag(mat, new_diag)

  def _eigvals(self):
    return self._ones_diag() * self.multiplier[..., array_ops.newaxis]

  def _cond(self):
    # Condition number for a scalar time identity matrix is one, except when the
    # scalar is zero.
    return array_ops.where_v2(
        math_ops.equal(self._multiplier, 0.),
        math_ops.cast(np.nan, dtype=self.dtype),
        math_ops.cast(1., dtype=self.dtype))

  @property
  def multiplier(self):
    """The [batch] scalar `Tensor`, `c` in `cI`."""
    return self._multiplier

  @property
  def _composite_tensor_prefer_static_fields(self):
    return ("num_rows",)

  @property
  def _composite_tensor_fields(self):
    return ("num_rows", "multiplier", "assert_proper_shapes")

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"multiplier": 0}

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Create a blockwise lower-triangular operator from `LinearOperators`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorBlockLowerTriangular"]


@tf_export("linalg.LinearOperatorBlockLowerTriangular")
@linear_operator.make_composite_tensor
class LinearOperatorBlockLowerTriangular(linear_operator.LinearOperator):
  """Combines `LinearOperators` into a blockwise lower-triangular matrix.

  This operator is initialized with a nested list of linear operators, which
  are combined into a new `LinearOperator` whose underlying matrix
  representation is square and has each operator on or below the main diagonal,
  and zero's elsewhere. Each element of the outer list is a list of
  `LinearOperators` corresponding to a row-partition of the blockwise structure.
  The number of `LinearOperator`s in row-partion `i` must be equal to `i`.

  For example, a blockwise `3 x 3` `LinearOperatorBlockLowerTriangular` is
  initialized with the list `[[op_00], [op_10, op_11], [op_20, op_21, op_22]]`,
  where the `op_ij`, `i < 3, j <= i`, are `LinearOperator` instances. The
  `LinearOperatorBlockLowerTriangular` behaves as the following blockwise
  matrix, where `0` represents appropriately-sized [batch] matrices of zeros:

  ```none
  [[op_00,     0,     0],
   [op_10, op_11,     0],
   [op_20, op_21, op_22]]
  ```

  Each `op_jj` on the diagonal is required to represent a square matrix, and
  hence will have shape `batch_shape_j + [M_j, M_j]`. `LinearOperator`s in row
  `j` of the blockwise structure must have `range_dimension` equal to that of
  `op_jj`, and `LinearOperators` in column `j` must have `domain_dimension`
  equal to that of `op_jj`.

  If each `op_jj` on the diagonal has shape `batch_shape_j + [M_j, M_j]`, then
  the combined operator has shape `broadcast_batch_shape + [sum M_j, sum M_j]`,
  where `broadcast_batch_shape` is the mutual broadcast of `batch_shape_j`,
  `j = 0, 1, ..., J`, assuming the intermediate batch shapes broadcast.
  Even if the combined shape is well defined, the combined operator's
  methods may fail due to lack of broadcasting ability in the defining
  operators' methods.

  For example, to create a 4 x 4 linear operator combined of three 2 x 2
  operators:
  >>> operator_0 = tf.linalg.LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  >>> operator_1 = tf.linalg.LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
  >>> operator_2 = tf.linalg.LinearOperatorLowerTriangular([[5., 6.], [7., 8]])
  >>> operator = LinearOperatorBlockLowerTriangular(
  ...   [[operator_0], [operator_1, operator_2]])

  >>> operator.to_dense()
  <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
  array([[1., 2., 0., 0.],
         [3., 4., 0., 0.],
         [1., 0., 5., 0.],
         [0., 1., 7., 8.]], dtype=float32)>

  >>> operator.shape
  TensorShape([4, 4])

  >>> operator.log_abs_determinant()
  <tf.Tensor: shape=(), dtype=float32, numpy=4.3820267>

  >>> x0 = [[1., 6.], [-3., 4.]]
  >>> x1 = [[0., 2.], [4., 0.]]
  >>> x = tf.concat([x0, x1], 0)  # Shape [2, 4] Tensor
  >>> operator.matmul(x)
  <tf.Tensor: shape=(4, 2), dtype=float32, numpy=
  array([[-5., 14.],
         [-9., 34.],
         [ 1., 16.],
         [29., 18.]], dtype=float32)>

  The above `matmul` is equivalent to:
  >>> tf.concat([operator_0.matmul(x0),
  ...   operator_1.matmul(x0) + operator_2.matmul(x1)], axis=0)
  <tf.Tensor: shape=(4, 2), dtype=float32, numpy=
  array([[-5., 14.],
         [-9., 34.],
         [ 1., 16.],
         [29., 18.]], dtype=float32)>

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```

  For example:

  Create a [2, 3] batch of 4 x 4 linear operators:
  >>> matrix_44 = tf.random.normal(shape=[2, 3, 4, 4])
  >>> operator_44 = tf.linalg.LinearOperatorFullMatrix(matrix_44)

  Create a [1, 3] batch of 5 x 4 linear operators:
  >>> matrix_54 = tf.random.normal(shape=[1, 3, 5, 4])
  >>> operator_54 = tf.linalg.LinearOperatorFullMatrix(matrix_54)

  Create a [1, 3] batch of 5 x 5 linear operators:
  >>> matrix_55 = tf.random.normal(shape=[1, 3, 5, 5])
  >>> operator_55 = tf.linalg.LinearOperatorFullMatrix(matrix_55)

  Combine to create a [2, 3] batch of 9 x 9 operators:
  >>> operator_99 = LinearOperatorBlockLowerTriangular(
  ...   [[operator_44], [operator_54, operator_55]])
  >>> operator_99.shape
  TensorShape([2, 3, 9, 9])

  Create a shape [2, 1, 9] batch of vectors and apply the operator to it.
  >>> x = tf.random.normal(shape=[2, 1, 9])
  >>> y = operator_99.matvec(x)
  >>> y.shape
  TensorShape([2, 3, 9])

  Create a blockwise list of vectors and apply the operator to it. A blockwise
  list is returned.
  >>> x4 = tf.random.normal(shape=[2, 1, 4])
  >>> x5 = tf.random.normal(shape=[2, 3, 5])
  >>> y_blockwise = operator_99.matvec([x4, x5])
  >>> y_blockwise[0].shape
  TensorShape([2, 3, 4])
  >>> y_blockwise[1].shape
  TensorShape([2, 3, 5])

  #### Performance

  Suppose `operator` is a `LinearOperatorBlockLowerTriangular` consisting of `D`
  row-partitions and `D` column-partitions, such that the total number of
  operators is `N = D * (D + 1) // 2`.

  * `operator.matmul` has complexity equal to the sum of the `matmul`
    complexities of the individual operators.
  * `operator.solve` has complexity equal to the sum of the `solve` complexities
    of the operators on the diagonal and the `matmul` complexities of the
    operators off the diagonal.
  * `operator.determinant` has complexity equal to the sum of the `determinant`
    complexities of the operators on the diagonal.

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
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorBlockLowerTriangular"):
    r"""Initialize a `LinearOperatorBlockLowerTriangular`.

    `LinearOperatorBlockLowerTriangular` is initialized with a list of lists of
    operators `[[op_0], [op_1, op_2], [op_3, op_4, op_5],...]`.

    Args:
      operators:  Iterable of iterables of `LinearOperator` objects, each with
        the same `dtype`. Each element of `operators` corresponds to a row-
        partition, in top-to-bottom order. The operators in each row-partition
        are filled in left-to-right. For example,
        `operators = [[op_0], [op_1, op_2], [op_3, op_4, op_5]]` creates a
        `LinearOperatorBlockLowerTriangular` with full block structure
        `[[op_0, 0, 0], [op_1, op_2, 0], [op_3, op_4, op_5]]`. The number of
        operators in the `i`th row must be equal to `i`, such that each operator
        falls on or below the diagonal of the blockwise structure.
        `LinearOperator`s that fall on the diagonal (the last elements of each
        row) must be square. The other `LinearOperator`s must have domain
        dimension equal to the domain dimension of the `LinearOperator`s in the
        same column-partition, and range dimension equal to the range dimension
        of the `LinearOperator`s in the same row-partition.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
        This will raise a `ValueError` if set to `False`.
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If all operators do not have the same `dtype`.
      ValueError:  If `operators` is empty, contains an erroneous number of
        elements, or contains operators with incompatible shapes.
    """
    parameters = dict(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    # Validate operators.
    check_ops.assert_proper_iterable(operators)
    for row in operators:
      check_ops.assert_proper_iterable(row)
    operators = [list(row) for row in operators]

    if not operators:
      raise ValueError(f"Argument `operators` must be a list of >=1 operators. "
                       f"Received: {operators}.")
    self._operators = operators
    self._diagonal_operators = [row[-1] for row in operators]

    dtype = operators[0][0].dtype
    self._validate_dtype(dtype)
    is_non_singular = self._validate_non_singular(is_non_singular)
    self._validate_num_operators()
    self._validate_operator_dimensions()
    is_square = self._validate_square(is_square)
    with ops.name_scope(name):
      super(LinearOperatorBlockLowerTriangular, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _validate_num_operators(self):
    for i, row in enumerate(self.operators):
      if len(row) != i + 1:
        raise ValueError(
            f"Argument `operators[{i}]` must contain `{i + 1}` blocks. "
            f"Received: {len(row)} blocks.")

  def _validate_operator_dimensions(self):
    """Check that `operators` have compatible dimensions."""
    for i in range(1, len(self.operators)):
      for j in range(i):
        op = self.operators[i][j]

        # `above_op` is the operator directly above `op` in the blockwise
        # structure, in row partition `i-1`, column partition `j`. `op` should
        # have the same `domain_dimension` as `above_op`.
        above_op = self.operators[i - 1][j]

        # `right_op` is the operator to the right of `op` in the blockwise
        # structure, in row partition `i`, column partition `j+1`. `op` should
        # have the same `range_dimension` as `right_op`.
        right_op = self.operators[i][j + 1]

        if (op.domain_dimension is not None and
            above_op.domain_dimension is not None):
          if op.domain_dimension != above_op.domain_dimension:
            raise ValueError(f"Argument `operators[{i}][{j}].domain_dimension` "
                             f"({op.domain_dimension}) must be the same as "
                             f"`operators[{i-1}][{j}].domain_dimension` "
                             f"({above_op.domain_dimension}).")
        if (op.range_dimension is not None and
            right_op.range_dimension is not None):
          if op.range_dimension != right_op.range_dimension:
            raise ValueError(f"Argument `operators[{i}][{j}].range_dimension` "
                             f"({op.range_dimension}) must be the same as "
                             f"`operators[{i}][{j + 1}].range_dimension` "
                             f"({right_op.range_dimension}).")

  # pylint: disable=g-bool-id-comparison
  def _validate_non_singular(self, is_non_singular):
    if all(op.is_non_singular for op in self._diagonal_operators):
      if is_non_singular is False:
        raise ValueError(
            f"A blockwise lower-triangular operator with non-singular "
            f"operators on the main diagonal is always non-singular. "
            f"Expected argument `is_non_singular` to be True. "
            f"Received: {is_non_singular}.")
      return True
    if any(op.is_non_singular is False for op in self._diagonal_operators):
      if is_non_singular is True:
        raise ValueError(
            f"A blockwise lower-triangular operator with a singular operator "
            f"on the main diagonal is always singular. Expected argument "
            f"`is_non_singular` to be True. Received: {is_non_singular}.")
      return False

  def _validate_square(self, is_square):
    if is_square is False:
      raise ValueError(f"`LinearOperatorBlockLowerTriangular` must be square. "
                       f"Expected argument `is_square` to be True. "
                       f"Received: {is_square}.")
    for i, op in enumerate(self._diagonal_operators):
      if op.is_square is False:
        raise ValueError(
            f"Matrices on the diagonal (the final elements of each "
            f"row-partition in the `operators` list) must be square. Expected "
            f"argument `operators[{i}][-1].is_square` to be True. "
            f"Received: {op.is_square}.")
    return True
  # pylint: enable=g-bool-id-comparison

  def _validate_dtype(self, dtype):
    for i, row in enumerate(self.operators):
      for operator in row:
        if operator.dtype != dtype:
          name_type = (str((o.name, o.dtype)) for o in row)
          raise TypeError(
              "Expected all operators to have the same dtype.  Found {} in row "
              "{} and {} in row 0.".format(name_type, i, str(dtype)))

  @property
  def operators(self):
    return self._operators

  def _block_range_dimensions(self):
    return [op.range_dimension for op in self._diagonal_operators]

  def _block_domain_dimensions(self):
    return [op.domain_dimension for op in self._diagonal_operators]

  def _block_range_dimension_tensors(self):
    return [op.range_dimension_tensor() for op in self._diagonal_operators]

  def _block_domain_dimension_tensors(self):
    return [op.domain_dimension_tensor() for op in self._diagonal_operators]

  def _shape(self):
    # Get final matrix shape.
    domain_dimension = sum(self._block_domain_dimensions())
    range_dimension = sum(self._block_range_dimensions())
    matrix_shape = tensor_shape.TensorShape([domain_dimension, range_dimension])

    # Get broadcast batch shape.
    # broadcast_shape checks for compatibility.
    batch_shape = self.operators[0][0].batch_shape
    for row in self.operators[1:]:
      for operator in row:
        batch_shape = common_shapes.broadcast_shape(
            batch_shape, operator.batch_shape)

    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    # Avoid messy broadcasting if possible.
    if self.shape.is_fully_defined():
      return ops.convert_to_tensor_v2_with_dispatch(
          self.shape.as_list(), dtype=dtypes.int32, name="shape")

    domain_dimension = sum(self._block_domain_dimension_tensors())
    range_dimension = sum(self._block_range_dimension_tensors())
    matrix_shape = array_ops.stack([domain_dimension, range_dimension])

    batch_shape = self.operators[0][0].batch_shape_tensor()
    for row in self.operators[1:]:
      for operator in row:
        batch_shape = array_ops.broadcast_dynamic_shape(
            batch_shape, operator.batch_shape_tensor())

    return array_ops.concat((batch_shape, matrix_shape), 0)

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
      x: `LinearOperator`, `Tensor` with compatible shape and same `dtype` as
        `self`, or a blockwise iterable of `LinearOperator`s or `Tensor`s. See
        class docstring for definition of shape compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is
        the hermitian transpose (transposition and complex conjugation).
      name:  A name for this `Op`.

    Returns:
      A `LinearOperator` or `Tensor` with shape `[..., M, R]` and same `dtype`
        as `self`, or if `x` is blockwise, a list of `Tensor`s with shapes that
        concatenate to `[..., M, R]`.
    """
    if isinstance(x, linear_operator.LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = x.adjoint() if adjoint_arg else x

      if (right_operator.range_dimension is not None and
          left_operator.domain_dimension is not None and
          right_operator.range_dimension != left_operator.domain_dimension):
        raise ValueError(
            "Operators are incompatible. Expected `x` to have dimension"
            " {} but got {}.".format(
                left_operator.domain_dimension, right_operator.range_dimension))
      with self._name_scope(name):  # pylint: disable=not-callable
        return linear_operator_algebra.matmul(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      arg_dim = -1 if adjoint_arg else -2
      block_dimensions = (self._block_range_dimensions() if adjoint
                          else self._block_domain_dimensions())
      if linear_operator_util.arg_is_blockwise(block_dimensions, x, arg_dim):
        for i, block in enumerate(x):
          if not isinstance(block, linear_operator.LinearOperator):
            block = ops.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[arg_dim])
            x[i] = block
      else:
        x = ops.convert_to_tensor_v2_with_dispatch(x, name="x")
        self._check_input_dtype(x)
        op_dimension = (self.range_dimension if adjoint
                        else self.domain_dimension)
        op_dimension.assert_is_compatible_with(x.shape[arg_dim])
      return self._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    arg_dim = -1 if adjoint_arg else -2
    block_dimensions = (self._block_range_dimensions() if adjoint
                        else self._block_domain_dimensions())
    blockwise_arg = linear_operator_util.arg_is_blockwise(
        block_dimensions, x, arg_dim)
    if blockwise_arg:
      split_x = x
    else:
      split_dim = -1 if adjoint_arg else -2
      # Split input by columns if adjoint_arg is True, else rows
      split_x = linear_operator_util.split_arg_into_blocks(
          self._block_domain_dimensions(),
          self._block_domain_dimension_tensors,
          x, axis=split_dim)

    result_list = []
    # Iterate over row-partitions (i.e. column-partitions of the adjoint).
    if adjoint:
      for index in range(len(self.operators)):
        # Begin with the operator on the diagonal and apply it to the
        # respective `rhs` block.
        result = self.operators[index][index].matmul(
            split_x[index], adjoint=adjoint, adjoint_arg=adjoint_arg)

        # Iterate top to bottom over the operators in the remainder of the
        # column-partition (i.e. left to right over the row-partition of the
        # adjoint), apply the operator to the respective `rhs` block and
        # accumulate the sum. For example, given the
        # `LinearOperatorBlockLowerTriangular`:
        #
        # op = [[A, 0, 0],
        #       [B, C, 0],
        #       [D, E, F]]
        #
        # if `index = 1`, the following loop calculates:
        # `y_1 = (C.matmul(x_1, adjoint=adjoint) +
        #         E.matmul(x_2, adjoint=adjoint)`,
        # where `x_1` and `x_2` are splits of `x`.
        for j in range(index + 1, len(self.operators)):
          result += self.operators[j][index].matmul(
              split_x[j], adjoint=adjoint, adjoint_arg=adjoint_arg)
        result_list.append(result)
    else:
      for row in self.operators:
        # Begin with the left-most operator in the row-partition and apply it
        # to the first `rhs` block.
        result = row[0].matmul(
            split_x[0], adjoint=adjoint, adjoint_arg=adjoint_arg)
        # Iterate left to right over the operators in the remainder of the row
        # partition, apply the operator to the respective `rhs` block, and
        # accumulate the sum.
        for j, operator in enumerate(row[1:]):
          result += operator.matmul(
              split_x[j + 1], adjoint=adjoint, adjoint_arg=adjoint_arg)
        result_list.append(result)

    if blockwise_arg:
      return result_list

    result_list = linear_operator_util.broadcast_matrix_batch_dims(
        result_list)
    return array_ops.concat(result_list, axis=-2)

  def matvec(self, x, adjoint=False, name="matvec"):
    """Transform [batch] vector `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)

    X = ... # shape [..., N], batch vector

    Y = operator.matvec(X)
    Y.shape
    ==> [..., M]

    Y[..., :] = sum_j A[..., :, j] X[..., j]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`, or an
        iterable of `Tensor`s. `Tensor`s are treated a [batch] vectors, meaning
        for every set of leading dimensions, the last dimension defines a
        vector.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      name:  A name for this `Op`.

    Returns:
      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      block_dimensions = (self._block_range_dimensions() if adjoint
                          else self._block_domain_dimensions())
      if linear_operator_util.arg_is_blockwise(block_dimensions, x, -1):
        for i, block in enumerate(x):
          if not isinstance(block, linear_operator.LinearOperator):
            block = ops.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[-1])
            x[i] = block
        x_mat = [block[..., array_ops.newaxis] for block in x]
        y_mat = self.matmul(x_mat, adjoint=adjoint)
        return [array_ops.squeeze(y, axis=-1) for y in y_mat]

      x = ops.convert_to_tensor_v2_with_dispatch(x, name="x")
      self._check_input_dtype(x)
      op_dimension = (self.range_dimension if adjoint
                      else self.domain_dimension)
      op_dimension.assert_is_compatible_with(x.shape[-1])
      x_mat = x[..., array_ops.newaxis]
      y_mat = self.matmul(x_mat, adjoint=adjoint)
      return array_ops.squeeze(y_mat, axis=-1)

  def _determinant(self):
    if all(op.is_positive_definite for op in self._diagonal_operators):
      return math_ops.exp(self._log_abs_determinant())
    result = self._diagonal_operators[0].determinant()
    for op in self._diagonal_operators[1:]:
      result *= op.determinant()
    return result

  def _log_abs_determinant(self):
    result = self._diagonal_operators[0].log_abs_determinant()
    for op in self._diagonal_operators[1:]:
      result += op.log_abs_determinant()
    return result

  def solve(self, rhs, adjoint=False, adjoint_arg=False, name="solve"):
    """Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Given the blockwise `n + 1`-by-`n + 1` linear operator:

    op = [[A_00     0  ...     0  ...    0],
          [A_10  A_11  ...     0  ...    0],
          ...
          [A_k0  A_k1  ...  A_kk  ...    0],
          ...
          [A_n0  A_n1  ...  A_nk  ... A_nn]]

    we find `x = op.solve(y)` by observing that

    `y_k = A_k0.matmul(x_0) + A_k1.matmul(x_1) + ... + A_kk.matmul(x_k)`

    and therefore

    `x_k = A_kk.solve(y_k -
                      A_k0.matmul(x_0) - ... - A_k(k-1).matmul(x_(k-1)))`

    where `x_k` and `y_k` are the `k`th blocks obtained by decomposing `x`
    and `y` along their appropriate axes.

    We first solve `x_0 = A_00.solve(y_0)`. Proceeding inductively, we solve
    for `x_k`, `k = 1..n`, given `x_0..x_(k-1)`.

    The adjoint case is solved similarly, beginning with
    `x_n = A_nn.solve(y_n, adjoint=True)` and proceeding backwards.

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
      rhs: `Tensor` with same `dtype` as this operator and compatible shape,
        or a list of `Tensor`s. `Tensor`s are treated like a [batch] matrices
        meaning for every set of leading dimensions, the last two dimensions
        defines a matrix.
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
    if isinstance(rhs, linear_operator.LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = rhs.adjoint() if adjoint_arg else rhs

      if (right_operator.range_dimension is not None and
          left_operator.domain_dimension is not None and
          right_operator.range_dimension != left_operator.domain_dimension):
        raise ValueError(
            "Operators are incompatible. Expected `rhs` to have dimension"
            " {} but got {}.".format(
                left_operator.domain_dimension, right_operator.range_dimension))
      with self._name_scope(name):  # pylint: disable=not-callable
        return linear_operator_algebra.solve(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      block_dimensions = (self._block_domain_dimensions() if adjoint
                          else self._block_range_dimensions())
      arg_dim = -1 if adjoint_arg else -2
      blockwise_arg = linear_operator_util.arg_is_blockwise(
          block_dimensions, rhs, arg_dim)
      if blockwise_arg:
        for i, block in enumerate(rhs):
          if not isinstance(block, linear_operator.LinearOperator):
            block = ops.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[arg_dim])
            rhs[i] = block
        if adjoint_arg:
          split_rhs = [linalg.adjoint(y) for y in rhs]
        else:
          split_rhs = rhs

      else:
        rhs = ops.convert_to_tensor_v2_with_dispatch(rhs, name="rhs")
        self._check_input_dtype(rhs)
        op_dimension = (self.domain_dimension if adjoint
                        else self.range_dimension)
        op_dimension.assert_is_compatible_with(rhs.shape[arg_dim])

        rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
        split_rhs = linear_operator_util.split_arg_into_blocks(
            self._block_domain_dimensions(),
            self._block_domain_dimension_tensors,
            rhs, axis=-2)

      solution_list = []
      if adjoint:
        # For an adjoint blockwise lower-triangular linear operator, the system
        # must be solved bottom to top. Iterate backwards over rows of the
        # adjoint (i.e. columns of the non-adjoint operator).
        for index in reversed(range(len(self.operators))):
          y = split_rhs[index]
          # Iterate top to bottom over the operators in the off-diagonal portion
          # of the column-partition (i.e. row-partition of the adjoint), apply
          # the operator to the respective block of the solution found in
          # previous iterations, and subtract the result from the `rhs` block.
          # For example,let `A`, `B`, and `D` be the linear operators in the top
          # row-partition of the adjoint of
          # `LinearOperatorBlockLowerTriangular([[A], [B, C], [D, E, F]])`,
          # and `x_1` and `x_2` be blocks of the solution found in previous
          # iterations of the outer loop. The following loop (when `index == 0`)
          # expresses
          # `Ax_0 + Bx_1 + Dx_2 = y_0` as `Ax_0 = y_0*`, where
          # `y_0* = y_0 - Bx_1 - Dx_2`.
          for j in reversed(range(index + 1, len(self.operators))):
            y = y - self.operators[j][index].matmul(
                solution_list[len(self.operators) - 1 - j],
                adjoint=adjoint)
          # Continuing the example above, solve `Ax_0 = y_0*` for `x_0`.
          solution_list.append(
              self._diagonal_operators[index].solve(y, adjoint=adjoint))
        solution_list.reverse()
      else:
        # Iterate top to bottom over the row-partitions.
        for row, y in zip(self.operators, split_rhs):
          # Iterate left to right over the operators in the off-diagonal portion
          # of the row-partition, apply the operator to the block of the
          # solution found in previous iterations, and subtract the result from
          # the `rhs` block. For example, let `D`, `E`, and `F` be the linear
          # operators in the bottom row-partition of
          # `LinearOperatorBlockLowerTriangular([[A], [B, C], [D, E, F]])` and
          # `x_0` and `x_1` be blocks of the solution found in previous
          # iterations of the outer loop. The following loop
          # (when `index == 2`), expresses
          # `Dx_0 + Ex_1 + Fx_2 = y_2` as `Fx_2 = y_2*`, where
          # `y_2* = y_2 - D_x0 - Ex_1`.
          for i, operator in enumerate(row[:-1]):
            y = y - operator.matmul(solution_list[i], adjoint=adjoint)
          # Continuing the example above, solve `Fx_2 = y_2*` for `x_2`.
          solution_list.append(row[-1].solve(y, adjoint=adjoint))

      if blockwise_arg:
        return solution_list

      solution_list = linear_operator_util.broadcast_matrix_batch_dims(
          solution_list)
      return array_ops.concat(solution_list, axis=-2)

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
      rhs: `Tensor` with same `dtype` as this operator, or list of `Tensor`s
        (for blockwise operators). `Tensor`s are treated as [batch] vectors,
        meaning for every set of leading dimensions, the last dimension defines
        a vector.  See class docstring for definition of compatibility regarding
        batch dimensions.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      block_dimensions = (self._block_domain_dimensions() if adjoint
                          else self._block_range_dimensions())
      if linear_operator_util.arg_is_blockwise(block_dimensions, rhs, -1):
        for i, block in enumerate(rhs):
          if not isinstance(block, linear_operator.LinearOperator):
            block = ops.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[-1])
            rhs[i] = block
        rhs_mat = [array_ops.expand_dims(block, axis=-1) for block in rhs]
        solution_mat = self.solve(rhs_mat, adjoint=adjoint)
        return [array_ops.squeeze(x, axis=-1) for x in solution_mat]
      rhs = ops.convert_to_tensor_v2_with_dispatch(rhs, name="rhs")
      self._check_input_dtype(rhs)
      op_dimension = (self.domain_dimension if adjoint
                      else self.range_dimension)
      op_dimension.assert_is_compatible_with(rhs.shape[-1])
      rhs_mat = array_ops.expand_dims(rhs, axis=-1)
      solution_mat = self.solve(rhs_mat, adjoint=adjoint)
      return array_ops.squeeze(solution_mat, axis=-1)

  def _diag_part(self):
    diag_list = []
    for op in self._diagonal_operators:
      # Extend the axis, since `broadcast_matrix_batch_dims` treats all but the
      # final two dimensions as batch dimensions.
      diag_list.append(op.diag_part()[..., array_ops.newaxis])
    diag_list = linear_operator_util.broadcast_matrix_batch_dims(diag_list)
    diagonal = array_ops.concat(diag_list, axis=-2)
    return array_ops.squeeze(diagonal, axis=-1)

  def _trace(self):
    result = self._diagonal_operators[0].trace()
    for op in self._diagonal_operators[1:]:
      result += op.trace()
    return result

  def _to_dense(self):
    num_cols = 0
    dense_rows = []
    flat_broadcast_operators = linear_operator_util.broadcast_matrix_batch_dims(
        [op.to_dense() for row in self.operators for op in row])  # pylint: disable=g-complex-comprehension
    broadcast_operators = [
        flat_broadcast_operators[i * (i + 1) // 2:(i + 1) * (i + 2) // 2]
        for i in range(len(self.operators))]
    for row_blocks in broadcast_operators:
      batch_row_shape = array_ops.shape(row_blocks[0])[:-1]
      num_cols += array_ops.shape(row_blocks[-1])[-1]
      zeros_to_pad_after_shape = array_ops.concat(
          [batch_row_shape,
           [self.domain_dimension_tensor() - num_cols]], axis=-1)
      zeros_to_pad_after = array_ops.zeros(
          shape=zeros_to_pad_after_shape, dtype=self.dtype)

      row_blocks.append(zeros_to_pad_after)
      dense_rows.append(array_ops.concat(row_blocks, axis=-1))

    mat = array_ops.concat(dense_rows, axis=-2)
    mat.set_shape(self.shape)
    return mat

  def _assert_non_singular(self):
    return control_flow_ops.group([
        op.assert_non_singular() for op in self._diagonal_operators])

  def _eigvals(self):
    eig_list = []
    for op in self._diagonal_operators:
      # Extend the axis for broadcasting.
      eig_list.append(op.eigvals()[..., array_ops.newaxis])
    eig_list = linear_operator_util.broadcast_matrix_batch_dims(eig_list)
    eigs = array_ops.concat(eig_list, axis=-2)
    return array_ops.squeeze(eigs, axis=-1)

  @property
  def _composite_tensor_fields(self):
    return ("operators",)

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
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorBlockLowerTriangular"]


@tf_export("linalg.LinearOperatorBlockLowerTriangular")
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
    # Validate operators.
    check_ops.assert_proper_iterable(operators)
    for row in operators:
      check_ops.assert_proper_iterable(row)
    operators = [list(row) for row in operators]

    if not operators:
      raise ValueError(
          "Expected a non-empty list of operators. Found: {}".format(operators))
    self._operators = operators

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
          name=name)

  def _validate_num_operators(self):
    for i, row in enumerate(self.operators):
      if len(row) != i + 1:
        raise ValueError(
            "The `i`th row-partition (`i`th element of `operators`) must "
            "contain `i` blocks (`LinearOperator` instances). Row {} contains "
            "{} blocks.".format(i + 1, len(row)))

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
            raise ValueError(
                "Operator domain dimensions {} and {} must be equal to fit a "
                "blockwise structure.".format(
                    op.domain_dimension, above_op.domain_dimension))
        if (op.range_dimension is not None and
            right_op.range_dimension is not None):
          if op.range_dimension != right_op.range_dimension:
            raise ValueError(
                "Operator range dimensions {} and {} must be equal to fit a "
                "blockwise structure.".format(
                    op.range_dimension, right_op.range_dimension))

  # pylint: disable=g-bool-id-comparison
  def _validate_non_singular(self, is_non_singular):
    if all(row[-1].is_non_singular for row in self.operators):
      if is_non_singular is False:
        raise ValueError(
            "A blockwise lower-triangular operator with non-singular operators "
            " on the main diagonal is always non-singular.")
      return True
    if any(row[-1].is_non_singular is False for row in self.operators):
      if is_non_singular is True:
        raise ValueError(
            "A blockwise lower-triangular operator with a singular operator on "
            "the main diagonal is always singular.")
      return False

  def _validate_square(self, is_square):
    if is_square is False:
      raise ValueError("`LinearOperatorBlockLowerTriangular` must be square.")
    if any(row[-1].is_square is False for row in self.operators):
      raise ValueError(
          "Matrices on the diagonal (the final elements of each row-partition "
          "in the `operators` list) must be square.")
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

  def _shape(self):
    # Get final matrix shape.
    domain_dimension = self.operators[0][0].domain_dimension
    range_dimension = self.operators[0][0].range_dimension
    for row in self.operators[1:]:
      domain_dimension += row[-1].domain_dimension
      range_dimension += row[-1].range_dimension

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
      return ops.convert_to_tensor(
          self.shape.as_list(), dtype=dtypes.int32, name="shape")

    domain_dimension = self.operators[0][0].domain_dimension_tensor()
    range_dimension = self.operators[0][0].range_dimension_tensor()

    for row in self.operators[1:]:
      domain_dimension += row[-1].domain_dimension_tensor()
      range_dimension += row[-1].range_dimension_tensor()

    matrix_shape = array_ops.stack([domain_dimension, range_dimension])

    batch_shape = self.operators[0][0].batch_shape_tensor()
    for row in self.operators[1:]:
      for operator in row:
        batch_shape = array_ops.broadcast_dynamic_shape(
            batch_shape, operator.batch_shape_tensor())

    return array_ops.concat((batch_shape, matrix_shape), 0)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    split_dim = -1 if adjoint_arg else -2
    # Split input by columns if adjoint_arg is True, else rows
    split_x = self._split_input_into_blocks(x, axis=split_dim)

    result_list = []
    # Iterate over row-partitions (i.e. column-partitions of the adjoint).
    if adjoint:
      for index in range(len(self.operators)):
        # Begin with the operator on the diagonal and apply it to the respective
        # `rhs` block.
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
        # Begin with the left-most operator in the row-partition and apply it to
        # the first `rhs` block.
        result = row[0].matmul(
            split_x[0], adjoint=adjoint, adjoint_arg=adjoint_arg)
        # Iterate left to right over the operators in the remainder of the row
        # partition, apply the operator to the respective `rhs` block, and
        # accumulate the sum.
        for j, operator in enumerate(row[1:]):
          result += operator.matmul(
              split_x[j + 1], adjoint=adjoint, adjoint_arg=adjoint_arg)
        result_list.append(result)

    result_list = linear_operator_util.broadcast_matrix_batch_dims(
        result_list)
    return array_ops.concat(result_list, axis=-2)

  def _determinant(self):
    if all(row[-1].is_positive_definite for row in self.operators):
      return math_ops.exp(self._log_abs_determinant())
    result = self.operators[0][0].determinant()
    for row in self.operators[1:]:
      result *= row[-1].determinant()
    return result

  def _log_abs_determinant(self):
    result = self.operators[0][0].log_abs_determinant()
    for row in self.operators[1:]:
      result += row[-1].log_abs_determinant()
    return result

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # Given the blockwise `n + 1`-by-`n + 1` linear operator:
    #
    # op = [[A_00     0  ...     0  ...    0],
    #       [A_10  A_11  ...     0  ...    0],
    #       ...
    #       [A_k0  A_k1  ...  A_kk  ...    0],
    #       ...
    #       [A_n0  A_n1  ...  A_nk  ... A_nn]]
    #
    # we find `x = op.solve(y)` by observing that
    #
    # `y_k = A_k0.matmul(x_0) + A_k1.matmul(x_1) + ... + A_kk.matmul(x_k)`
    #
    # and therefore
    #
    # `x_k = A_kk.solve(y_k -
    #                   A_k0.matmul(x_0) - ... - A_k(k-1).matmul(x_(k-1)))`
    #
    # where `x_k` and `y_k` are the `k`th blocks obtained by decomposing `x`
    # and `y` along their appropriate axes.
    #
    # We first solve `x_0 = A_00.solve(y_0)`. Proceeding inductively, we solve
    # for `x_k`, `k = 1..n`, given `x_0..x_(k-1)`.
    #
    # The adjoint case is solved similarly, beginning with
    # `x_n = A_nn.solve(y_n, adjoint=True)` and proceeding backwards.
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    split_rhs = self._split_input_into_blocks(rhs, axis=-2)

    solution_list = []
    if adjoint:
      # For an adjoint blockwise lower-triangular linear operator, the system
      # must be solved bottom to top. Iterate backwards over rows of the adjoint
      # (i.e. columns of the non-adjoint operator).
      for index in reversed(range(len(self.operators))):
        y = split_rhs[index]
        # Iterate top to bottom over the operators in the off-diagonal portion
        # of the column-partition (i.e. row-partition of the adjoint), apply
        # the operator to the respective block of the solution found in previous
        # iterations, and subtract the result from the `rhs` block. For example,
        # let `A`, `B`, and `D` be the linear operators in the top row-partition
        # of the adjoint of
        # `LinearOperatorBlockLowerTriangular([[A], [B, C], [D, E, F]])`,
        # and `x_1` and `x_2` be blocks of the solution found in previous
        # iterations of the outer loop. The following loop (when `index == 0`)
        # expresses
        # `Ax_0 + Bx_1 + Dx_2 = y_0` as `Ax_0 = y_0*`, where
        # `y_0* = y_0 - Bx_1 - Dx_2`.
        for j in reversed(range(index + 1, len(self.operators))):
          y -= self.operators[j][index].matmul(
              solution_list[len(self.operators) - 1 - j],
              adjoint=adjoint)
        # Continuing the example above, solve `Ax_0 = y_0*` for `x_0`.
        solution_list.append(
            self.operators[index][index].solve(y, adjoint=adjoint))
      solution_list.reverse()
    else:
      # Iterate top to bottom over the row-partitions.
      for row, y in zip(self.operators, split_rhs):
        # Iterate left to right over the operators in the off-diagonal portion
        # of the row-partition, apply the operator to the block of the solution
        # found in previous iterations, and subtract the result from the `rhs`
        # block. For example, let `D`, `E`, and `F` be the linear operators in
        # the bottom row-partition of
        # `LinearOperatorBlockLowerTriangular([[A], [B, C], [D, E, F]])` and
        # `x_0` and `x_1` be blocks of the solution found in previous iterations
        # of the outer loop. The following loop (when `index == 2`), expresses
        # `Dx_0 + Ex_1 + Fx_2 = y_2` as `Fx_2 = y_2*`, where
        # `y_2* = y_2 - D_x0 - Ex_1`.
        for i, operator in enumerate(row[:-1]):
          y -= operator.matmul(solution_list[i], adjoint=adjoint)
        # Continuing the example above, solve `Fx_2 = y_2*` for `x_2`.
        solution_list.append(row[-1].solve(y, adjoint=adjoint))

    solution_list = linear_operator_util.broadcast_matrix_batch_dims(
        solution_list)
    return array_ops.concat(solution_list, axis=-2)

  def _diag_part(self):
    diag_list = []
    for row in self.operators:
      # Extend the axis, since `broadcast_matrix_batch_dims` treats all but the
      # final two dimensions as batch dimensions.
      diag_list.append(row[-1].diag_part()[..., array_ops.newaxis])
    diag_list = linear_operator_util.broadcast_matrix_batch_dims(diag_list)
    diagonal = array_ops.concat(diag_list, axis=-2)
    return array_ops.squeeze(diagonal, axis=-1)

  def _trace(self):
    result = self.operators[0][0].trace()
    for row in self.operators[1:]:
      result += row[-1].trace()
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
        row[-1].assert_non_singular() for row in self.operators])

  def _eigvals(self):
    eig_list = []
    for row in self.operators:
      # Extend the axis for broadcasting.
      eig_list.append(row[-1].eigvals()[..., array_ops.newaxis])
    eig_list = linear_operator_util.broadcast_matrix_batch_dims(eig_list)
    eigs = array_ops.concat(eig_list, axis=-2)
    return array_ops.squeeze(eigs, axis=-1)

  def _split_input_into_blocks(self, x, axis=-1):
    """Split `x` into blocks matching `operators`'s `domain_dimension`.

    Specifically, if we have a blockwise lower-triangular matrix, with block
    sizes along the diagonal `[M_j, M_j] j = 0,1,2..J`,  this method splits `x`
    on `axis` into `J` tensors, whose shape at `axis` is `M_j`.

    Args:
      x: `Tensor`. `x` is split into `J` tensors.
      axis: Python `Integer` representing the axis to split `x` on.

    Returns:
      A list of `Tensor`s.
    """
    block_sizes = []
    if self.shape.is_fully_defined():
      for row in self.operators:
        block_sizes.append(row[-1].domain_dimension.value)
    else:
      for row in self.operators:
        block_sizes.append(row[-1].domain_dimension_tensor())

    return array_ops.split(x, block_sizes, axis=axis)

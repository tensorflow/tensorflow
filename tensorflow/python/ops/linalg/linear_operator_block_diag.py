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
"""Create a Block Diagonal operator from one or more `LinearOperators`."""

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorBlockDiag"]


@tf_export("linalg.LinearOperatorBlockDiag")
@linear_operator.make_composite_tensor
class LinearOperatorBlockDiag(linear_operator.LinearOperator):
  """Combines one or more `LinearOperators` in to a Block Diagonal matrix.

  This operator combines one or more linear operators `[op1,...,opJ]`,
  building a new `LinearOperator`, whose underlying matrix representation
  has each operator `opi` on the main diagonal, and zero's elsewhere.

  #### Shape compatibility

  If `opj` acts like a [batch] matrix `Aj`, then `op_combined` acts like
  the [batch] matrix formed by having each matrix `Aj` on the main
  diagonal.

  Each `opj` is required to represent a matrix, and hence will have
  shape `batch_shape_j + [M_j, N_j]`.

  If `opj` has shape `batch_shape_j + [M_j, N_j]`, then the combined operator
  has shape `broadcast_batch_shape + [sum M_j, sum N_j]`, where
  `broadcast_batch_shape` is the mutual broadcast of `batch_shape_j`,
  `j = 1,...,J`, assuming the intermediate batch shapes broadcast.

  Arguments to `matmul`, `matvec`, `solve`, and `solvevec` may either be single
  `Tensor`s or lists of `Tensor`s that are interpreted as blocks. The `j`th
  element of a blockwise list of `Tensor`s must have dimensions that match
  `opj` for the given method. If a list of blocks is input, then a list of
  blocks is returned as well.

  When the `opj` are not guaranteed to be square, this operator's methods might
  fail due to the combined operator not being square and/or lack of efficient
  methods.

  ```python
  # Create a 4 x 4 linear operator combined of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
  operator = LinearOperatorBlockDiag([operator_1, operator_2])

  operator.to_dense()
  ==> [[1., 2., 0., 0.],
       [3., 4., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]]

  operator.shape
  ==> [4, 4]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x1 = ... # Shape [2, 2] Tensor
  x2 = ... # Shape [2, 2] Tensor
  x = tf.concat([x1, x2], 0)  # Shape [2, 4] Tensor
  operator.matmul(x)
  ==> tf.concat([operator_1.matmul(x1), operator_2.matmul(x2)])

  # Create a 5 x 4 linear operator combining three blocks.
  operator_1 = LinearOperatorFullMatrix([[1.], [3.]])
  operator_2 = LinearOperatorFullMatrix([[1., 6.]])
  operator_3 = LinearOperatorFullMatrix([[2.], [7.]])
  operator = LinearOperatorBlockDiag([operator_1, operator_2, operator_3])

  operator.to_dense()
  ==> [[1., 0., 0., 0.],
       [3., 0., 0., 0.],
       [0., 1., 6., 0.],
       [0., 0., 0., 2.]]
       [0., 0., 0., 7.]]

  operator.shape
  ==> [5, 4]


  # Create a [2, 3] batch of 4 x 4 linear operators.
  matrix_44 = tf.random.normal(shape=[2, 3, 4, 4])
  operator_44 = LinearOperatorFullMatrix(matrix)

  # Create a [1, 3] batch of 5 x 5 linear operators.
  matrix_55 = tf.random.normal(shape=[1, 3, 5, 5])
  operator_55 = LinearOperatorFullMatrix(matrix_55)

  # Combine to create a [2, 3] batch of 9 x 9 operators.
  operator_99 = LinearOperatorBlockDiag([operator_44, operator_55])

  # Create a shape [2, 3, 9] vector.
  x = tf.random.normal(shape=[2, 3, 9])
  operator_99.matmul(x)
  ==> Shape [2, 3, 9] Tensor

  # Create a blockwise list of vectors.
  x = [tf.random.normal(shape=[2, 3, 4]), tf.random.normal(shape=[2, 3, 5])]
  operator_99.matmul(x)
  ==> [Shape [2, 3, 4] Tensor, Shape [2, 3, 5] Tensor]
  ```

  #### Performance

  The performance of `LinearOperatorBlockDiag` on any operation is equal to
  the sum of the individual operators' operations.


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
               is_square=True,
               name=None):
    r"""Initialize a `LinearOperatorBlockDiag`.

    `LinearOperatorBlockDiag` is initialized with a list of operators
    `[op_1,...,op_J]`.

    Args:
      operators:  Iterable of `LinearOperator` objects, each with
        the same `dtype` and composable shape.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
        This is true by default, and will raise a `ValueError` otherwise.
      name: A name for this `LinearOperator`.  Default is the individual
        operators names joined with `_o_`.

    Raises:
      TypeError:  If all operators do not have the same `dtype`.
      ValueError:  If `operators` is empty or are non-square.
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
    operators = list(operators)
    if not operators:
      raise ValueError(
          "Expected a non-empty list of operators. Found: %s" % operators)
    self._operators = operators

    # Define diagonal operators, for functions that are shared across blockwise
    # `LinearOperator` types.
    self._diagonal_operators = operators

    # Validate dtype.
    dtype = operators[0].dtype
    for operator in operators:
      if operator.dtype != dtype:
        name_type = (str((o.name, o.dtype)) for o in operators)
        raise TypeError(
            "Expected all operators to have the same dtype.  Found %s"
            % "   ".join(name_type))

    # Auto-set and check hints.
    if all(operator.is_non_singular for operator in operators):
      if is_non_singular is False:
        raise ValueError(
            "The direct sum of non-singular operators is always non-singular.")
      is_non_singular = True

    if all(operator.is_self_adjoint for operator in operators):
      if is_self_adjoint is False:
        raise ValueError(
            "The direct sum of self-adjoint operators is always self-adjoint.")
      is_self_adjoint = True

    if all(operator.is_positive_definite for operator in operators):
      if is_positive_definite is False:
        raise ValueError(
            "The direct sum of positive definite operators is always "
            "positive definite.")
      is_positive_definite = True

    # Initialization.
    graph_parents = []
    for operator in operators:
      graph_parents.extend(operator.graph_parents)

    if name is None:
      # Using ds to mean direct sum.
      name = "_ds_".join(operator.name for operator in operators)
    with ops.name_scope(name, values=graph_parents):
      super(LinearOperatorBlockDiag, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

    # TODO(b/143910018) Remove graph_parents in V3.
    self._set_graph_parents(graph_parents)

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
    matrix_shape = tensor_shape.TensorShape([range_dimension, domain_dimension])

    # Get broadcast batch shape.
    # broadcast_shape checks for compatibility.
    batch_shape = self.operators[0].batch_shape
    for operator in self.operators[1:]:
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
    matrix_shape = array_ops.stack([range_dimension, domain_dimension])

    # Dummy Tensor of zeros.  Will never be materialized.
    zeros = array_ops.zeros(shape=self.operators[0].batch_shape_tensor())
    for operator in self.operators[1:]:
      zeros += array_ops.zeros(shape=operator.batch_shape_tensor())
    batch_shape = array_ops.shape(zeros)

    return array_ops.concat((batch_shape, matrix_shape), 0)

  # TODO(b/188080761): Add a more efficient implementation of `cond` that
  # constructs the condition number from the blockwise singular values.

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
    def _check_operators_agree(r, l, message):
      if (r.range_dimension is not None and
          l.domain_dimension is not None and
          r.range_dimension != l.domain_dimension):
        raise ValueError(message)

    if isinstance(x, linear_operator.LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = x.adjoint() if adjoint_arg else x

      _check_operators_agree(
          right_operator, left_operator,
          "Operators are incompatible. Expected `x` to have dimension"
          " {} but got {}.".format(
              left_operator.domain_dimension, right_operator.range_dimension))

      # We can efficiently multiply BlockDiag LinearOperators if the number of
      # blocks agree.
      if isinstance(x, LinearOperatorBlockDiag):
        if len(left_operator.operators) != len(right_operator.operators):
          raise ValueError(
              "Can not efficiently multiply two `LinearOperatorBlockDiag`s "
              "together when number of blocks differ.")

        for o1, o2 in zip(left_operator.operators, right_operator.operators):
          _check_operators_agree(
              o2, o1,
              "Blocks are incompatible. Expected `x` to have dimension"
              " {} but got {}.".format(
                  o1.domain_dimension, o2.range_dimension))

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
    block_dimensions_fn = (
        self._block_range_dimension_tensors if adjoint
        else self._block_domain_dimension_tensors)
    blockwise_arg = linear_operator_util.arg_is_blockwise(
        block_dimensions, x, arg_dim)
    if blockwise_arg:
      split_x = x

    else:
      split_dim = -1 if adjoint_arg else -2
      # Split input by rows normally, and otherwise columns.
      split_x = linear_operator_util.split_arg_into_blocks(
          block_dimensions, block_dimensions_fn, x, axis=split_dim)

    result_list = []
    for index, operator in enumerate(self.operators):
      result_list += [operator.matmul(
          split_x[index], adjoint=adjoint, adjoint_arg=adjoint_arg)]

    if blockwise_arg:
      return result_list

    result_list = linear_operator_util.broadcast_matrix_batch_dims(
        result_list)
    return array_ops.concat(result_list, axis=-2)

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
      x: `Tensor` with compatible shape and same `dtype` as `self`, or an
        iterable of `Tensor`s (for blockwise operators). `Tensor`s are treated
        a [batch] vectors, meaning for every set of leading dimensions, the last
        dimension defines a vector.
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
    result = self.operators[0].determinant()
    for operator in self.operators[1:]:
      result *= operator.determinant()
    return result

  def _log_abs_determinant(self):
    result = self.operators[0].log_abs_determinant()
    for operator in self.operators[1:]:
      result += operator.log_abs_determinant()
    return result

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
      rhs: `Tensor` with same `dtype` as this operator and compatible shape,
        or a list of `Tensor`s (for blockwise operators). `Tensor`s are treated
        like a [batch] matrices meaning for every set of leading dimensions, the
        last two dimensions defines a matrix.
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

    def _check_operators_agree(r, l, message):
      if (r.range_dimension is not None and
          l.domain_dimension is not None and
          r.range_dimension != l.domain_dimension):
        raise ValueError(message)

    if isinstance(rhs, linear_operator.LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = rhs.adjoint() if adjoint_arg else rhs

      _check_operators_agree(
          right_operator, left_operator,
          "Operators are incompatible. Expected `x` to have dimension"
          " {} but got {}.".format(
              left_operator.domain_dimension, right_operator.range_dimension))

      # We can efficiently solve BlockDiag LinearOperators if the number of
      # blocks agree.
      if isinstance(right_operator, LinearOperatorBlockDiag):
        if len(left_operator.operators) != len(right_operator.operators):
          raise ValueError(
              "Can not efficiently solve `LinearOperatorBlockDiag` when "
              "number of blocks differ.")

        for o1, o2 in zip(left_operator.operators, right_operator.operators):
          _check_operators_agree(
              o2, o1,
              "Blocks are incompatible. Expected `x` to have dimension"
              " {} but got {}.".format(
                  o1.domain_dimension, o2.range_dimension))

      with self._name_scope(name):  # pylint: disable=not-callable
        return linear_operator_algebra.solve(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      block_dimensions = (self._block_domain_dimensions() if adjoint
                          else self._block_range_dimensions())
      arg_dim = -1 if adjoint_arg else -2
      blockwise_arg = linear_operator_util.arg_is_blockwise(
          block_dimensions, rhs, arg_dim)

      if blockwise_arg:
        split_rhs = rhs
        for i, block in enumerate(split_rhs):
          if not isinstance(block, linear_operator.LinearOperator):
            block = ops.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[arg_dim])
            split_rhs[i] = block
      else:
        rhs = ops.convert_to_tensor_v2_with_dispatch(rhs, name="rhs")
        self._check_input_dtype(rhs)
        op_dimension = (self.domain_dimension if adjoint
                        else self.range_dimension)
        op_dimension.assert_is_compatible_with(rhs.shape[arg_dim])
        split_dim = -1 if adjoint_arg else -2
        # Split input by rows normally, and otherwise columns.
        split_rhs = linear_operator_util.split_arg_into_blocks(
            self._block_domain_dimensions(),
            self._block_domain_dimension_tensors,
            rhs, axis=split_dim)

      solution_list = []
      for index, operator in enumerate(self.operators):
        solution_list += [operator.solve(
            split_rhs[index], adjoint=adjoint, adjoint_arg=adjoint_arg)]

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
    if not all(operator.is_square for operator in self.operators):
      raise NotImplementedError(
          "`diag_part` not implemented for an operator whose blocks are not "
          "square.")
    diag_list = []
    for operator in self.operators:
      # Extend the axis for broadcasting.
      diag_list += [operator.diag_part()[..., array_ops.newaxis]]
    diag_list = linear_operator_util.broadcast_matrix_batch_dims(diag_list)
    diagonal = array_ops.concat(diag_list, axis=-2)
    return array_ops.squeeze(diagonal, axis=-1)

  def _trace(self):
    if not all(operator.is_square for operator in self.operators):
      raise NotImplementedError(
          "`trace` not implemented for an operator whose blocks are not "
          "square.")
    result = self.operators[0].trace()
    for operator in self.operators[1:]:
      result += operator.trace()
    return result

  def _to_dense(self):
    num_cols = 0
    rows = []
    broadcasted_blocks = [operator.to_dense() for operator in self.operators]
    broadcasted_blocks = linear_operator_util.broadcast_matrix_batch_dims(
        broadcasted_blocks)
    for block in broadcasted_blocks:
      batch_row_shape = array_ops.shape(block)[:-1]

      zeros_to_pad_before_shape = array_ops.concat(
          [batch_row_shape, [num_cols]], axis=-1)
      zeros_to_pad_before = array_ops.zeros(
          shape=zeros_to_pad_before_shape, dtype=block.dtype)
      num_cols += array_ops.shape(block)[-1]
      zeros_to_pad_after_shape = array_ops.concat(
          [batch_row_shape,
           [self.domain_dimension_tensor() - num_cols]], axis=-1)
      zeros_to_pad_after = array_ops.zeros(
          shape=zeros_to_pad_after_shape, dtype=block.dtype)

      rows.append(array_ops.concat(
          [zeros_to_pad_before, block, zeros_to_pad_after], axis=-1))

    mat = array_ops.concat(rows, axis=-2)
    mat.set_shape(self.shape)
    return mat

  def _assert_non_singular(self):
    return control_flow_ops.group([
        operator.assert_non_singular() for operator in self.operators])

  def _assert_self_adjoint(self):
    return control_flow_ops.group([
        operator.assert_self_adjoint() for operator in self.operators])

  def _assert_positive_definite(self):
    return control_flow_ops.group([
        operator.assert_positive_definite() for operator in self.operators])

  def _eigvals(self):
    if not all(operator.is_square for operator in self.operators):
      raise NotImplementedError(
          "`eigvals` not implemented for an operator whose blocks are not "
          "square.")
    eig_list = []
    for operator in self.operators:
      # Extend the axis for broadcasting.
      eig_list += [operator.eigvals()[..., array_ops.newaxis]]
    eig_list = linear_operator_util.broadcast_matrix_batch_dims(eig_list)
    eigs = array_ops.concat(eig_list, axis=-2)
    return array_ops.squeeze(eigs, axis=-1)

  @property
  def _composite_tensor_fields(self):
    return ("operators",)

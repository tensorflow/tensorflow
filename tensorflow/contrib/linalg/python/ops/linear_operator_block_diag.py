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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util


class LinearOperatorBlockDiag(linear_operator.LinearOperator):
  """Combines one or more `LinearOperators` in to a Block Diagonal matrix.

  This operator combines one or more linear operators `[op1,...,opJ]`,
  building a new `LinearOperator`, whose underlying matrix representation is
  square and has each operator `opi` on the main diagonal, and zero's elsewhere.

  #### Shape compatibility

  If `opj` acts like a [batch] square matrix `Aj`, then `op_combined` acts like
  the [batch] square matrix formed by having each matrix `Aj` on the main
  diagonal.


  Each `opj` is required to represent a square matrix, and hence will have
  shape `batch_shape_j + [M_j, M_j]`.

  If `opj` has shape `batch_shape_j + [M_j, M_j]`, then the combined operator
  has shape `broadcast_batch_shape + [sum M_j, sum M_j]`, where
  `broadcast_batch_shape` is the mutual broadcast of `batch_shape_j`,
  `j = 1,...,J`, assuming the intermediate batch shapes broadcast.
  Even if the combined shape is well defined, the combined operator's
  methods may fail due to lack of broadcasting ability in the defining
  operators' methods.

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

  # Create a [2, 3] batch of 4 x 4 linear operators.
  matrix_44 = tf.random_normal(shape=[2, 3, 4, 4])
  operator_44 = LinearOperatorFullMatrix(matrix)

  # Create a [1, 3] batch of 5 x 5 linear operators.
  matrix_55 = tf.random_normal(shape=[1, 3, 5, 5])
  operator_55 = LinearOperatorFullMatrix(matrix_55)

  # Combine to create a [2, 3] batch of 9 x 9 operators.
  operator_99 = LinearOperatorBlockDiag([operator_44, operator_55])

  # Create a shape [2, 3, 9] vector.
  x = tf.random_normal(shape=[2, 3, 9])
  operator_99.matmul(x)
  ==> Shape [2, 3, 9] Tensor
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
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
        This is true by default, and will raise a `ValueError` otherwise.
      name: A name for this `LinearOperator`.  Default is the individual
        operators names joined with `_o_`.

    Raises:
      TypeError:  If all operators do not have the same `dtype`.
      ValueError:  If `operators` is empty or are non-square.
    """
    # Validate operators.
    check_ops.assert_proper_iterable(operators)
    operators = list(operators)
    if not operators:
      raise ValueError(
          "Expected a non-empty list of operators. Found: %s" % operators)
    self._operators = operators

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

    if not (is_square and all(operator.is_square for operator in operators)):
      raise ValueError(
          "Can only represent a block diagonal of square matrices.")

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
          graph_parents=graph_parents,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True,
          name=name)

  @property
  def operators(self):
    return self._operators

  def _shape(self):
    # Get final matrix shape.
    domain_dimension = self.operators[0].domain_dimension
    range_dimension = self.operators[0].range_dimension
    for operator in self.operators[1:]:
      domain_dimension += operator.domain_dimension
      range_dimension += operator.range_dimension

    matrix_shape = tensor_shape.TensorShape([domain_dimension, range_dimension])

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
      return ops.convert_to_tensor(
          self.shape.as_list(), dtype=dtypes.int32, name="shape")

    domain_dimension = self.operators[0].domain_dimension_tensor()
    range_dimension = self.operators[0].range_dimension_tensor()
    for operator in self.operators[1:]:
      domain_dimension += operator.domain_dimension_tensor()
      range_dimension += operator.range_dimension_tensor()

    matrix_shape = array_ops.stack([domain_dimension, range_dimension])

    # Dummy Tensor of zeros.  Will never be materialized.
    zeros = array_ops.zeros(shape=self.operators[0].batch_shape_tensor())
    for operator in self.operators[1:]:
      zeros += array_ops.zeros(shape=operator.batch_shape_tensor())
    batch_shape = array_ops.shape(zeros)

    return array_ops.concat((batch_shape, matrix_shape), 0)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    split_dim = -1 if adjoint_arg else -2
    # Split input by rows normally, and otherwise columns.
    split_x = self._split_input_into_blocks(x, axis=split_dim)

    result_list = []
    for index, operator in enumerate(self.operators):
      result_list += [operator.matmul(
          split_x[index], adjoint=adjoint, adjoint_arg=adjoint_arg)]
    result_list = linear_operator_util.broadcast_matrix_batch_dims(
        result_list)
    return array_ops.concat(result_list, axis=-2)

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

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    split_dim = -1 if adjoint_arg else -2
    # Split input by rows normally, and otherwise columns.
    split_rhs = self._split_input_into_blocks(rhs, axis=split_dim)

    solution_list = []
    for index, operator in enumerate(self.operators):
      solution_list += [operator.solve(
          split_rhs[index], adjoint=adjoint, adjoint_arg=adjoint_arg)]

    solution_list = linear_operator_util.broadcast_matrix_batch_dims(
        solution_list)
    return array_ops.concat(solution_list, axis=-2)

  def _diag_part(self):
    diag_list = []
    for operator in self.operators:
      # Extend the axis for broadcasting.
      diag_list += [operator.diag_part()[..., array_ops.newaxis]]
    diag_list = linear_operator_util.broadcast_matrix_batch_dims(diag_list)
    diagonal = array_ops.concat(diag_list, axis=-2)
    return array_ops.squeeze(diagonal, axis=-1)

  def _trace(self):
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

  def _split_input_into_blocks(self, x, axis=-1):
    """Split `x` into blocks matching `operators`'s `domain_dimension`.

    Specifically, if we have a block diagonal matrix, with block sizes
    `[M_j, M_j] j = 1..J`,  this method splits `x` on `axis` into `J`
    tensors, whose shape at `axis` is `M_j`.

    Args:
      x: `Tensor`. `x` is split into `J` tensors.
      axis: Python `Integer` representing the axis to split `x` on.

    Returns:
      A list of `Tensor`s.
    """
    block_sizes = []
    if self.shape.is_fully_defined():
      for operator in self.operators:
        block_sizes += [operator.domain_dimension.value]
    else:
      for operator in self.operators:
        block_sizes += [operator.domain_dimension_tensor()]

    return array_ops.split(x, block_sizes, axis=axis)

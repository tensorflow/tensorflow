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
"""Construct the Kronecker product of one or more `LinearOperators`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "LinearOperatorKronecker",
]


def _vec(x):
  """Stacks column of matrix to form a single column."""
  return array_ops.reshape(
      array_ops.matrix_transpose(x),
      array_ops.concat(
          [array_ops.shape(x)[:-2], [-1]], axis=0))


def _unvec_by(y, num_col):
  """Unstack vector to form a matrix, with a specified amount of columns."""
  return array_ops.matrix_transpose(
      array_ops.reshape(
          y,
          array_ops.concat(
              [array_ops.shape(y)[:-1], [num_col, -1]], axis=0)))


def _rotate_last_dim(x, rotate_right=False):
  """Rotate the last dimension either left or right."""
  ndims = array_ops.rank(x)
  if rotate_right:
    transpose_perm = array_ops.concat(
        [[ndims - 1], math_ops.range(0, ndims - 1)], axis=0)
  else:
    transpose_perm = array_ops.concat(
        [math_ops.range(1, ndims), [0]], axis=0)
  return array_ops.transpose(x, transpose_perm)


@tf_export("linalg.LinearOperatorKronecker")
class LinearOperatorKronecker(linear_operator.LinearOperator):
  """Kronecker product between two `LinearOperators`.

  This operator composes one or more linear operators `[op1,...,opJ]`,
  building a new `LinearOperator` representing the Kronecker product:
  `op1 x op2 x .. opJ` (we omit parentheses as the Kronecker product is
  associative).

  If `opj` has shape `batch_shape_j` + [M_j, N_j`, then the composed operator
  will have shape equal to `broadcast_batch_shape + [prod M_j, prod N_j]`,
  where the product is over all operators.

  ```python
  # Create a 4 x 4 linear operator composed of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [2., 1.]])
  operator = LinearOperatorKronecker([operator_1, operator_2])

  operator.to_dense()
  ==> [[1., 2., 0., 0.],
       [3., 4., 0., 0.],
       [2., 4., 1., 2.],
       [6., 8., 3., 4.]]

  operator.shape
  ==> [4, 4]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [4, 2] Tensor
  operator.matmul(x)
  ==> Shape [4, 2] Tensor

  # Create a [2, 3] batch of 4 x 5 linear operators.
  matrix_45 = tf.random_normal(shape=[2, 3, 4, 5])
  operator_45 = LinearOperatorFullMatrix(matrix)

  # Create a [2, 3] batch of 5 x 6 linear operators.
  matrix_56 = tf.random_normal(shape=[2, 3, 5, 6])
  operator_56 = LinearOperatorFullMatrix(matrix_56)

  # Compose to create a [2, 3] batch of 20 x 30 operators.
  operator_large = LinearOperatorKronecker([operator_45, operator_56])

  # Create a shape [2, 3, 20, 2] vector.
  x = tf.random_normal(shape=[2, 3, 6, 2])
  operator_large.matmul(x)
  ==> Shape [2, 3, 30, 2] Tensor
  ```

  #### Performance

  The performance of `LinearOperatorKronecker` on any operation is equal to
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
               is_square=None,
               name=None):
    r"""Initialize a `LinearOperatorKronecker`.

    `LinearOperatorKronecker` is initialized with a list of operators
    `[op_1,...,op_J]`.

    Args:
      operators:  Iterable of `LinearOperator` objects, each with
        the same `dtype` and composable shape, representing the Kronecker
        factors.
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
      name: A name for this `LinearOperator`.  Default is the individual
        operators names joined with `_x_`.

    Raises:
      TypeError:  If all operators do not have the same `dtype`.
      ValueError:  If `operators` is empty.
    """
    # Validate operators.
    check_ops.assert_proper_iterable(operators)
    operators = list(operators)
    if not operators:
      raise ValueError(
          "Expected a list of >=1 operators. Found: %s" % operators)
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
    # A Kronecker product is invertible, if and only if all factors are
    # invertible.
    if all(operator.is_non_singular for operator in operators):
      if is_non_singular is False:
        raise ValueError(
            "The Kronecker product of non-singular operators is always "
            "non-singular.")
      is_non_singular = True

    if all(operator.is_self_adjoint for operator in operators):
      if is_self_adjoint is False:
        raise ValueError(
            "The Kronecker product of self-adjoint operators is always "
            "self-adjoint.")
      is_self_adjoint = True

    # The eigenvalues of a Kronecker product are equal to the products of eigen
    # values of the corresponding factors.
    if all(operator.is_positive_definite for operator in operators):
      if is_positive_definite is False:
        raise ValueError("The Kronecker product of positive-definite operators "
                         "is always positive-definite.")
      is_positive_definite = True

    # Initialization.
    graph_parents = []
    for operator in operators:
      graph_parents.extend(operator.graph_parents)

    if name is None:
      name = operators[0].name
      for operator in operators[1:]:
        name += "_x_" + operator.name
    with ops.name_scope(name, values=graph_parents):
      super(LinearOperatorKronecker, self).__init__(
          dtype=dtype,
          graph_parents=graph_parents,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          name=name)

  @property
  def operators(self):
    return self._operators

  def _shape(self):
    # Get final matrix shape.
    domain_dimension = self.operators[0].domain_dimension
    for operator in self.operators[1:]:
      domain_dimension *= operator.domain_dimension

    range_dimension = self.operators[0].range_dimension
    for operator in self.operators[1:]:
      range_dimension *= operator.range_dimension

    matrix_shape = tensor_shape.TensorShape([
        range_dimension, domain_dimension])

    # Get broadcast batch shape.
    # broadcast_shape checks for compatibility.
    batch_shape = self.operators[0].batch_shape
    for operator in self.operators[1:]:
      batch_shape = common_shapes.broadcast_shape(
          batch_shape, operator.batch_shape)

    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    domain_dimension = self.operators[0].domain_dimension_tensor()
    for operator in self.operators[1:]:
      domain_dimension *= operator.domain_dimension_tensor()

    range_dimension = self.operators[0].range_dimension_tensor()
    for operator in self.operators[1:]:
      range_dimension *= operator.range_dimension_tensor()

    matrix_shape = [range_dimension, domain_dimension]

    # Get broadcast batch shape.
    # broadcast_shape checks for compatibility.
    batch_shape = self.operators[0].batch_shape_tensor()
    for operator in self.operators[1:]:
      batch_shape = array_ops.broadcast_dynamic_shape(
          batch_shape, operator.batch_shape_tensor())

    return array_ops.concat((batch_shape, matrix_shape), 0)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # Here we heavily rely on Roth's column Lemma [1]:
    # (A x B) * vec X = vec BXA^T,
    # where vec stacks all the columns of the matrix under each other. In our
    # case, x represents a batch of vec X (i.e. we think of x as a batch of
    # column vectors, rather than a matrix). Each member of the batch can be
    # reshaped to a matrix (hence we get a batch of matrices).
    # We can iteratively apply this lemma by noting that if B is a Kronecker
    # product, then we can apply the lemma again.

    # [1] W. E. Roth, "On direct product matrices,"
    # Bulletin of the American Mathematical Society, vol. 40, pp. 461-468,
    # 1934

    # Efficiency

    # Naively doing the Kronecker product, by calculating the dense matrix and
    # applying it will can take cubic time in  the size of domain_dimension
    # (assuming a square matrix). The other issue is that calculating the dense
    # matrix can be prohibitively expensive, in that it can take a large amount
    # of memory.
    #
    # This implementation avoids this memory blow up by only computing matmuls
    # with the factors. In this way, we don't have to realize the dense matrix.
    # In terms of complexity, if we have Kronecker Factors of size:
    # (n1, n1), (n2, n2), (n3, n3), ... (nJ, nJ), with N = \prod n_i, and we
    # have as input a [N, M] matrix, the naive approach would take O(N^2 M).
    # With this approach (ignoring reshaping of tensors and transposes for now),
    # the time complexity can be O(M * (\sum n_i) * N). There is also the
    # benefit of batched multiplication (In this example, the batch size is
    # roughly M * N) so this can be much faster. However, not factored in are
    # the costs of the several transposing of tensors, which can affect cache
    # behavior.

    # Below we document the shape manipulation for adjoint=False,
    # adjoint_arg=False, but the general case of different adjoints is still
    # handled.

    if adjoint_arg:
      x = linalg.adjoint(x)

    # Always add a batch dimension to enable broadcasting to work.
    batch_shape = array_ops.concat(
        [array_ops.ones_like(self.batch_shape_tensor()), [1, 1]], 0)
    x += array_ops.zeros(batch_shape, dtype=x.dtype.base_dtype)

    # x has shape [B, R, C], where B represent some number of batch dimensions,
    # R represents the number of rows, and C represents the number of columns.
    # In order to apply Roth's column lemma, we need to operate on a batch of
    # column vectors, so we reshape into a batch of column vectors. We put it
    # at the front to ensure that broadcasting between operators to the batch
    # dimensions B still works.
    output = _rotate_last_dim(x, rotate_right=True)

    # Also expand the shape to be [A, C, B, R]. The first dimension will be
    # used to accumulate dimensions from each operator matmul.
    output = output[array_ops.newaxis, ...]

    # In this loop, A is going to refer to the value of the accumulated
    # dimension. A = 1 at the start, and will end up being self.range_dimension.
    # V will refer to the last dimension. V = R at the start, and will end up
    # being 1 in the end.
    for operator in self.operators[:-1]:
      # Reshape output from [A, C, B, V] to be
      # [A, C, B, V / op.domain_dimension, op.domain_dimension]
      if adjoint:
        operator_dimension = operator.range_dimension_tensor()
      else:
        operator_dimension = operator.domain_dimension_tensor()

      output = _unvec_by(output, operator_dimension)

      # We are computing (XA^T) = (AX^T)^T.
      # output has [A, C, B, V / op.domain_dimension, op.domain_dimension],
      # which is being converted to:
      # [A, C, B, V / op.domain_dimension, op.range_dimension]
      output = array_ops.matrix_transpose(output)
      output = operator.matmul(output, adjoint=adjoint, adjoint_arg=False)
      output = array_ops.matrix_transpose(output)
      # Rearrange it to [A * op.range_dimension, C, B, V / op.domain_dimension]
      output = _rotate_last_dim(output, rotate_right=False)
      output = _vec(output)
      output = _rotate_last_dim(output, rotate_right=True)

    # After the loop, we will have
    # A = self.range_dimension / op[-1].range_dimension
    # V = op[-1].domain_dimension

    # We convert that using matvec to get:
    # [A, C, B, op[-1].range_dimension]
    output = self.operators[-1].matvec(output, adjoint=adjoint)
    # Rearrange shape to be [B1, ... Bn, self.range_dimension, C]
    output = _rotate_last_dim(output, rotate_right=False)
    output = _vec(output)
    output = _rotate_last_dim(output, rotate_right=False)

    if x.shape.is_fully_defined():
      column_dim = x.shape[-1]
      broadcast_batch_shape = common_shapes.broadcast_shape(
          x.shape[:-2], self.batch_shape)
      if adjoint:
        matrix_dimensions = [self.domain_dimension, column_dim]
      else:
        matrix_dimensions = [self.range_dimension, column_dim]

      output.set_shape(broadcast_batch_shape.concatenate(
          matrix_dimensions))

    return output

  def _determinant(self):
    # Note that we have |X1 x X2| = |X1| ** n * |X2| ** m, where X1 is an m x m
    # matrix, and X2 is an n x n matrix. We can iteratively apply this property
    # to get the determinant of |X1 x X2 x X3 ...|. If T is the product of the
    # domain dimension of all operators, then we have:
    # |X1 x X2 x X3 ...| =
    #    |X1| ** (T / m) * |X2 x X3 ... | ** m =
    #    |X1| ** (T / m) * |X2| ** (m * (T / m) / n) *  ... =
    #    |X1| ** (T / m) * |X2| ** (T / n) * | X3 x X4... | ** (m * n)
    #    And by doing induction we have product(|X_i| ** (T / dim(X_i))).
    total = self.domain_dimension_tensor()
    determinant = 1.
    for operator in self.operators:
      determinant *= operator.determinant() ** math_ops.cast(
          total / operator.domain_dimension_tensor(),
          dtype=operator.dtype)
    return determinant

  def _log_abs_determinant(self):
    # This will be sum((total / dim(x_i)) * log |X_i|)
    total = self.domain_dimension_tensor()
    log_abs_det = 0.
    for operator in self.operators:
      log_abs_det += operator.log_abs_determinant() * math_ops.cast(
          total / operator.domain_dimension_tensor(),
          dtype=operator.dtype)
    return log_abs_det

  def _trace(self):
    # tr(A x B) = tr(A) * tr(B)
    trace = 1.
    for operator in self.operators:
      trace *= operator.trace()
    return trace

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # Here we follow the same use of Roth's column lemma as in `matmul`, with
    # the key difference that we replace all `matmul` instances with `solve`.
    # This follows from the property that inv(A x B) = inv(A) x inv(B).

    # Below we document the shape manipulation for adjoint=False,
    # adjoint_arg=False, but the general case of different adjoints is still
    # handled.

    if adjoint_arg:
      rhs = linalg.adjoint(rhs)

    # Always add a batch dimension to enable broadcasting to work.
    batch_shape = array_ops.concat(
        [array_ops.ones_like(self.batch_shape_tensor()), [1, 1]], 0)
    rhs += array_ops.zeros(batch_shape, dtype=rhs.dtype.base_dtype)

    # rhs has shape [B, R, C], where B represent some number of batch
    # dimensions,
    # R represents the number of rows, and C represents the number of columns.
    # In order to apply Roth's column lemma, we need to operate on a batch of
    # column vectors, so we reshape into a batch of column vectors. We put it
    # at the front to ensure that broadcasting between operators to the batch
    # dimensions B still works.
    output = _rotate_last_dim(rhs, rotate_right=True)

    # Also expand the shape to be [A, C, B, R]. The first dimension will be
    # used to accumulate dimensions from each operator matmul.
    output = output[array_ops.newaxis, ...]

    # In this loop, A is going to refer to the value of the accumulated
    # dimension. A = 1 at the start, and will end up being self.range_dimension.
    # V will refer to the last dimension. V = R at the start, and will end up
    # being 1 in the end.
    for operator in self.operators[:-1]:
      # Reshape output from [A, C, B, V] to be
      # [A, C, B, V / op.domain_dimension, op.domain_dimension]
      if adjoint:
        operator_dimension = operator.range_dimension_tensor()
      else:
        operator_dimension = operator.domain_dimension_tensor()

      output = _unvec_by(output, operator_dimension)

      # We are computing (XA^-1^T) = (A^-1 X^T)^T.
      # output has [A, C, B, V / op.domain_dimension, op.domain_dimension],
      # which is being converted to:
      # [A, C, B, V / op.domain_dimension, op.range_dimension]
      output = array_ops.matrix_transpose(output)
      output = operator.solve(output, adjoint=adjoint, adjoint_arg=False)
      output = array_ops.matrix_transpose(output)
      # Rearrange it to [A * op.range_dimension, C, B, V / op.domain_dimension]
      output = _rotate_last_dim(output, rotate_right=False)
      output = _vec(output)
      output = _rotate_last_dim(output, rotate_right=True)

    # After the loop, we will have
    # A = self.range_dimension / op[-1].range_dimension
    # V = op[-1].domain_dimension

    # We convert that using matvec to get:
    # [A, C, B, op[-1].range_dimension]
    output = self.operators[-1].solvevec(output, adjoint=adjoint)
    # Rearrange shape to be [B1, ... Bn, self.range_dimension, C]
    output = _rotate_last_dim(output, rotate_right=False)
    output = _vec(output)
    output = _rotate_last_dim(output, rotate_right=False)

    if rhs.shape.is_fully_defined():
      column_dim = rhs.shape[-1]
      broadcast_batch_shape = common_shapes.broadcast_shape(
          rhs.shape[:-2], self.batch_shape)
      if adjoint:
        matrix_dimensions = [self.domain_dimension, column_dim]
      else:
        matrix_dimensions = [self.range_dimension, column_dim]

      output.set_shape(broadcast_batch_shape.concatenate(
          matrix_dimensions))

    return output

  def _diag_part(self):
    diag_part = self.operators[0].diag_part()
    for operator in self.operators[1:]:
      diag_part = diag_part[..., :, array_ops.newaxis]
      op_diag_part = operator.diag_part()[..., array_ops.newaxis, :]
      diag_part *= op_diag_part
      diag_part = array_ops.reshape(
          diag_part,
          shape=array_ops.concat(
              [array_ops.shape(diag_part)[:-2], [-1]], axis=0))
    if self.range_dimension > self.domain_dimension:
      diag_dimension = self.domain_dimension
    else:
      diag_dimension = self.range_dimension
    diag_part.set_shape(
        self.batch_shape.concatenate(diag_dimension))
    return diag_part

  def _to_dense(self):
    product = self.operators[0].to_dense()
    for operator in self.operators[1:]:
      # Product has shape [B, R1, 1, C1].
      product = product[
          ..., :, array_ops.newaxis, :, array_ops.newaxis]
      # Operator has shape [B, 1, R2, 1, C2].
      op_to_mul = operator.to_dense()[
          ..., array_ops.newaxis, :, array_ops.newaxis, :]
      # This is now [B, R1, R2, C1, C2].
      product *= op_to_mul
      # Now merge together dimensions to get [B, R1 * R2, C1 * C2].
      product = array_ops.reshape(
          product,
          shape=array_ops.concat(
              [array_ops.shape(product)[:-4],
               [array_ops.shape(product)[-4] * array_ops.shape(product)[-3],
                array_ops.shape(product)[-2] * array_ops.shape(product)[-1]]
              ], axis=0))
    product.set_shape(self.shape)
    return product

  def _assert_non_singular(self):
    if all(operator.is_square for operator in self.operators):
      asserts = [operator.assert_non_singular() for operator in self.operators]
      return control_flow_ops.group(asserts)
    else:
      raise errors.InvalidArgumentError(
          node_def=None, op=None, message="All Kronecker factors must be "
          "square for the product to be invertible.")

  def _assert_self_adjoint(self):
    if all(operator.is_square for operator in self.operators):
      asserts = [operator.assert_self_adjoint() for operator in self.operators]
      return control_flow_ops.group(asserts)
    else:
      raise errors.InvalidArgumentError(
          node_def=None, op=None, message="All Kronecker factors must be "
          "square for the product to be self adjoint.")

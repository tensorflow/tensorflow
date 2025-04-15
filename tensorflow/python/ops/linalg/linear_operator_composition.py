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
"""Composes one or more `LinearOperators`."""

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorComposition"]


@tf_export("linalg.LinearOperatorComposition")
@linear_operator.make_composite_tensor
class LinearOperatorComposition(linear_operator.LinearOperator):
  """Composes one or more `LinearOperators`.

  This operator composes one or more linear operators `[op1,...,opJ]`,
  building a new `LinearOperator` with action defined by:

  ```
  op_composed(x) := op1(op2(...(opJ(x)...))
  ```

  If `opj` acts like [batch] matrix `Aj`, then `op_composed` acts like the
  [batch] matrix formed with the multiplication `A1 A2...AJ`.

  If `opj` has shape `batch_shape_j + [M_j, N_j]`, then we must have
  `N_j = M_{j+1}`, in which case the composed operator has shape equal to
  `broadcast_batch_shape + [M_1, N_J]`, where `broadcast_batch_shape` is the
  mutual broadcast of `batch_shape_j`, `j = 1,...,J`, assuming the intermediate
  batch shapes broadcast.  Even if the composed shape is well defined, the
  composed operator's methods may fail due to lack of broadcasting ability in
  the defining operators' methods.

  ```python
  # Create a 2 x 2 linear operator composed of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
  operator = LinearOperatorComposition([operator_1, operator_2])

  operator.to_dense()
  ==> [[1., 2.]
       [3., 4.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 5 linear operators.
  matrix_45 = tf.random.normal(shape=[2, 3, 4, 5])
  operator_45 = LinearOperatorFullMatrix(matrix)

  # Create a [2, 3] batch of 5 x 6 linear operators.
  matrix_56 = tf.random.normal(shape=[2, 3, 5, 6])
  operator_56 = LinearOperatorFullMatrix(matrix_56)

  # Compose to create a [2, 3] batch of 4 x 6 operators.
  operator_46 = LinearOperatorComposition([operator_45, operator_56])

  # Create a shape [2, 3, 6, 2] vector.
  x = tf.random.normal(shape=[2, 3, 6, 2])
  operator.matmul(x)
  ==> Shape [2, 3, 4, 2] Tensor
  ```

  #### Performance

  The performance of `LinearOperatorComposition` on any operation is equal to
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
    r"""Initialize a `LinearOperatorComposition`.

    `LinearOperatorComposition` is initialized with a list of operators
    `[op_1,...,op_J]`.  For the `matmul` method to be well defined, the
    composition `op_i.matmul(op_{i+1}(x))` must be defined.  Other methods have
    similar constraints.

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
      name: A name for this `LinearOperator`.  Default is the individual
        operators names joined with `_o_`.

    Raises:
      TypeError:  If all operators do not have the same `dtype`.
      ValueError:  If `operators` is empty.
    """
    parameters = dict(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)

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
      if is_non_singular is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError(
            "The composition of non-singular operators is always non-singular.")
      is_non_singular = True

    if _composition_must_be_self_adjoint(operators):
      if is_self_adjoint is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError(
            "The composition was determined to be self-adjoint but user "
            "provided incorrect `False` hint.")
      is_self_adjoint = True

    if linear_operator_util.is_aat_form(operators):
      if is_square is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError(
            "The composition was determined have the form "
            "A @ A.H, hence it must be square. The user "
            "provided an incorrect `False` hint.")
      is_square = True

    if linear_operator_util.is_aat_form(operators) and is_non_singular:
      if is_positive_definite is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError(
            "The composition was determined to be non-singular and have the "
            "form A @ A.H, hence it must be positive-definite. The user "
            "provided an incorrect `False` hint.")
      is_positive_definite = True

    # Initialization.

    if name is None:
      name = "_o_".join(operator.name for operator in operators)
    with ops.name_scope(name):
      super(LinearOperatorComposition, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  @property
  def operators(self):
    return self._operators

  def _shape(self):
    # Get final matrix shape.
    domain_dimension = self.operators[0].domain_dimension
    for operator in self.operators[1:]:
      domain_dimension.assert_is_compatible_with(operator.range_dimension)
      domain_dimension = operator.domain_dimension

    matrix_shape = tensor_shape.TensorShape(
        [self.operators[0].range_dimension,
         self.operators[-1].domain_dimension])

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

    # Don't check the matrix dimensions.  That would add unnecessary Asserts to
    # the graph.  Things will fail at runtime naturally if shapes are
    # incompatible.
    matrix_shape = array_ops_stack.stack([
        self.operators[0].range_dimension_tensor(),
        self.operators[-1].domain_dimension_tensor()
    ])

    # Dummy Tensor of zeros.  Will never be materialized.
    zeros = array_ops.zeros(shape=self.operators[0].batch_shape_tensor())
    for operator in self.operators[1:]:
      zeros += array_ops.zeros(shape=operator.batch_shape_tensor())
    batch_shape = array_ops.shape(zeros)

    return array_ops.concat((batch_shape, matrix_shape), 0)

  def _linop_cholesky(self) -> linear_operator.LinearOperator:
    """Computes Cholesky(LinearOperatorComposition)."""
    # L @ L.H will be handled with special code below. Why is L @ L.H the most
    # important special case?
    # Note that Diag @ Diag.H  and Diag @ TriL and TriL @ Diag are already
    # compressed to Diag or TriL by diag matmul
    # registration. Similarly for Identity and ScaledIdentity.
    # So these would not appear in a LinearOperatorComposition unless explicitly
    # constructed as such. So the most important thing to check is L @ L.H.
    def _is_llt_product(self):
      """Determines if linop = L @ L.H for L = LinearOperatorLowerTriangular."""
      if len(self.operators) != 2:
        return False
      if not linear_operator_util.is_aat_form(self.operators):
        return False
      return isinstance(
          self.operators[0],
          linear_operator_lower_triangular.LinearOperatorLowerTriangular)

    if not _is_llt_product(self):
      return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
          linalg_ops.cholesky(self.to_dense()),
          is_non_singular=True,
          is_self_adjoint=False,
          is_square=True)

    left_op = self.operators[0]

    # left_op.is_positive_definite ==> op already has positive diag,return it.
    if left_op.is_positive_definite:
      return left_op

    # Recall that the base class has already verified
    # linop.is_positive_definite, else linop.cholesky() would have raised.
    # So in particular, we know the diagonal has nonzero entries.
    # In the generic case, we make op have positive diag by dividing each row
    # by the sign of the diag. This is equivalent to setting A = L @ D where
    # D is diag(sign(1 / L.diag_part())). Then A is lower triangular with
    # positive diag and A @ A^H = L @ D @ D^H @ L^H = L @ L^H = linop.
    # This also works for complex L,
    # since sign(x + iy) = exp(i * angle(x + iy)).
    diag_sign = array_ops.expand_dims(
        math_ops.sign(left_op.diag_part()), axis=-2)
    return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
        tril=left_op.tril / diag_sign,
        is_non_singular=left_op.is_non_singular,
        # L.is_self_adjoint ==> L is diagonal ==> L @ D is diagonal ==> SA
        # L.is_self_adjoint is False ==> L not diagonal ==> L @ D not diag ...
        is_self_adjoint=left_op.is_self_adjoint,
        # L.is_positive_definite ==> L has positive diag ==> L = L @ D
        #   ==> (L @ D).is_positive_definite.
        # L.is_positive_definite is False could result
        # in L @ D being PD or not.
        # Consider L = [[1, 0], [-2, 1]] and quadratic form with x = [1, 1].
        # Note we will already return left_op if left_op.is_positive_definite
        # above, but to be explicit write this below.
        is_positive_definite=True if left_op.is_positive_definite else None,
        is_square=True,
    )

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # If self.operators = [A, B], and not adjoint, then
    # matmul_order_list = [B, A].
    # As a result, we return A.matmul(B.matmul(x))
    if adjoint:
      matmul_order_list = self.operators
    else:
      matmul_order_list = list(reversed(self.operators))

    result = matmul_order_list[0].matmul(
        x, adjoint=adjoint, adjoint_arg=adjoint_arg)
    for operator in matmul_order_list[1:]:
      result = operator.matmul(result, adjoint=adjoint)
    return result

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
    # TODO(langmore) Implement solve using solve_ls if some intermediate
    # operator maps to a high dimensional space.
    # In that case, an exact solve may still be possible.

    # If self.operators = [A, B], and not adjoint, then
    # solve_order_list = [A, B].
    # As a result, we return B.solve(A.solve(x))
    if adjoint:
      solve_order_list = list(reversed(self.operators))
    else:
      solve_order_list = self.operators

    solution = solve_order_list[0].solve(
        rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
    for operator in solve_order_list[1:]:
      solution = operator.solve(solution, adjoint=adjoint)
    return solution

  def _assert_non_singular(self):
    if all(operator.is_square for operator in self.operators):
      asserts = [operator.assert_non_singular() for operator in self.operators]
      return control_flow_ops.group(asserts)
    return super(LinearOperatorComposition, self)._assert_non_singular()

  @property
  def _composite_tensor_fields(self):
    return ("operators",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"operators": [0] * len(self.operators)}


def _composition_must_be_self_adjoint(operators):
  """Runs some checks to see if composition operators must be SA.

  Args:
    operators: List of LinearOperators.

  Returns:
    True if the composition must be SA. False if it is not SA OR if we did not
      determine whether the composition is SA.
  """
  if len(operators) == 1 and operators[0].is_self_adjoint:
    return True

  # Check for forms like A @ A.H or (A1 @ A2) @ (A2.H @ A1.H) or ...
  if linear_operator_util.is_aat_form(operators):
    return True

  # Done checking...could still be SA.
  # We may not catch some cases. E.g. (A @ I) @ A.H is SA, but is not AAT form.
  return False

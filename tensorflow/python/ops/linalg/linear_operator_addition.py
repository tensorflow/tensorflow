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
"""Add one or more `LinearOperators` efficiently."""

import abc

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular

__all__ = []


def add_operators(operators,
                  operator_name=None,
                  addition_tiers=None,
                  name=None):
  """Efficiently add one or more linear operators.

  Given operators `[A1, A2,...]`, this `Op` returns a possibly shorter list of
  operators `[B1, B2,...]` such that

  ```sum_k Ak.matmul(x) = sum_k Bk.matmul(x).```

  The operators `Bk` result by adding some of the `Ak`, as allowed by
  `addition_tiers`.

  Example of efficient adding of diagonal operators.

  ```python
  A1 = LinearOperatorDiag(diag=[1., 1.], name="A1")
  A2 = LinearOperatorDiag(diag=[2., 2.], name="A2")

  # Use two tiers, the first contains an Adder that returns Diag.  Since both
  # A1 and A2 are Diag, they can use this Adder.  The second tier will not be
  # used.
  addition_tiers = [
      [_AddAndReturnDiag()],
      [_AddAndReturnMatrix()]]
  B_list = add_operators([A1, A2], addition_tiers=addition_tiers)

  len(B_list)
  ==> 1

  B_list[0].__class__.__name__
  ==> 'LinearOperatorDiag'

  B_list[0].to_dense()
  ==> [[3., 0.],
       [0., 3.]]

  B_list[0].name
  ==> 'Add/A1__A2/'
  ```

  Args:
    operators:  Iterable of `LinearOperator` objects with same `dtype`, domain
      and range dimensions, and broadcastable batch shapes.
    operator_name:  String name for returned `LinearOperator`.  Defaults to
      concatenation of "Add/A__B/" that indicates the order of addition steps.
    addition_tiers:  List tiers, like `[tier_0, tier_1, ...]`, where `tier_i`
      is a list of `Adder` objects.  This function attempts to do all additions
      in tier `i` before trying tier `i + 1`.
    name:  A name for this `Op`.  Defaults to `add_operators`.

  Returns:
    Subclass of `LinearOperator`.  Class and order of addition may change as new
      (and better) addition strategies emerge.

  Raises:
    ValueError:  If `operators` argument is empty.
    ValueError:  If shapes are incompatible.
  """
  # Default setting
  if addition_tiers is None:
    addition_tiers = _DEFAULT_ADDITION_TIERS

  # Argument checking.
  check_ops.assert_proper_iterable(operators)
  operators = list(reversed(operators))
  if len(operators) < 1:
    raise ValueError(
        f"Argument `operators` must contain at least one operator. "
        f"Received: {operators}.")
  if not all(
      isinstance(op, linear_operator.LinearOperator) for op in operators):
    raise TypeError(
        f"Argument `operators` must contain only LinearOperator instances. "
        f"Received: {operators}.")
  _static_check_for_same_dimensions(operators)
  _static_check_for_broadcastable_batch_shape(operators)

  with ops.name_scope(name or "add_operators"):

    # Additions done in one of the tiers.  Try tier 0, 1,...
    ops_to_try_at_next_tier = list(operators)
    for tier in addition_tiers:
      ops_to_try_at_this_tier = ops_to_try_at_next_tier
      ops_to_try_at_next_tier = []
      while ops_to_try_at_this_tier:
        op1 = ops_to_try_at_this_tier.pop()
        op2, adder = _pop_a_match_at_tier(op1, ops_to_try_at_this_tier, tier)
        if op2 is not None:
          # Will try to add the result of this again at this same tier.
          new_operator = adder.add(op1, op2, operator_name)
          ops_to_try_at_this_tier.append(new_operator)
        else:
          ops_to_try_at_next_tier.append(op1)

    return ops_to_try_at_next_tier


def _pop_a_match_at_tier(op1, operator_list, tier):
  # Search from the back of list to the front in order to create nice default
  # order of operations.
  for i in range(1, len(operator_list) + 1):
    op2 = operator_list[-i]
    for adder in tier:
      if adder.can_add(op1, op2):
        return operator_list.pop(-i), adder
  return None, None


def _infer_hints_allowing_override(op1, op2, hints):
  """Infer hints from op1 and op2.  hints argument is an override.

  Args:
    op1:  LinearOperator
    op2:  LinearOperator
    hints:  _Hints object holding "is_X" boolean hints to use for returned
      operator.
      If some hint is None, try to set using op1 and op2.  If the
      hint is provided, ignore op1 and op2 hints.  This allows an override
      of previous hints, but does not allow forbidden hints (e.g. you still
      cannot say a real diagonal operator is not self-adjoint.

  Returns:
    _Hints object.
  """
  hints = hints or _Hints()
  # If A, B are self-adjoint, then so is A + B.
  if hints.is_self_adjoint is None:
    is_self_adjoint = op1.is_self_adjoint and op2.is_self_adjoint
  else:
    is_self_adjoint = hints.is_self_adjoint

  # If A, B are positive definite, then so is A + B.
  if hints.is_positive_definite is None:
    is_positive_definite = op1.is_positive_definite and op2.is_positive_definite
  else:
    is_positive_definite = hints.is_positive_definite

  # A positive definite operator is always non-singular.
  if is_positive_definite and hints.is_positive_definite is None:
    is_non_singular = True
  else:
    is_non_singular = hints.is_non_singular

  return _Hints(
      is_non_singular=is_non_singular,
      is_self_adjoint=is_self_adjoint,
      is_positive_definite=is_positive_definite)


def _static_check_for_same_dimensions(operators):
  """ValueError if operators determined to have different dimensions."""
  if len(operators) < 2:
    return

  domain_dimensions = [
      (op.name, tensor_shape.dimension_value(op.domain_dimension))
      for op in operators
      if tensor_shape.dimension_value(op.domain_dimension) is not None]
  if len(set(value for name, value in domain_dimensions)) > 1:
    raise ValueError(f"All `operators` must have the same `domain_dimension`. "
                     f"Received: {domain_dimensions}.")

  range_dimensions = [
      (op.name, tensor_shape.dimension_value(op.range_dimension))
      for op in operators
      if tensor_shape.dimension_value(op.range_dimension) is not None]
  if len(set(value for name, value in range_dimensions)) > 1:
    raise ValueError(f"All operators must have the same `range_dimension`. "
                     f"Received: {range_dimensions}.")


def _static_check_for_broadcastable_batch_shape(operators):
  """ValueError if operators determined to have non-broadcastable shapes."""
  if len(operators) < 2:
    return

  # This will fail if they cannot be broadcast together.
  batch_shape = operators[0].batch_shape
  for op in operators[1:]:
    batch_shape = array_ops.broadcast_static_shape(batch_shape, op.batch_shape)


class _Hints:
  """Holds 'is_X' flags that every LinearOperator is initialized with."""

  def __init__(self,
               is_non_singular=None,
               is_positive_definite=None,
               is_self_adjoint=None):
    self.is_non_singular = is_non_singular
    self.is_positive_definite = is_positive_definite
    self.is_self_adjoint = is_self_adjoint


################################################################################
# Classes to add two linear operators.
################################################################################


class _Adder(metaclass=abc.ABCMeta):
  """Abstract base class to add two operators.

  Each `Adder` acts independently, adding everything it can, paying no attention
  as to whether another `Adder` could have done the addition more efficiently.
  """

  @property
  def name(self):
    return self.__class__.__name__

  @abc.abstractmethod
  def can_add(self, op1, op2):
    """Returns `True` if this `Adder` can add `op1` and `op2`.  Else `False`."""
    pass

  @abc.abstractmethod
  def _add(self, op1, op2, operator_name, hints):
    # Derived classes can assume op1 and op2 have been validated, e.g. they have
    # the same dtype, and their domain/range dimensions match.
    pass

  def add(self, op1, op2, operator_name, hints=None):
    """Return new `LinearOperator` acting like `op1 + op2`.

    Args:
      op1:  `LinearOperator`
      op2:  `LinearOperator`, with `shape` and `dtype` such that adding to
        `op1` is allowed.
      operator_name:  `String` name to give to returned `LinearOperator`
      hints:  `_Hints` object.  Returned `LinearOperator` will be created with
        these hints.

    Returns:
      `LinearOperator`
    """
    updated_hints = _infer_hints_allowing_override(op1, op2, hints)

    if operator_name is None:
      operator_name = "Add/" + op1.name + "__" + op2.name + "/"

    scope_name = self.name
    if scope_name.startswith("_"):
      scope_name = scope_name[1:]
    with ops.name_scope(scope_name):
      return self._add(op1, op2, operator_name, updated_hints)


class _AddAndReturnScaledIdentity(_Adder):
  """Handles additions resulting in an Identity family member.

  The Identity (`LinearOperatorScaledIdentity`, `LinearOperatorIdentity`) family
  is closed under addition.  This `Adder` respects that, and returns an Identity
  """

  def can_add(self, op1, op2):
    types = {_type(op1), _type(op2)}
    return not types.difference(_IDENTITY_FAMILY)

  def _add(self, op1, op2, operator_name, hints):
    # Will build a LinearOperatorScaledIdentity.

    if _type(op1) == _SCALED_IDENTITY:
      multiplier_1 = op1.multiplier
    else:
      multiplier_1 = array_ops.ones(op1.batch_shape_tensor(), dtype=op1.dtype)

    if _type(op2) == _SCALED_IDENTITY:
      multiplier_2 = op2.multiplier
    else:
      multiplier_2 = array_ops.ones(op2.batch_shape_tensor(), dtype=op2.dtype)

    return linear_operator_identity.LinearOperatorScaledIdentity(
        num_rows=op1.range_dimension_tensor(),
        multiplier=multiplier_1 + multiplier_2,
        is_non_singular=hints.is_non_singular,
        is_self_adjoint=hints.is_self_adjoint,
        is_positive_definite=hints.is_positive_definite,
        name=operator_name)


class _AddAndReturnDiag(_Adder):
  """Handles additions resulting in a Diag operator."""

  def can_add(self, op1, op2):
    types = {_type(op1), _type(op2)}
    return not types.difference(_DIAG_LIKE)

  def _add(self, op1, op2, operator_name, hints):
    return linear_operator_diag.LinearOperatorDiag(
        diag=op1.diag_part() + op2.diag_part(),
        is_non_singular=hints.is_non_singular,
        is_self_adjoint=hints.is_self_adjoint,
        is_positive_definite=hints.is_positive_definite,
        name=operator_name)


class _AddAndReturnTriL(_Adder):
  """Handles additions resulting in a TriL operator."""

  def can_add(self, op1, op2):
    types = {_type(op1), _type(op2)}
    return not types.difference(_DIAG_LIKE.union({_TRIL}))

  def _add(self, op1, op2, operator_name, hints):
    if _type(op1) in _EFFICIENT_ADD_TO_TENSOR:
      op_add_to_tensor, op_other = op1, op2
    else:
      op_add_to_tensor, op_other = op2, op1

    return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
        tril=op_add_to_tensor.add_to_tensor(op_other.to_dense()),
        is_non_singular=hints.is_non_singular,
        is_self_adjoint=hints.is_self_adjoint,
        is_positive_definite=hints.is_positive_definite,
        name=operator_name)


class _AddAndReturnMatrix(_Adder):
  """"Handles additions resulting in a `LinearOperatorFullMatrix`."""

  def can_add(self, op1, op2):  # pylint: disable=unused-argument
    return isinstance(op1, linear_operator.LinearOperator) and isinstance(
        op2, linear_operator.LinearOperator)

  def _add(self, op1, op2, operator_name, hints):
    if _type(op1) in _EFFICIENT_ADD_TO_TENSOR:
      op_add_to_tensor, op_other = op1, op2
    else:
      op_add_to_tensor, op_other = op2, op1
    return linear_operator_full_matrix.LinearOperatorFullMatrix(
        matrix=op_add_to_tensor.add_to_tensor(op_other.to_dense()),
        is_non_singular=hints.is_non_singular,
        is_self_adjoint=hints.is_self_adjoint,
        is_positive_definite=hints.is_positive_definite,
        name=operator_name)


################################################################################
# Constants designating types of LinearOperators
################################################################################

# Type name constants for LinearOperator classes.
_IDENTITY = "identity"
_SCALED_IDENTITY = "scaled_identity"
_DIAG = "diag"
_TRIL = "tril"
_MATRIX = "matrix"

# Groups of operators.
_DIAG_LIKE = {_DIAG, _IDENTITY, _SCALED_IDENTITY}
_IDENTITY_FAMILY = {_IDENTITY, _SCALED_IDENTITY}
# operators with an efficient .add_to_tensor() method.
_EFFICIENT_ADD_TO_TENSOR = _DIAG_LIKE

# Supported LinearOperator classes.
SUPPORTED_OPERATORS = [
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_lower_triangular.LinearOperatorLowerTriangular,
    linear_operator_full_matrix.LinearOperatorFullMatrix,
    linear_operator_identity.LinearOperatorIdentity,
    linear_operator_identity.LinearOperatorScaledIdentity
]


def _type(operator):
  """Returns the type name constant (e.g. _TRIL) for operator."""
  if isinstance(operator, linear_operator_diag.LinearOperatorDiag):
    return _DIAG
  if isinstance(operator,
                linear_operator_lower_triangular.LinearOperatorLowerTriangular):
    return _TRIL
  if isinstance(operator, linear_operator_full_matrix.LinearOperatorFullMatrix):
    return _MATRIX
  if isinstance(operator, linear_operator_identity.LinearOperatorIdentity):
    return _IDENTITY
  if isinstance(operator,
                linear_operator_identity.LinearOperatorScaledIdentity):
    return _SCALED_IDENTITY
  raise TypeError(f"Expected operator to be one of [LinearOperatorDiag, "
                  f"LinearOperatorLowerTriangular, LinearOperatorFullMatrix, "
                  f"LinearOperatorIdentity, LinearOperatorScaledIdentity]. "
                  f"Received: {operator}")


################################################################################
# Addition tiers:
# We attempt to use Adders in tier K before K+1.
#
# Organize tiers to
#   (i) reduce O(..) complexity of forming final operator, and
#   (ii) produce the "most efficient" final operator.
# Dev notes:
#  * Results of addition at tier K will be added at tier K or higher.
#  * Tiers may change, and we warn the user that it may change.
################################################################################

# Note that the final tier, _AddAndReturnMatrix, will convert everything to a
# dense matrix.  So it is sometimes very inefficient.
_DEFAULT_ADDITION_TIERS = [
    [_AddAndReturnScaledIdentity()],
    [_AddAndReturnDiag()],
    [_AddAndReturnTriL()],
    [_AddAndReturnMatrix()],
]

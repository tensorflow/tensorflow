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

"""Registration mechanisms for various n-ary operations on LinearOperators."""

import itertools

from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect


_SOLVE = {}


def _registered_function(type_list, registry):
  """Given a list of classes, finds the most specific function registered."""
  enumerated_hierarchies = [enumerate(tf_inspect.getmro(t)) for t in type_list]
  # Get all possible combinations of hierarchies.
  cls_combinations = list(itertools.product(*enumerated_hierarchies))

  def hierarchy_distance(cls_combination):
    candidate_distance = sum(c[0] for c in cls_combination)
    if tuple(c[1] for c in cls_combination) in registry:
      return candidate_distance
    return 10000

  registered_combination = min(cls_combinations, key=hierarchy_distance)
  return registry.get(tuple(r[1] for r in registered_combination), None)


def _registered_solve(type_a, type_b):
  """Get the Solve function registered for classes a and b."""
  return _registered_function([type_a, type_b], _SOLVE)


def solve(lin_op_a, lin_op_b, name=None):
  """Compute lin_op_a.solve(lin_op_b).

  Args:
    lin_op_a: The LinearOperator on the left.
    lin_op_b: The LinearOperator on the right.
    name: Name to use for this operation.

  Returns:
    A LinearOperator that represents the solve between `lin_op_a` and
      `lin_op_b`.

  Raises:
    NotImplementedError: If no solve method is defined between types of
      `lin_op_a` and `lin_op_b`.
  """
  solve_fn = _registered_solve(type(lin_op_a), type(lin_op_b))
  if solve_fn is None:
    raise ValueError("No solve registered for {}.solve({})".format(
        type(lin_op_a), type(lin_op_b)))

  with ops.name_scope(name, "Solve"):
    return solve_fn(lin_op_a, lin_op_b)


class RegisterSolve:
  """Decorator to register a Solve implementation function.

  Usage:

  @linear_operator_algebra.RegisterSolve(
    lin_op.LinearOperatorIdentity,
    lin_op.LinearOperatorIdentity)
  def _solve_identity(a, b):
    # Return the identity matrix.
  """

  def __init__(self, lin_op_cls_a, lin_op_cls_b):
    """Initialize the LinearOperator registrar.

    Args:
      lin_op_cls_a: the class of the LinearOperator that is computing solve.
      lin_op_cls_b: the class of the second LinearOperator to solve.
    """
    self._key = (lin_op_cls_a, lin_op_cls_b)

  def __call__(self, solve_fn):
    """Perform the Solve registration.

    Args:
      solve_fn: The function to use for the Solve.

    Returns:
      solve_fn

    Raises:
      TypeError: if solve_fn is not a callable.
      ValueError: if a Solve function has already been registered for
        the given argument classes.
    """
    if not callable(solve_fn):
      raise TypeError(
          "solve_fn must be callable, received: {}".format(solve_fn))
    if self._key in _SOLVE:
      raise ValueError("Solve({}, {}) has already been registered.".format(
          self._key[0].__name__,
          self._key[1].__name__))
    _SOLVE[self._key] = solve_fn
    return solve_fn

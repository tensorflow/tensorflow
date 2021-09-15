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
"""Inverts a non-singular `LinearOperator`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = []


@tf_export("linalg.LinearOperatorInversion")
@linear_operator.make_composite_tensor
class LinearOperatorInversion(linear_operator.LinearOperator):
  """`LinearOperator` representing the inverse of another operator.

  This operator represents the inverse of another operator.

  ```python
  # Create a 2 x 2 linear operator.
  operator = LinearOperatorFullMatrix([[1., 0.], [0., 2.]])
  operator_inv = LinearOperatorInversion(operator)

  operator_inv.to_dense()
  ==> [[1., 0.]
       [0., 0.5]]

  operator_inv.shape
  ==> [2, 2]

  operator_inv.log_abs_determinant()
  ==> - log(2)

  x = ... Shape [2, 4] Tensor
  operator_inv.matmul(x)
  ==> Shape [2, 4] Tensor, equal to operator.solve(x)
  ```

  #### Performance

  The performance of `LinearOperatorInversion` depends on the underlying
  operators performance:  `solve` and `matmul` are swapped, and determinant is
  inverted.

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
               operator,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
    r"""Initialize a `LinearOperatorInversion`.

    `LinearOperatorInversion` is initialized with an operator `A`.  The `solve`
    and `matmul` methods are effectively swapped.  E.g.

    ```
    A = MyLinearOperator(...)
    B = LinearOperatorInversion(A)
    x = [....]  # a vector

    assert A.matvec(x) == B.solvevec(x)
    ```

    Args:
      operator: `LinearOperator` object. If `operator.is_non_singular == False`,
        an exception is raised.  We do allow `operator.is_non_singular == None`,
        in which case this operator will have `is_non_singular == None`.
        Similarly for `is_self_adjoint` and `is_positive_definite`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`. Default is `operator.name +
        "_inv"`.

    Raises:
      ValueError:  If `operator.is_non_singular` is False.
    """
    parameters = dict(
        operator=operator,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    self._operator = operator

    # Auto-set and check hints.
    if operator.is_non_singular is False or is_non_singular is False:
      raise ValueError(
          f"Argument `is_non_singular` or argument `operator` must have "
          f"supplied hint `is_non_singular` equal to `True` or `None`. "
          f"Found `operator.is_non_singular`: {operator.is_non_singular}, "
          f"`is_non_singular`: {is_non_singular}.")
    if operator.is_square is False or is_square is False:
      raise ValueError(
          f"Argument `is_square` or argument `operator` must have supplied "
          f"hint `is_square` equal to `True` or `None`. Found "
          f"`operator.is_square`: {operator.is_square}, "
          f"`is_square`: {is_square}.")

    # The congruency of is_non_singular and is_self_adjoint was checked in the
    # base operator.  Other hints are, in this special case of inversion, ones
    # that must be the same for base/derived operator.
    combine_hint = (
        linear_operator_util.use_operator_or_provided_hint_unless_contradicting)

    is_square = combine_hint(
        operator, "is_square", is_square,
        "An operator is square if and only if its inverse is square.")

    is_non_singular = combine_hint(
        operator, "is_non_singular", is_non_singular,
        "An operator is non-singular if and only if its inverse is "
        "non-singular.")

    is_self_adjoint = combine_hint(
        operator, "is_self_adjoint", is_self_adjoint,
        "An operator is self-adjoint if and only if its inverse is "
        "self-adjoint.")

    is_positive_definite = combine_hint(
        operator, "is_positive_definite", is_positive_definite,
        "An operator is positive-definite if and only if its inverse is "
        "positive-definite.")

    # Initialization.
    if name is None:
      name = operator.name + "_inv"
    with ops.name_scope(name, values=operator.graph_parents):
      super(LinearOperatorInversion, self).__init__(
          dtype=operator.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)
    # TODO(b/143910018) Remove graph_parents in V3.
    self._set_graph_parents(operator.graph_parents)

  @property
  def operator(self):
    """The operator before inversion."""
    return self._operator

  def _assert_non_singular(self):
    return self.operator.assert_non_singular()

  def _assert_positive_definite(self):
    return self.operator.assert_positive_definite()

  def _assert_self_adjoint(self):
    return self.operator.assert_self_adjoint()

  def _shape(self):
    return self.operator.shape

  def _shape_tensor(self):
    return self.operator.shape_tensor()

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    return self.operator.solve(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _determinant(self):
    return 1. / self.operator.determinant()

  def _log_abs_determinant(self):
    return -1. * self.operator.log_abs_determinant()

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    return self.operator.matmul(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _eigvals(self):
    return 1. / self.operator.eigvals()

  def _cond(self):
    return self.operator.cond()

  @property
  def _composite_tensor_fields(self):
    return ("operator",)

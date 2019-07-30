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
"""Takes the adjoint of a `LinearOperator`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = []


@tf_export("linalg.LinearOperatorAdjoint")
class LinearOperatorAdjoint(linear_operator.LinearOperator):
  """`LinearOperator` representing the adjoint of another operator.

  This operator represents the adjoint of another operator.

  ```python
  # Create a 2 x 2 linear operator.
  operator = LinearOperatorFullMatrix([[1 - i., 3.], [0., 1. + i]])
  operator_adjoint = LinearOperatorAdjoint(operator)

  operator_adjoint.to_dense()
  ==> [[1. + i, 0.]
       [3., 1 - i]]

  operator_adjoint.shape
  ==> [2, 2]

  operator_adjoint.log_abs_determinant()
  ==> - log(2)

  x = ... Shape [2, 4] Tensor
  operator_adjoint.matmul(x)
  ==> Shape [2, 4] Tensor, equal to operator.matmul(x, adjoint=True)
  ```

  #### Performance

  The performance of `LinearOperatorAdjoint` depends on the underlying
  operators performance.

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
    r"""Initialize a `LinearOperatorAdjoint`.

    `LinearOperatorAdjoint` is initialized with an operator `A`.  The `solve`
    and `matmul` methods  effectively flip the `adjoint` argument.  E.g.

    ```
    A = MyLinearOperator(...)
    B = LinearOperatorAdjoint(A)
    x = [....]  # a vector

    assert A.matvec(x, adjoint=True) == B.matvec(x, adjoint=False)
    ```

    Args:
      operator: `LinearOperator` object.
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
        "_adjoint"`.

    Raises:
      ValueError:  If `operator.is_non_singular` is False.
    """

    self._operator = operator

    # The congruency of is_non_singular and is_self_adjoint was checked in the
    # base operator.
    combine_hint = (
        linear_operator_util.use_operator_or_provided_hint_unless_contradicting)

    is_square = combine_hint(
        operator, "is_square", is_square,
        "An operator is square if and only if its adjoint is square.")

    is_non_singular = combine_hint(
        operator, "is_non_singular", is_non_singular,
        "An operator is non-singular if and only if its adjoint is "
        "non-singular.")

    is_self_adjoint = combine_hint(
        operator, "is_self_adjoint", is_self_adjoint,
        "An operator is self-adjoint if and only if its adjoint is "
        "self-adjoint.")

    is_positive_definite = combine_hint(
        operator, "is_positive_definite", is_positive_definite,
        "An operator is positive-definite if and only if its adjoint is "
        "positive-definite.")

    # Initialization.
    if name is None:
      name = operator.name + "_adjoint"
    with ops.name_scope(name, values=operator.graph_parents):
      super(LinearOperatorAdjoint, self).__init__(
          dtype=operator.dtype,
          graph_parents=operator.graph_parents,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          name=name)

  @property
  def operator(self):
    """The operator before taking the adjoint."""
    return self._operator

  def _assert_non_singular(self):
    return self.operator.assert_non_singular()

  def _assert_positive_definite(self):
    return self.operator.assert_positive_definite()

  def _assert_self_adjoint(self):
    return self.operator.assert_self_adjoint()

  def _shape(self):
    # Rotate last dimension
    shape = self.operator.shape
    return shape[:-2].concatenate([shape[-1], shape[-2]])

  def _shape_tensor(self):
    # Rotate last dimension
    shape = self.operator.shape_tensor()
    return array_ops.concat([
        shape[:-2], [shape[-1], shape[-2]]], axis=-1)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    return self.operator.matmul(
        x, adjoint=(not adjoint), adjoint_arg=adjoint_arg)

  def _matvec(self, x, adjoint=False):
    return self.operator.matvec(x, adjoint=(not adjoint))

  def _determinant(self):
    if self.is_self_adjoint:
      return self.operator.determinant()
    return math_ops.conj(self.operator.determinant())

  def _log_abs_determinant(self):
    return self.operator.log_abs_determinant()

  def _trace(self):
    if self.is_self_adjoint:
      return self.operator.trace()
    return math_ops.conj(self.operator.trace())

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    return self.operator.solve(
        rhs, adjoint=(not adjoint), adjoint_arg=adjoint_arg)

  def _solvevec(self, rhs, adjoint=False):
    return self.operator.solvevec(rhs, adjoint=(not adjoint))

  def _to_dense(self):
    if self.is_self_adjoint:
      return self.operator.to_dense()
    return linalg.adjoint(self.operator.to_dense())

  def _add_to_tensor(self, x):
    return self.to_dense() + x

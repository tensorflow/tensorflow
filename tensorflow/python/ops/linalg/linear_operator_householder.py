# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""`LinearOperator` acting like a Householder transformation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorHouseholder",]


@tf_export("linalg.LinearOperatorHouseholder")
@linear_operator.make_composite_tensor
class LinearOperatorHouseholder(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] of Householder transformations.

  This operator acts like a [batch] of householder reflections with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorHouseholder` is initialized with a (batch) vector.

  A Householder reflection, defined via a vector `v`, which reflects points
  in `R^n` about the hyperplane orthogonal to `v` and through the origin.

  ```python
  # Create a 2 x 2 householder transform.
  vec = [1 / np.sqrt(2), 1. / np.sqrt(2)]
  operator = LinearOperatorHouseholder(vec)

  operator.to_dense()
  ==> [[0.,  -1.]
       [-1., -0.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

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
               reflection_axis,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorHouseholder"):
    r"""Initialize a `LinearOperatorHouseholder`.

    Args:
      reflection_axis:  Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.
        The vector defining the hyperplane to reflect about.
        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,
        `complex128`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  This is autoset to true
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
        This is autoset to false.
      is_square:  Expect that this operator acts like square [batch] matrices.
        This is autoset to true.
      name: A name for this `LinearOperator`.

    Raises:
      ValueError:  `is_self_adjoint` is not `True`, `is_positive_definite` is
        not `False` or `is_square` is not `True`.
    """
    parameters = dict(
        reflection_axis=reflection_axis,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with ops.name_scope(name, values=[reflection_axis]):
      self._reflection_axis = linear_operator_util.convert_nonref_to_tensor(
          reflection_axis, name="reflection_axis")
      self._check_reflection_axis(self._reflection_axis)

      # Check and auto-set hints.
      if is_self_adjoint is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError("A Householder operator is always self adjoint.")
      else:
        is_self_adjoint = True

      if is_positive_definite is True:  # pylint:disable=g-bool-id-comparison
        raise ValueError(
            "A Householder operator is always non-positive definite.")
      else:
        is_positive_definite = False

      if is_square is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError("A Householder operator is always square.")
      is_square = True

      super(LinearOperatorHouseholder, self).__init__(
          dtype=self._reflection_axis.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)
      # TODO(b/143910018) Remove graph_parents in V3.
      self._set_graph_parents([self._reflection_axis])

  def _check_reflection_axis(self, reflection_axis):
    """Static check of reflection_axis."""
    if (reflection_axis.shape.ndims is not None and
        reflection_axis.shape.ndims < 1):
      raise ValueError(
          "Argument reflection_axis must have at least 1 dimension.  "
          "Found: %s" % reflection_axis)

  def _shape(self):
    # If d_shape = [5, 3], we return [5, 3, 3].
    d_shape = self._reflection_axis.shape
    return d_shape.concatenate(d_shape[-1:])

  def _shape_tensor(self):
    d_shape = array_ops.shape(self._reflection_axis)
    k = d_shape[-1]
    return array_ops.concat((d_shape, [k]), 0)

  def _assert_non_singular(self):
    return control_flow_ops.no_op("assert_non_singular")

  def _assert_positive_definite(self):
    raise errors.InvalidArgumentError(
        node_def=None, op=None, message="Householder operators are always "
        "non-positive definite.")

  def _assert_self_adjoint(self):
    return control_flow_ops.no_op("assert_self_adjoint")

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # Given a vector `v`, we would like to reflect `x` about the hyperplane
    # orthogonal to `v` going through the origin.  We first project `x` to `v`
    # to get v * dot(v, x) / dot(v, v).  After we project, we can reflect the
    # projection about the hyperplane by flipping sign to get
    # -v * dot(v, x) / dot(v, v).  Finally, we can add back the component
    # that is orthogonal to v. This is invariant under reflection, since the
    # whole hyperplane is invariant. This component is equal to x - v * dot(v,
    # x) / dot(v, v), giving the formula x - 2 * v * dot(v, x) / dot(v, v)
    # for the reflection.

    # Note that because this is a reflection, it lies in O(n) (for real vector
    # spaces) or U(n) (for complex vector spaces), and thus is its own adjoint.
    reflection_axis = ops.convert_to_tensor_v2_with_dispatch(
        self.reflection_axis)
    x = linalg.adjoint(x) if adjoint_arg else x
    normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
    mat = normalized_axis[..., array_ops.newaxis]
    x_dot_normalized_v = math_ops.matmul(mat, x, adjoint_a=True)

    return x - 2 * mat * x_dot_normalized_v

  def _trace(self):
    # We have (n - 1) +1 eigenvalues and a single -1 eigenvalue.
    shape = self.shape_tensor()
    return math_ops.cast(
        self._domain_dimension_tensor(shape=shape) - 2,
        self.dtype) * array_ops.ones(
            shape=self._batch_shape_tensor(shape=shape), dtype=self.dtype)

  def _determinant(self):
    # For householder transformations, the determinant is -1.
    return -array_ops.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)  # pylint: disable=invalid-unary-operand-type

  def _log_abs_determinant(self):
    # Orthogonal matrix -> log|Q| = 0.
    return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # A householder reflection is a reflection, hence is idempotent. Thus we
    # can just apply a matmul.
    return self._matmul(rhs, adjoint, adjoint_arg)

  def _to_dense(self):
    reflection_axis = ops.convert_to_tensor_v2_with_dispatch(
        self.reflection_axis)
    normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
    mat = normalized_axis[..., array_ops.newaxis]
    matrix = -2 * math_ops.matmul(mat, mat, adjoint_b=True)
    return array_ops.matrix_set_diag(
        matrix, 1. + array_ops.matrix_diag_part(matrix))

  def _diag_part(self):
    reflection_axis = ops.convert_to_tensor_v2_with_dispatch(
        self.reflection_axis)
    normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
    return 1. - 2 * normalized_axis * math_ops.conj(normalized_axis)

  def _eigvals(self):
    # We have (n - 1) +1 eigenvalues and a single -1 eigenvalue.
    result_shape = array_ops.shape(self.reflection_axis)
    n = result_shape[-1]
    ones_shape = array_ops.concat([result_shape[:-1], [n - 1]], axis=-1)
    neg_shape = array_ops.concat([result_shape[:-1], [1]], axis=-1)
    eigvals = array_ops.ones(shape=ones_shape, dtype=self.dtype)
    eigvals = array_ops.concat(
        [-array_ops.ones(shape=neg_shape, dtype=self.dtype), eigvals], axis=-1)  # pylint: disable=invalid-unary-operand-type
    return eigvals

  def _cond(self):
    # Householder matrices are rotations which have condition number 1.
    return array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)

  @property
  def reflection_axis(self):
    return self._reflection_axis

  @property
  def _composite_tensor_fields(self):
    return ("reflection_axis",)

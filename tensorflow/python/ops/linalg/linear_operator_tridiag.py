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
"""`LinearOperator` acting like a tridiagonal matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ['LinearOperatorTridiag',]

_COMPACT = 'compact'
_MATRIX = 'matrix'
_SEQUENCE = 'sequence'
_DIAGONAL_FORMATS = frozenset({_COMPACT, _MATRIX, _SEQUENCE})


@tf_export('linalg.LinearOperatorTridiag')
@linear_operator.make_composite_tensor
class LinearOperatorTridiag(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] square tridiagonal matrix.

  This operator acts like a [batch] square tridiagonal matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x M` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  Example usage:

  Create a 3 x 3 tridiagonal linear operator.

  >>> superdiag = [3., 4., 5.]
  >>> diag = [1., -1., 2.]
  >>> subdiag = [6., 7., 8]
  >>> operator = tf.linalg.LinearOperatorTridiag(
  ...    [superdiag, diag, subdiag],
  ...    diagonals_format='sequence')
  >>> operator.to_dense()
  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
  array([[ 1.,  3.,  0.],
         [ 7., -1.,  4.],
         [ 0.,  8.,  2.]], dtype=float32)>
  >>> operator.shape
  TensorShape([3, 3])

  Scalar Tensor output.

  >>> operator.log_abs_determinant()
  <tf.Tensor: shape=(), dtype=float32, numpy=4.3307333>

  Create a [2, 3] batch of 4 x 4 linear operators.

  >>> diagonals = tf.random.normal(shape=[2, 3, 3, 4])
  >>> operator = tf.linalg.LinearOperatorTridiag(
  ...   diagonals,
  ...   diagonals_format='compact')

  Create a shape [2, 1, 4, 2] vector.  Note that this shape is compatible
  since the batch dimensions, [2, 1], are broadcast to
  operator.batch_shape = [2, 3].

  >>> y = tf.random.normal(shape=[2, 1, 4, 2])
  >>> x = operator.solve(y)
  >>> x
  <tf.Tensor: shape=(2, 3, 4, 2), dtype=float32, numpy=...,
  dtype=float32)>

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb].
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorTridiag` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` will take O(N * R) time.
  * `operator.solve(x)` will take O(N * R) time.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

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
               diagonals,
               diagonals_format=_COMPACT,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name='LinearOperatorTridiag'):
    r"""Initialize a `LinearOperatorTridiag`.

    Args:
      diagonals: `Tensor` or list of `Tensor`s depending on `diagonals_format`.

        If `diagonals_format=sequence`, this is a list of three `Tensor`'s each
        with shape `[B1, ..., Bb, N]`, `b >= 0, N >= 0`, representing the
        superdiagonal, diagonal and subdiagonal in that order. Note the
        superdiagonal is padded with an element in the last position, and the
        subdiagonal is padded with an element in the front.

        If `diagonals_format=matrix` this is a `[B1, ... Bb, N, N]` shaped
        `Tensor` representing the full tridiagonal matrix.

        If `diagonals_format=compact` this is a `[B1, ... Bb, 3, N]` shaped
        `Tensor` with the second to last dimension indexing the
        superdiagonal, diagonal and subdiagonal in that order. Note the
        superdiagonal is padded with an element in the last position, and the
        subdiagonal is padded with an element in the front.

        In every case, these `Tensor`s are all floating dtype.
      diagonals_format: one of `matrix`, `sequence`, or `compact`. Default is
        `compact`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `diag.dtype` is real, this is auto-set to `True`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
      ValueError:  If `diag.dtype` is real, and `is_self_adjoint` is not `True`.
    """
    parameters = dict(
        diagonals=diagonals,
        diagonals_format=diagonals_format,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with ops.name_scope(name, values=[diagonals]):
      if diagonals_format not in _DIAGONAL_FORMATS:
        raise ValueError(
            'Diagonals Format must be one of compact, matrix, sequence'
            ', got : {}'.format(diagonals_format))
      if diagonals_format == _SEQUENCE:
        self._diagonals = [linear_operator_util.convert_nonref_to_tensor(
            d, name='diag_{}'.format(i)) for i, d in enumerate(diagonals)]
        dtype = self._diagonals[0].dtype
      else:
        self._diagonals = linear_operator_util.convert_nonref_to_tensor(
            diagonals, name='diagonals')
        dtype = self._diagonals.dtype
      self._diagonals_format = diagonals_format

      super(LinearOperatorTridiag, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _shape(self):
    if self.diagonals_format == _MATRIX:
      return self.diagonals.shape
    if self.diagonals_format == _COMPACT:
      # Remove the second to last dimension that contains the value 3.
      d_shape = self.diagonals.shape[:-2].concatenate(
          self.diagonals.shape[-1])
    else:
      broadcast_shape = array_ops.broadcast_static_shape(
          self.diagonals[0].shape[:-1],
          self.diagonals[1].shape[:-1])
      broadcast_shape = array_ops.broadcast_static_shape(
          broadcast_shape,
          self.diagonals[2].shape[:-1])
      d_shape = broadcast_shape.concatenate(self.diagonals[1].shape[-1])
    return d_shape.concatenate(d_shape[-1])

  def _shape_tensor(self, diagonals=None):
    diagonals = diagonals if diagonals is not None else self.diagonals
    if self.diagonals_format == _MATRIX:
      return array_ops.shape(diagonals)
    if self.diagonals_format == _COMPACT:
      d_shape = array_ops.shape(diagonals[..., 0, :])
    else:
      broadcast_shape = array_ops.broadcast_dynamic_shape(
          array_ops.shape(self.diagonals[0])[:-1],
          array_ops.shape(self.diagonals[1])[:-1])
      broadcast_shape = array_ops.broadcast_dynamic_shape(
          broadcast_shape,
          array_ops.shape(self.diagonals[2])[:-1])
      d_shape = array_ops.concat(
          [broadcast_shape, [array_ops.shape(self.diagonals[1])[-1]]], axis=0)
    return array_ops.concat([d_shape, [d_shape[-1]]], axis=-1)

  def _assert_self_adjoint(self):
    # Check the diagonal has non-zero imaginary, and the super and subdiagonals
    # are conjugate.

    asserts = []
    diag_message = (
        'This tridiagonal operator contained non-zero '
        'imaginary values on the diagonal.')
    off_diag_message = (
        'This tridiagonal operator has non-conjugate '
        'subdiagonal and superdiagonal.')

    if self.diagonals_format == _MATRIX:
      asserts += [check_ops.assert_equal(
          self.diagonals, linalg.adjoint(self.diagonals),
          message='Matrix was not equal to its adjoint.')]
    elif self.diagonals_format == _COMPACT:
      diagonals = ops.convert_to_tensor_v2_with_dispatch(self.diagonals)
      asserts += [linear_operator_util.assert_zero_imag_part(
          diagonals[..., 1, :], message=diag_message)]
      # Roll the subdiagonal so the shifted argument is at the end.
      subdiag = manip_ops.roll(diagonals[..., 2, :], shift=-1, axis=-1)
      asserts += [check_ops.assert_equal(
          math_ops.conj(subdiag[..., :-1]),
          diagonals[..., 0, :-1],
          message=off_diag_message)]
    else:
      asserts += [linear_operator_util.assert_zero_imag_part(
          self.diagonals[1], message=diag_message)]
      subdiag = manip_ops.roll(self.diagonals[2], shift=-1, axis=-1)
      asserts += [check_ops.assert_equal(
          math_ops.conj(subdiag[..., :-1]),
          self.diagonals[0][..., :-1],
          message=off_diag_message)]
    return control_flow_ops.group(asserts)

  def _construct_adjoint_diagonals(self, diagonals):
    # Constructs adjoint tridiagonal matrix from diagonals.
    if self.diagonals_format == _SEQUENCE:
      diagonals = [math_ops.conj(d) for d in reversed(diagonals)]
      # The subdiag and the superdiag swap places, so we need to shift the
      # padding argument.
      diagonals[0] = manip_ops.roll(diagonals[0], shift=-1, axis=-1)
      diagonals[2] = manip_ops.roll(diagonals[2], shift=1, axis=-1)
      return diagonals
    elif self.diagonals_format == _MATRIX:
      return linalg.adjoint(diagonals)
    else:
      diagonals = math_ops.conj(diagonals)
      superdiag, diag, subdiag = array_ops.unstack(
          diagonals, num=3, axis=-2)
      # The subdiag and the superdiag swap places, so we need
      # to shift all arguments.
      new_superdiag = manip_ops.roll(subdiag, shift=-1, axis=-1)
      new_subdiag = manip_ops.roll(superdiag, shift=1, axis=-1)
      return array_ops.stack([new_superdiag, diag, new_subdiag], axis=-2)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    diagonals = self.diagonals
    if adjoint:
      diagonals = self._construct_adjoint_diagonals(diagonals)
    x = linalg.adjoint(x) if adjoint_arg else x
    return linalg.tridiagonal_matmul(
        diagonals, x,
        diagonals_format=self.diagonals_format)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    diagonals = self.diagonals
    if adjoint:
      diagonals = self._construct_adjoint_diagonals(diagonals)

    # TODO(b/144860784): Remove the broadcasting code below once
    # tridiagonal_solve broadcasts.

    rhs_shape = array_ops.shape(rhs)
    k = self._shape_tensor(diagonals)[-1]
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        self._shape_tensor(diagonals)[:-2], rhs_shape[:-2])
    rhs = array_ops.broadcast_to(
        rhs, array_ops.concat(
            [broadcast_shape, rhs_shape[-2:]], axis=-1))
    if self.diagonals_format == _MATRIX:
      diagonals = array_ops.broadcast_to(
          diagonals, array_ops.concat(
              [broadcast_shape, [k, k]], axis=-1))
    elif self.diagonals_format == _COMPACT:
      diagonals = array_ops.broadcast_to(
          diagonals, array_ops.concat(
              [broadcast_shape, [3, k]], axis=-1))
    else:
      diagonals = [
          array_ops.broadcast_to(d, array_ops.concat(
              [broadcast_shape, [k]], axis=-1)) for d in diagonals]

    y = linalg.tridiagonal_solve(
        diagonals, rhs,
        diagonals_format=self.diagonals_format,
        transpose_rhs=adjoint_arg,
        conjugate_rhs=adjoint_arg)
    return y

  def _diag_part(self):
    if self.diagonals_format == _MATRIX:
      return array_ops.matrix_diag_part(self.diagonals)
    elif self.diagonals_format == _SEQUENCE:
      diagonal = self.diagonals[1]
      return array_ops.broadcast_to(
          diagonal, self.shape_tensor()[:-1])
    else:
      return self.diagonals[..., 1, :]

  def _to_dense(self):
    if self.diagonals_format == _MATRIX:
      return self.diagonals

    if self.diagonals_format == _COMPACT:
      return gen_array_ops.matrix_diag_v3(
          self.diagonals,
          k=(-1, 1),
          num_rows=-1,
          num_cols=-1,
          align='LEFT_RIGHT',
          padding_value=0.)

    diagonals = [
        ops.convert_to_tensor_v2_with_dispatch(d) for d in self.diagonals
    ]
    diagonals = array_ops.stack(diagonals, axis=-2)

    return gen_array_ops.matrix_diag_v3(
        diagonals,
        k=(-1, 1),
        num_rows=-1,
        num_cols=-1,
        align='LEFT_RIGHT',
        padding_value=0.)

  @property
  def diagonals(self):
    return self._diagonals

  @property
  def diagonals_format(self):
    return self._diagonals_format

  @property
  def _composite_tensor_fields(self):
    return ('diagonals', 'diagonals_format')

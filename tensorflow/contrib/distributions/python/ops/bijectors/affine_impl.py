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
"""Affine bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import operator_pd_cholesky
from tensorflow.contrib.distributions.python.ops import operator_pd_diag
from tensorflow.contrib.distributions.python.ops import operator_pd_identity
from tensorflow.contrib.distributions.python.ops import operator_pd_vdvt_update
from tensorflow.contrib.distributions.python.ops.shape import _DistributionShape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector


__all__ = [
    "Affine",
]


def _as_tensor(x, name):
  """Convenience to convert to `Tensor` or leave as `None`."""
  return None if x is None else ops.convert_to_tensor(x, name=name)


# TODO(srvasude): Deprecate this class with a dedicated Linear Operator
# corresponding to TriL + V D V.T.
class _TriLPlusVDVTLightweightOperatorPD(object):
  """Helper/hidden class fake an OperatorPD for TriL+VDV.T."""

  def __init__(self, tril, v, diag=None, validate_args=False):
    """Creates an instance of _TriLPlusVDVTLightweightOperatorPD.

    WARNING: This object is not to be used outside of `Affine` where it is
    currently being temporarily used for refactoring purposes.

    Args:
      tril: `Tensor` of shape `[B1,..,Bb, d, d]`.
      v: `Tensor` of shape `[B1,...,Bb, d, k]`.
      diag: `Tensor` of shape `[B1,...,Bb, k, k]` or None
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
    """
    self._m = tril
    self._v = v
    self._validate_args = validate_args
    self._inputs = [tril, v]
    if diag is not None:
      self._inputs += [diag]
      self._d = operator_pd_diag.OperatorPDDiag(diag, verify_pd=validate_args)
      self._d_inv = operator_pd_diag.OperatorPDDiag(1. / diag,
                                                    verify_pd=validate_args)
      return
    if v.get_shape().is_fully_defined():
      v_shape = v.get_shape().as_list()
      id_shape = v_shape[:-2] + [v_shape[-1], v_shape[-1]]
    else:
      v_shape = array_ops.shape(v)
      id_shape = array_ops.concat([v_shape[:-2], [v_shape[-1], v_shape[-1]]], 0)
    self._d = operator_pd_identity.OperatorPDIdentity(
        id_shape, v.dtype, verify_pd=self.validate_args)
    self._d_inv = self._d

  @property
  def inputs(self):
    return self._inputs

  @property
  def dtype(self):
    return self._m.dtype.base_dtype

  @property
  def validate_args(self):
    return self._validate_args

  def rank(self):
    """Returns `rank(self)`."""
    return array_ops.rank(self._m)

  def sqrt_matmul(self, x):
    """Computes `matmul(self, x)`.

    Doesn't actually do the sqrt! Named as such to agree with API.

    Args:
      x: `Tensor`

    Returns:
      self_times_x: `Tensor`
    """
    m_x = math_ops.matmul(self._m, x)
    vt_x = math_ops.matmul(self._v, x, adjoint_a=True)
    d_vt_x = self._d.matmul(vt_x)
    v_d_vt_x = math_ops.matmul(self._v, d_vt_x)
    return m_x + v_d_vt_x

  def sqrt_solve(self, x):
    """Computes `solve(self, x)`.

    Doesn't actually do the sqrt! Named as such to agree with API.

    To compute (M + V D V.T), we use the Woodbury matrix identity:
      inv(M + V D V.T) = inv(M) - inv(M) V inv(C) V.T inv(M)
    where,
      C = inv(D) + V.T inv(M) V.
    See: https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    Args:
      x: `Tensor`

    Returns:
      inv_of_self_times_x: `Tensor`
    """
    minv_x = linalg_ops.matrix_triangular_solve(self._m, x)
    vt_minv_x = math_ops.matmul(self._v, minv_x, transpose_a=True)
    cinv_vt_minv_x = linalg_ops.matrix_solve(
        self._woodbury_sandwiched_term(), vt_minv_x)
    v_cinv_vt_minv_x = math_ops.matmul(self._v, cinv_vt_minv_x)
    minv_v_cinv_vt_minv_x = linalg_ops.matrix_triangular_solve(
        self._m, v_cinv_vt_minv_x)
    return minv_x - minv_v_cinv_vt_minv_x

  def sqrt_log_abs_det(self):
    """Computes (log o abs o det)(X) for matrix X.

    Doesn't actually do the sqrt! Named as such to agree with API.

    To compute det(M + V D V.T), we use the matrix determinant lemma:
      det(Tril + V D V.T) = det(C) det(D) det(M)
    where C is defined as in `_inverse`, ie,
      C = inv(D) + V.T inv(M) V.

    See: https://en.wikipedia.org/wiki/Matrix_determinant_lemma

    Returns:
      log_abs_det: `Tensor`.
    """
    log_det_c = math_ops.log(math_ops.abs(
        linalg_ops.matrix_determinant(self._woodbury_sandwiched_term())))
    # Reduction is ok because we always prepad inputs to this class.
    log_det_m = math_ops.reduce_sum(math_ops.log(math_ops.abs(
        array_ops.matrix_diag_part(self._m))), axis=[-1])
    return log_det_c + 2. * self._d.sqrt_log_abs_det() + log_det_m

  def _woodbury_sandwiched_term(self):
    """Computes the sandwiched term in the Woodbury identity.

    Computes the "`C`" in the identity:
       inv(M + V D V.T) = inv(M) - inv(M) V inv(C) V.T inv(M)
    where,
       C = inv(D) + V.T inv(M) V.

    See: https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    Returns:
      woodbury_sandwich_term: A `Tensor` to be used like `C`, above.
    """
    minv_v = linalg_ops.matrix_triangular_solve(self._m, self._v)
    vt_minv_v = math_ops.matmul(self._v, minv_v, adjoint_a=True)
    return self._d_inv.add_to_tensor(vt_minv_v)


class Affine(bijector.Bijector):
  """Compute `Y = g(X; shift, scale) = scale @ X + shift`.

  Here `scale = c * I + diag(D1) + tril(L) + V @ diag(D2) @ V.T`.

  In TF parlance, the `scale` term is logically equivalent to:

  ```python
  scale = (
    scale_identity_multiplier * tf.diag(tf.ones(d)) +
    tf.diag(scale_diag) +
    scale_tril +
    scale_perturb_factor @ diag(scale_perturb_diag) @
      tf.transpose([scale_perturb_factor])
  )
  ```

  The `scale` term is applied without necessarily materializing constituent
  matrices, i.e., the matmul is [matrix-free](
  https://en.wikipedia.org/wiki/Matrix-free_methods) when possible.

  Examples:

  ```python
  # Y = X
  b = Affine()

  # Y = X + shift
  b = Affine(shift=[1., 2, 3])

  # Y = 2 * I @ X.T + shift
  b = Affine(shift=[1., 2, 3],
             scale_identity_multiplier=2.)

  # Y = tf.diag(d1) @ X.T + shift
  b = Affine(shift=[1., 2, 3],
             scale_diag=[-1., 2, 1])         # Implicitly 3x3.

  # Y = (I + v * v.T) @ X.T + shift
  b = Affine(shift=[1., 2, 3],
             scale_perturb_factor=[[1., 0],
                                   [0, 1],
                                   [1, 1]])

  # Y = (diag(d1) + v * diag(d2) * v.T) @ X.T + shift
  b = Affine(shift=[1., 2, 3],
             scale_diag=[1., 3, 3],          # Implicitly 3x3.
             scale_perturb_diag=[2., 1],     # Implicitly 2x2.
             scale_perturb_factor=[[1., 0],
                                   [0, 1],
                                   [1, 1]])

  ```

  """

  def __init__(self,
               shift=None,
               scale_identity_multiplier=None,
               scale_diag=None,
               scale_tril=None,
               scale_perturb_factor=None,
               scale_perturb_diag=None,
               event_ndims=1,
               validate_args=False,
               name="affine"):
    """Instantiates the `Affine` bijector.

    This `Bijector` is initialized with `shift` `Tensor` and `scale` arguments,
    giving the forward operation:

    ```none
    Y = g(X) = scale @ X + shift
    ```

    where the `scale` term is logically equivalent to:

    ```python
    scale = (
      scale_identity_multiplier * tf.diag(tf.ones(d)) +
      tf.diag(scale_diag) +
      scale_tril +
      scale_perturb_factor @ diag(scale_perturb_diag) @
        tf.transpose([scale_perturb_factor])
    )
    ```

    If none of `scale_identity_multiplier`, `scale_diag`, or `scale_tril` are
    specified then `scale += IdentityMatrix`. Otherwise specifying a
    `scale` argument has the semantics of `scale += Expand(arg)`, i.e.,
    `scale_diag != None` means `scale += tf.diag(scale_diag)`.

    Args:
      shift: Floating-point `Tensor`. If this is set to `None`, no shift is
        applied.
      scale_identity_multiplier: floating point rank 0 `Tensor` representing a
        scaling done to the identity matrix.
        When `scale_identity_multiplier = scale_diag = scale_tril = None` then
        `scale += IdentityMatrix`. Otherwise no scaled-identity-matrix is added
        to `scale`.
      scale_diag: Floating-point `Tensor` representing the diagonal matrix.
        `scale_diag` has shape [N1, N2, ...  k], which represents a k x k
        diagonal matrix.
        When `None` no diagonal term is added to `scale`.
      scale_tril: Floating-point `Tensor` representing the diagonal matrix.
        `scale_diag` has shape [N1, N2, ...  k, k], which represents a k x k
        lower triangular matrix.
        When `None` no `scale_tril` term is added to `scale`.
        The upper triangular elements above the diagonal are ignored.
      scale_perturb_factor: Floating-point `Tensor` representing factor matrix
        with last two dimensions of shape `(k, r)`. When `None`, no rank-r
        update is added to `scale`.
      scale_perturb_diag: Floating-point `Tensor` representing the diagonal
        matrix. `scale_perturb_diag` has shape [N1, N2, ...  r], which
        represents an `r x r` diagonal matrix. When `None` low rank updates will
        take the form `scale_perturb_factor * scale_perturb_factor.T`.
      event_ndims: Scalar `int32` `Tensor` indicating the number of dimensions
        associated with a particular draw from the distribution. Must be 0 or 1.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.

    Raises:
      ValueError: if `perturb_diag` is specified but not `perturb_factor`.
      TypeError: if `shift` has different `dtype` from `scale` arguments.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args
    # Ambiguous definition of low rank update.
    if scale_perturb_diag is not None and scale_perturb_factor is None:
      raise ValueError("When scale_perturb_diag is specified, "
                       "scale_perturb_factor must be specified.")
    # Special case, only handling a scaled identity matrix. We don't know its
    # dimensions, so this is special cased.
    # We don't check identity_multiplier, since below we set it to 1. if all
    # other scale args are None.
    self._is_only_identity_multiplier = (scale_tril is None and
                                         scale_diag is None and
                                         scale_perturb_factor is None)
    # When no args are specified, pretend the scale matrix is the identity
    # matrix.
    if self._is_only_identity_multiplier and scale_identity_multiplier is None:
      scale_identity_multiplier = 1.
    with self._name_scope("init", values=[
        shift, scale_identity_multiplier, scale_diag, scale_tril,
        scale_perturb_diag, scale_perturb_factor, event_ndims]):
      event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
      if validate_args:
        is_less_than_two = check_ops.assert_less(
            event_ndims, 2,
            message="event_ndims must be 0 or 1")
        event_ndims = control_flow_ops.with_dependencies(
            [is_less_than_two], event_ndims)
      self._shift = _as_tensor(shift, "shift")
      # self._create_scale_operator returns an OperatorPD in all cases except if
      # self._is_only_identity_multiplier; in which case it returns a scalar
      # Tensor.
      self._scale = self._create_scale_operator(
          identity_multiplier=scale_identity_multiplier,
          diag=scale_diag,
          tril=scale_tril,
          perturb_diag=scale_perturb_diag,
          perturb_factor=scale_perturb_factor,
          event_ndims=event_ndims,
          validate_args=validate_args)
      if (self._shift is not None and
          self._shift.dtype.base_dtype != self._scale.dtype.base_dtype):
        raise TypeError("shift.dtype({}) does not match scale.dtype({})".format(
            self._shift.dtype, self._scale.dtype))
      self._shaper = _DistributionShape(
          batch_ndims=self._infer_batch_ndims(),
          event_ndims=event_ndims,
          validate_args=validate_args)
      super(Affine, self).__init__(
          event_ndims=event_ndims,
          graph_parents=(
              [event_ndims] +
              [self._scale] if tensor_util.is_tensor(self._scale)
              else self._scale.inputs +
              [self._shift] if self._shift is not None else []),
          is_constant_jacobian=True,
          dtype=self._scale.dtype,
          validate_args=validate_args,
          name=name)

  def _create_scale_operator(self, identity_multiplier, diag, tril,
                             perturb_diag, perturb_factor, event_ndims,
                             validate_args):
    """Construct `scale` from various components.

    Args:
      identity_multiplier: floating point rank 0 `Tensor` representing a scaling
        done to the identity matrix.
      diag: Floating-point `Tensor` representing the diagonal matrix.
        `scale_diag` has shape [N1, N2, ...  k], which represents a k x k
        diagonal matrix.
      tril: Floating-point `Tensor` representing the diagonal matrix.
        `scale_tril` has shape [N1, N2, ...  k], which represents a k x k lower
        triangular matrix.
      perturb_diag: Floating-point `Tensor` representing the diagonal matrix of
        the low rank update.
      perturb_factor: Floating-point `Tensor` representing factor matrix.
      event_ndims: Scalar `int32` `Tensor` indicating the number of dimensions
        associated with a particular draw from the distribution. Must be 0 or 1
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.

    Returns:
      scale. In the case of scaling by a constant, scale is a
      floating point `Tensor`. Otherwise, scale is an `OperatorPD`.

    Raises:
      ValueError: if all of `tril`, `diag` and `identity_multiplier` are `None`.
    """
    identity_multiplier = _as_tensor(identity_multiplier, "identity_multiplier")
    diag = _as_tensor(diag, "diag")
    tril = _as_tensor(tril, "tril")
    perturb_diag = _as_tensor(perturb_diag, "perturb_diag")
    perturb_factor = _as_tensor(perturb_factor, "perturb_factor")

    identity_multiplier = self._maybe_validate_identity_multiplier(
        identity_multiplier, validate_args)

    if perturb_factor is not None:
      perturb_factor = self._process_matrix(
          perturb_factor, min_rank=2, event_ndims=event_ndims)

    if perturb_diag is not None:
      perturb_diag = self._process_matrix(
          perturb_diag, min_rank=1, event_ndims=event_ndims)

    # The following if-statments are ordered by increasingly stronger
    # assumptions in the base matrix, i.e., we process in the order:
    # TriL, Diag, Identity.

    if tril is not None:
      tril = self._preprocess_tril(
          identity_multiplier, diag, tril, event_ndims)
      if perturb_factor is None:
        return operator_pd_cholesky.OperatorPDCholesky(
            tril, verify_pd=validate_args)
      return _TriLPlusVDVTLightweightOperatorPD(
          tril=tril, v=perturb_factor, diag=perturb_diag,
          validate_args=validate_args)

    if diag is not None:
      diag = self._preprocess_diag(identity_multiplier, diag, event_ndims)
      if perturb_factor is None:
        return operator_pd_diag.OperatorPDSqrtDiag(
            diag, verify_pd=validate_args)
      return operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
          operator=operator_pd_diag.OperatorPDDiag(
              diag, verify_pd=validate_args),
          v=perturb_factor,
          diag=perturb_diag,
          verify_pd=validate_args)

    if identity_multiplier is not None:
      if perturb_factor is None:
        return identity_multiplier
      # Infer the shape from the V and D.
      v_shape = array_ops.shape(perturb_factor)
      identity_shape = array_ops.concat([v_shape[:-1], [v_shape[-2]]], 0)
      scaled_identity = operator_pd_identity.OperatorPDIdentity(
          identity_shape,
          perturb_factor.dtype.base_dtype,
          scale=identity_multiplier,
          verify_pd=validate_args)
      return operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
          operator=scaled_identity,
          v=perturb_factor,
          diag=perturb_diag,
          verify_pd=validate_args)

    raise ValueError("One of tril, diag and/or identity_multiplier must be "
                     "specified.")

  def _maybe_validate_identity_multiplier(self, identity_multiplier,
                                          validate_args):
    """Check that the init arg `identity_multiplier` is valid."""
    if identity_multiplier is None or not validate_args:
      return identity_multiplier
    if validate_args:
      identity_multiplier = control_flow_ops.with_dependencies(
          [check_ops.assert_positive(identity_multiplier)],
          identity_multiplier)
    return identity_multiplier

  def _preprocess_tril(self, identity_multiplier, diag, tril, event_ndims):
    """Helper to preprocess a lower triangular matrix."""
    tril = array_ops.matrix_band_part(tril, -1, 0)  # Zero out TriU.
    if identity_multiplier is None and diag is None:
      return self._process_matrix(tril, min_rank=2, event_ndims=event_ndims)
    new_diag = array_ops.matrix_diag_part(tril)
    if identity_multiplier is not None:
      new_diag += identity_multiplier
    if diag is not None:
      new_diag += diag
    tril = array_ops.matrix_set_diag(tril, new_diag)
    return self._process_matrix(tril, min_rank=2, event_ndims=event_ndims)

  def _preprocess_diag(self, identity_multiplier, diag, event_ndims):
    """Helper to preprocess a diagonal matrix."""
    if identity_multiplier is not None:
      diag += identity_multiplier
    return self._process_matrix(diag, min_rank=1, event_ndims=event_ndims)

  def _process_matrix(self, matrix, min_rank, event_ndims):
    """Helper to __init__ which gets matrix in batch-ready form."""
    # Pad the matrix so that matmul works in the case of a matrix and vector
    # input. Keep track if the matrix was padded, to distinguish between a
    # rank 3 tensor and a padded rank 2 tensor.
    # TODO(srvasude): Remove side-effects from functions. Its currently unbroken
    # but error-prone since the function call order may change in the future.
    self._rank_two_event_ndims_one = math_ops.logical_and(
        math_ops.equal(array_ops.rank(matrix), min_rank),
        math_ops.equal(event_ndims, 1))
    left = array_ops.where(self._rank_two_event_ndims_one, 1, 0)
    pad = array_ops.concat(
        [array_ops.ones(
            [left], dtype=dtypes.int32), array_ops.shape(matrix)],
        0)
    return array_ops.reshape(matrix, pad)

  def _infer_batch_ndims(self):
    """Return batch_ndims."""
    if self._is_only_identity_multiplier:
      return 0
    # The real batch dims is one less when we pad in the case of event_ndims =
    # 1, and the rank of the underlying scale being 2. This allows us to have
    # non-negative sample dims.
    return (self._scale.rank() - 2 -
            array_ops.where(self._rank_two_event_ndims_one, 1, 0))

  @property
  def shift(self):
    """The `shift` `Tensor` in `Y = scale @ X + shift`."""
    return self._shift

  @property
  def scale(self):
    """The `scale` `LinearOperator` in `Y = scale @ X + shift`."""
    # TODO(srvasude): Remove this exception once TriLPlusVDVT is properly
    # implemented.
    if isinstance(self._scale, _TriLPlusVDVTLightweightOperatorPD):
      raise NotImplementedError("Cannot access scale when Tril+VDV.T.")
    return self._scale

  def _forward(self, x):
    y = x
    if self._is_only_identity_multiplier:
      y *= self._scale
      if self.shift is not None:
        return y + self.shift
      return  y
    y, sample_shape = self._shaper.make_batch_of_event_sample_matrices(y)
    y = self._scale.sqrt_matmul(y)
    y = self._shaper.undo_make_batch_of_event_sample_matrices(y, sample_shape)
    if self.shift is not None:
      return y + self.shift
    return y

  def _inverse(self, y):
    x = y
    if self.shift is not None:
      x -= self.shift
    if self._is_only_identity_multiplier:
      return x / self._scale
    x, sample_shape = self._shaper.make_batch_of_event_sample_matrices(x)
    x = self._scale.sqrt_solve(x)
    x = self._shaper.undo_make_batch_of_event_sample_matrices(x, sample_shape)
    return x

  def _inverse_log_det_jacobian(self, y):
    return -self._forward_log_det_jacobian(y)

  def _forward_log_det_jacobian(self, x):
    if self._is_only_identity_multiplier:
      # TODO(jvdillon): We don't pad in this case and instead let the fldj be
      # applied via broadcast.
      d = math_ops.cast(array_ops.shape(x)[-1], dtype=self._scale.dtype)
      return math_ops.log(math_ops.abs(self._scale)) * array_ops.where(
          math_ops.equal(self._shaper.event_ndims, 0), 1., d)
    fldj = self._scale.sqrt_log_abs_det()
    # We need to squeeze off the padded dimension.
    start = array_ops.where(self._rank_two_event_ndims_one, 1, 0)
    return array_ops.reshape(fldj, array_ops.shape(fldj)[start:])

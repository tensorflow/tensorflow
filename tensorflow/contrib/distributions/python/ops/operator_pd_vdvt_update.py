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
"""Operator defined: `A = SS^T` where `S = M + VDV^T`, for `OperatorPD` `M`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import operator_pd
from tensorflow.contrib.distributions.python.ops import operator_pd_diag
from tensorflow.contrib.distributions.python.ops import operator_pd_identity
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


class OperatorPDSqrtVDVTUpdate(operator_pd.OperatorPDBase):
  r"""Operator defined by `A=SS^T`, where `S = M + VDV^T` for `OperatorPD` `M`.

  This provides efficient low-rank updates of arbitrary `OperatorPD`.

  Some math:

  Given positive definite operator representing positive definite (batch) matrix
  `M` in `R^{k x k}`, diagonal matrix `D` in `R^{r x r}`, and low rank `V` in
  `R^{k x r}` this class represents the batch matrix `A`, defined by its square
  root `S` as follows:

  ```
  A = SS^T, where
  S := M + VDV^T
  ```

  Defining an operator in terms of its square root means that
  `A_{ij} = S_i S_j^T`, where `S_i` is the ith row of `S`.  The update
  `VDV^T` has `ij` coordinate equal to `sum_k V_{ik} D_{kk} V_{jk}`.

  Computational efficiency:

  Defining `A` via its square root eliminates the need to compute the square
  root.

  Performance depends on the operator representing `M`, the batch size `B`, and
  the width of the matrix being multiplied, or systems being solved `L`.

  Since `V` is rank `r`, the update adds

  * `O(B L k r)` to matmul, which requires a call to `M.matmul`.
  * `O(B L r^3)` to solves, which require a call to `M.solve` as well as the
    solution to a batch of rank `r` systems.
  * `O(B r^3)` to determinants, which require a call to `M.solve` as well as the
    solution to a batch of rank `r` systems.

  The rank `r` solve and determinant are both done through a Cholesky
  factorization, thus some computation is shared.

  See
    https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    https://en.wikipedia.org/wiki/Matrix_determinant_lemma
  """

  # Note that diag must be nonsingular to use Woodbury lemma, and must be
  # positive def to use a Cholesky factorization, so we enforce that here.
  def __init__(self,
               operator,
               v,
               diag=None,
               verify_pd=True,
               verify_shapes=True,
               name="OperatorPDSqrtVDVTUpdate"):
    """Initialize an `OperatorPDSqrtVDVTUpdate`.

    Args:
      operator:  Subclass of `OperatorPDBase`.  Represents the (batch) positive
        definite matrix `M` in `R^{k x k}`.
      v: `Tensor` defining batch matrix of same `dtype` and `batch_shape` as
        `operator`, and last two dimensions of shape `(k, r)`.
      diag:  Optional `Tensor` defining batch vector of same `dtype` and
        `batch_shape` as `operator`, and last dimension of size `r`.  If `None`,
        the update becomes `VV^T` rather than `VDV^T`.
      verify_pd:  `Boolean`.  If `True`, add asserts that `diag > 0`, which,
        along with the positive definiteness of `operator`, is sufficient to
        make the resulting operator positive definite.
      verify_shapes:  `Boolean`.  If `True`, check that `operator`, `v`, and
        `diag` have compatible shapes.
      name:  A name to prepend to `Op` names.
    """

    if not isinstance(operator, operator_pd.OperatorPDBase):
      raise TypeError("operator was not instance of OperatorPDBase.")

    with ops.name_scope(name):
      with ops.name_scope("init", values=operator.inputs + [v, diag]):
        self._operator = operator
        self._v = ops.convert_to_tensor(v, name="v")
        self._verify_pd = verify_pd
        self._verify_shapes = verify_shapes
        self._name = name

        # This operator will be PD so long as the diag is PSD, but Woodbury
        # and determinant lemmas require diag to be PD.  So require diag PD
        # whenever we ask to "verify_pd".
        if diag is not None:
          self._diag = ops.convert_to_tensor(diag, name="diag")
          self._diag_operator = operator_pd_diag.OperatorPDDiag(
              diag, verify_pd=self.verify_pd)
          # No need to verify that the inverse of a PD is PD.
          self._diag_inv_operator = operator_pd_diag.OperatorPDDiag(
              1 / self._diag, verify_pd=False)
        else:
          self._diag = None
          self._diag_operator = self._get_identity_operator(self._v)
          self._diag_inv_operator = self._diag_operator

        self._check_types(operator, self._v, self._diag)
        # Always check static.
        checked = self._check_shapes_static(operator, self._v, self._diag)
        if not checked and self._verify_shapes:
          self._v, self._diag = self._check_shapes_dynamic(
              operator, self._v, self._diag)

  def _get_identity_operator(self, v):
    """Get an `OperatorPDIdentity` to play the role of `D` in `VDV^T`."""
    with ops.name_scope("get_identity_operator", values=[v]):
      if v.get_shape().is_fully_defined():
        v_shape = v.get_shape().as_list()
        v_batch_shape = v_shape[:-2]
        r = v_shape[-1]
        id_shape = v_batch_shape + [r, r]
      else:
        v_shape = array_ops.shape(v)
        v_rank = array_ops.rank(v)
        v_batch_shape = array_ops.strided_slice(v_shape, [0], [v_rank - 2])
        r = array_ops.gather(v_shape, v_rank - 1)  # Last dim of v
        id_shape = array_ops.concat_v2((v_batch_shape, [r, r]), 0)
      return operator_pd_identity.OperatorPDIdentity(
          id_shape, v.dtype, verify_pd=self._verify_pd)

  def _check_types(self, operator, v, diag):
    def msg():
      string = (
          "dtypes must match:  Found operator.dtype = %s, v.dtype = %s"
          % (operator.dtype, v.dtype))
      return string

    if operator.dtype != v.dtype:
      raise TypeError(msg())
    if diag is not None:
      if diag.dtype != v.dtype:
        raise TypeError("%s, diag.dtype = %s" % (msg(), diag.dtype))

  def _check_shapes_static(self, operator, v, diag):
    """True if they are compatible. Raise if not. False if could not check."""
    def msg():
      # Error message when shapes don't match.
      string = "  Found: operator.shape = %s, v.shape = %s" % (s_op, s_v)
      if diag is not None:
        string += ", diag.shape = " % s_d
      return string

    s_op = operator.get_shape()
    s_v = v.get_shape()

    # If everything is not fully defined, return False because we couldn"t check
    if not (s_op.is_fully_defined() and s_v.is_fully_defined()):
      return False
    if diag is not None:
      s_d = diag.get_shape()
      if not s_d.is_fully_defined():
        return False

    # Now perform the checks, raising ValueError if they fail.

    # Check tensor rank.
    if s_v.ndims != s_op.ndims:
      raise ValueError("v should have same rank as operator" + msg())
    if diag is not None:
      if s_d.ndims != s_op.ndims - 1:
        raise ValueError("diag should have rank 1 less than operator" + msg())

    # Check batch shape
    if s_v[:-2] != s_op[:-2]:
      raise ValueError("v and operator should have same batch shape" + msg())
    if diag is not None:
      if s_d[:-1] != s_op[:-2]:
        raise ValueError(
            "diag and operator should have same batch shape" + msg())

    # Check event shape
    if s_v[-2] != s_op[-1]:
      raise ValueError(
          "v and operator should be compatible for matmul" + msg())
    if diag is not None:
      if s_d[-1] != s_v[-1]:
        raise ValueError("diag and v should have same last dimension" + msg())

    return True

  def _check_shapes_dynamic(self, operator, v, diag):
    """Return (v, diag) with Assert dependencies, which check shape."""
    checks = []
    with ops.name_scope("check_shapes", values=[operator, v, diag]):
      s_v = array_ops.shape(v)
      r_op = operator.rank()
      r_v = array_ops.rank(v)
      if diag is not None:
        s_d = array_ops.shape(diag)
        r_d = array_ops.rank(diag)

      # Check tensor rank.
      checks.append(check_ops.assert_rank(
          v, r_op, message="v is not the same rank as operator."))
      if diag is not None:
        checks.append(check_ops.assert_rank(
            diag, r_op - 1, message="diag is not the same rank as operator."))

      # Check batch shape
      checks.append(check_ops.assert_equal(
          operator.batch_shape(), array_ops.strided_slice(s_v, [0], [r_v - 2]),
          message="v does not have same batch shape as operator."))
      if diag is not None:
        checks.append(check_ops.assert_equal(
            operator.batch_shape(), array_ops.strided_slice(
                s_d, [0], [r_d - 1]),
            message="diag does not have same batch shape as operator."))

      # Check event shape
      checks.append(check_ops.assert_equal(
          operator.vector_space_dimension(), array_ops.gather(s_v, r_v - 2),
          message="v does not have same event shape as operator."))
      if diag is not None:
        checks.append(check_ops.assert_equal(
            array_ops.gather(s_v, r_v - 1), array_ops.gather(s_d, r_d - 1),
            message="diag does not have same event shape as v."))

      v = control_flow_ops.with_dependencies(checks, v)
      if diag is not None:
        diag = control_flow_ops.with_dependencies(checks, diag)
      return v, diag

  @property
  def name(self):
    """String name identifying this `Operator`."""
    return self._name

  @property
  def verify_pd(self):
    """Whether to verify that this `Operator` is positive definite."""
    return self._verify_pd

  @property
  def dtype(self):
    """Data type of matrix elements of `A`."""
    return self._v.dtype

  def _inv_quadratic_form_on_vectors(self, x):
    return self._iqfov_via_sqrt_solve(x)

  @property
  def inputs(self):
    """List of tensors that were provided as initialization inputs."""
    return self._operator.inputs + self._diag_operator.inputs + [self._v]

  def get_shape(self):
    """Static `TensorShape` of entire operator.

    If this operator represents the batch matrix `A` with
    `A.shape = [N1,...,Nn, k, k]`, then this returns
    `TensorShape([N1,...,Nn, k, k])`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    return self._operator.get_shape()

  def _shape(self):
    return self._operator.shape()

  def _det(self):
    return math_ops.exp(self.log_det())

  def _batch_log_det(self):
    return 2 * self._batch_sqrt_log_det()

  def _log_det(self):
    return 2 * self._sqrt_log_det()

  def _sqrt_log_det(self):
    # The matrix determinant lemma states:
    # det(M + VDV^T) = det(D^{-1} + V^T M^{-1} V) * det(D) * det(M)
    #                = det(C) * det(D) * det(M)
    #
    # Here we compute the Cholesky factor of "C", then pass the result on.
    abs_diag_chol_c = math_ops.abs(array_ops.matrix_diag_part(
        self._chol_capacitance(batch_mode=False)))
    return self._sqrt_log_det_core(abs_diag_chol_c)

  def _batch_sqrt_log_det(self):
    # Here we compute the Cholesky factor of "C", then pass the result on.
    abs_diag_chol_c = math_ops.abs(array_ops.matrix_diag_part(
        self._chol_capacitance(batch_mode=True)))
    return self._sqrt_log_det_core(abs_diag_chol_c)

  def _chol_capacitance(self, batch_mode):
    """Cholesky factorization of the capacitance term."""
    # Cholesky factor for (D^{-1} + V^T M^{-1} V), which is sometimes
    # known as the "capacitance" matrix.
    # We can do a Cholesky decomposition, since a priori M is a
    # positive-definite Hermitian matrix, which causes the "capacitance" to
    # also be positive-definite Hermitian, and thus have a Cholesky
    # decomposition.

    # self._operator will use batch if need be. Automatically.  We cannot force
    # that here.
    # M^{-1} V
    minv_v = self._operator.solve(self._v)
    # V^T M^{-1} V
    vt_minv_v = math_ops.matmul(self._v, minv_v, adjoint_a=True)

    # D^{-1} + V^T M^{-1} V
    capacitance = self._diag_inv_operator.add_to_tensor(vt_minv_v)
    # Cholesky[D^{-1} + V^T M^{-1} V]
    return linalg_ops.cholesky(capacitance)

  def _sqrt_log_det_core(self, diag_chol_c):
    """Finish computation of Sqrt[Log[Det]]."""
    # Complete computation of ._log_det and ._batch_log_det, after the initial
    # Cholesky factor has been taken with the appropriate batch/non-batch method

    # det(M + VDV^T) = det(D^{-1} + V^T M^{-1} V) * det(D) * det(M)
    #                = det(C) * det(D) * det(M)
    # Multiply by 2 here because this is the log-det of the Cholesky factor of C
    log_det_c = 2 * math_ops.reduce_sum(
        math_ops.log(math_ops.abs(diag_chol_c)),
        reduction_indices=[-1])
    # Add together to get Log[det(M + VDV^T)], the Log-det of the updated square
    # root.
    log_det_updated_sqrt = (
        log_det_c + self._diag_operator.log_det() + self._operator.log_det())
    return log_det_updated_sqrt

  def _batch_matmul(self, x, transpose_x=False):
    # Since the square root is PD, it is symmetric, and so A = SS^T = SS.
    s_x = self._batch_sqrt_matmul(x, transpose_x=transpose_x)
    return self._batch_sqrt_matmul(s_x)

  def _matmul(self, x, transpose_x=False):
    # Since the square root is PD, it is symmetric, and so A = SS^T = SS.
    s_x = self._sqrt_matmul(x, transpose_x=transpose_x)
    return self._sqrt_matmul(s_x)

  def _batch_sqrt_matmul(self, x, transpose_x=False):
    v = self._v
    m = self._operator
    d = self._diag_operator
    # The operators call the appropriate matmul/batch_matmul automatically.
    # We cannot override.
    # batch_matmul is defined as:  x * y, so adjoint_a and adjoint_b are the
    # ways to transpose the left and right.
    mx = m.matmul(x, transpose_x=transpose_x)
    vt_x = math_ops.matmul(v, x, adjoint_a=True, adjoint_b=transpose_x)
    d_vt_x = d.matmul(vt_x)
    v_d_vt_x = math_ops.matmul(v, d_vt_x)

    return mx + v_d_vt_x

  def _sqrt_matmul(self, x, transpose_x=False):
    v = self._v
    m = self._operator
    d = self._diag_operator
    # The operators call the appropriate matmul/batch_matmul automatically.  We
    # cannot override.
    # matmul is defined as:  a * b, so transpose_a, transpose_b are used.
    # transpose the left and right.
    mx = m.matmul(x, transpose_x=transpose_x)
    vt_x = math_ops.matmul(v, x, transpose_a=True, transpose_b=transpose_x)
    d_vt_x = d.matmul(vt_x)
    v_d_vt_x = math_ops.matmul(v, d_vt_x)

    return mx + v_d_vt_x

  def _solve(self, rhs):
    # This operator represents A = SS^T, but S is symmetric, so A = SS,
    # which means A^{-1} = S^{-1}S^{-2}
    # S^{-1} rhs
    sqrtinv_rhs = self._sqrt_solve(rhs)
    return self._sqrt_solve(sqrtinv_rhs)

  def _batch_solve(self, rhs):
    sqrtinv_rhs = self._batch_sqrt_solve(rhs)
    return self._batch_sqrt_solve(sqrtinv_rhs)

  def _sqrt_solve(self, rhs):
    # Recall the square root of this operator is M + VDV^T.
    # The Woodbury formula gives:
    # (M + VDV^T)^{-1}
    # = M^{-1} - M^{-1} V (D^{-1} + V^T M^{-1} V)^{-1} V^T M^{-1}
    # = M^{-1} - M^{-1} V C^{-1} V^T M^{-1}
    # where C is the capacitance matrix.
    # TODO(jvdillon) Determine if recursively applying rank-1 updates is more
    # efficient.  May not be possible because a general n x n matrix can be
    # represeneted as n rank-1 updates, and solving with this matrix is always
    # done in O(n^3) time.
    m = self._operator
    v = self._v
    cchol = self._chol_capacitance(batch_mode=False)

    # The operators will use batch/singleton mode automatically.  We don't
    # override.
    # M^{-1} rhs
    minv_rhs = m.solve(rhs)
    # V^T M^{-1} rhs
    vt_minv_rhs = math_ops.matmul(v, minv_rhs, transpose_a=True)
    # C^{-1} V^T M^{-1} rhs
    cinv_vt_minv_rhs = linalg_ops.cholesky_solve(cchol, vt_minv_rhs)
    # V C^{-1} V^T M^{-1} rhs
    v_cinv_vt_minv_rhs = math_ops.matmul(v, cinv_vt_minv_rhs)
    # M^{-1} V C^{-1} V^T M^{-1} rhs
    minv_v_cinv_vt_minv_rhs = m.solve(v_cinv_vt_minv_rhs)

    # M^{-1} - M^{-1} V C^{-1} V^T M^{-1}
    return minv_rhs - minv_v_cinv_vt_minv_rhs

  def _batch_sqrt_solve(self, rhs):
    # Recall the square root of this operator is M + VDV^T.
    # The Woodbury formula gives:
    # (M + VDV^T)^{-1}
    # = M^{-1} - M^{-1} V (D^{-1} + V^T M^{-1} V)^{-1} V^T M^{-1}
    # = M^{-1} - M^{-1} V C^{-1} V^T M^{-1}
    # where C is the capacitance matrix.
    m = self._operator
    v = self._v
    cchol = self._chol_capacitance(batch_mode=True)

    # The operators will use batch/singleton mode automatically.  We don't
    # override.
    # M^{-1} rhs
    minv_rhs = m.solve(rhs)
    # V^T M^{-1} rhs
    vt_minv_rhs = math_ops.matmul(v, minv_rhs, adjoint_a=True)
    # C^{-1} V^T M^{-1} rhs
    cinv_vt_minv_rhs = linalg_ops.cholesky_solve(cchol, vt_minv_rhs)
    # V C^{-1} V^T M^{-1} rhs
    v_cinv_vt_minv_rhs = math_ops.matmul(v, cinv_vt_minv_rhs)
    # M^{-1} V C^{-1} V^T M^{-1} rhs
    minv_v_cinv_vt_minv_rhs = m.solve(v_cinv_vt_minv_rhs)

    # M^{-1} - M^{-1} V C^{-1} V^T M^{-1}
    return minv_rhs - minv_v_cinv_vt_minv_rhs

  def _to_dense(self):
    sqrt = self.sqrt_to_dense()
    return math_ops.matmul(sqrt, sqrt, adjoint_b=True)

  def _sqrt_to_dense(self):
    v = self._v
    d = self._diag_operator
    m = self._operator

    d_vt = d.matmul(v, transpose_x=True)
    # Batch op won't be efficient for singletons.  Currently we don't break
    # to_dense into batch/singleton methods.
    v_d_vt = math_ops.matmul(v, d_vt)
    m_plus_v_d_vt = m.to_dense() + v_d_vt
    return m_plus_v_d_vt

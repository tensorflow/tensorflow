# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Gradients for operators defined in linalg_ops.py.

Useful reference for derivative formulas is (Mike Giles, 2008).

Ionescu et al. (2015) provide a detailed derivation of formulas for
backpropagating through spectral layers (SVD and Eig).

References:
  An extended collection of matrix derivative results for
  forward and reverse mode automatic differentiation:
    [Mike Giles, 2008](https://ora.ox.ac.uk/objects/uuid:8d0c0a29-c92b-4153-a1d2-38b276e93124)
    ([pdf](http://eprints.maths.ox.ac.uk/1079/1/NA-08-01.pdf))
  Matrix Backpropagation for Deep Networks with Structured Layers
    [Ionescu et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.html)
    ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.pdf))
  Training Deep Networks with Structured Layers by Matrix Backpropagation:
    [Ionescu et al., 2015](https://arxiv.org/abs/1509.07838)
    ([pdf](https://arxiv.org/pdf/1509.07838.pdf))
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg


@ops.RegisterGradient("MatrixInverse")
def _MatrixInverseGrad(op, grad):
  """Gradient for MatrixInverse."""
  ainv = op.outputs[0]
  return -math_ops.matmul(
      ainv, math_ops.matmul(grad, ainv, adjoint_b=True), adjoint_a=True)


@ops.RegisterGradient("MatrixDeterminant")
def _MatrixDeterminantGrad(op, grad):
  """Gradient for MatrixDeterminant."""
  a = op.inputs[0]
  c = op.outputs[0]
  a_adj_inv = linalg_ops.matrix_inverse(a, adjoint=True)
  multipliers = array_ops.reshape(grad * c,
                                  array_ops.concat([array_ops.shape(c), [1, 1]],
                                                   0))
  return multipliers * a_adj_inv


@ops.RegisterGradient("MatrixSquareRoot")
def _MatrixSquareRootGrad(op, grad):
  """Gradient for MatrixSquareRoot."""

  # Let A be an m x m square matrix (or batch of matrices)
  # Let R = sqrtm(A)
  # By definition, A = RR
  # Take the differential: dA = d(RR) = RdR + dRR
  # Solve the resulting Sylvester equation for dR

  # Used to find Kronecker products within the Sylvester equation
  def _KroneckerProduct(b1, b2):
    """Computes the Kronecker product of two batches of square matrices"""
    b1_shape = array_ops.shape(b1)
    b2_shape = array_ops.shape(b2)
    b1_order = b1_shape[-1]
    b2_order = b2_shape[-1]

    shape_slice_size = [math_ops.subtract(array_ops.size(b1_shape), 2)]
    shape_slice = array_ops.slice(b1_shape, [0],
                                  shape_slice_size)  # Same for both batches
    b1_reshape_shape = array_ops.concat(
        [shape_slice, [b1_order], [1], [b1_order], [1]], 0)
    b2_reshape_shape = array_ops.concat(
        [shape_slice, [1], [b2_order], [1], [b2_order]], 0)

    b1_reshape = array_ops.reshape(b1, b1_reshape_shape)
    b2_reshape = array_ops.reshape(b2, b2_reshape_shape)

    order_prod = b1_order * b2_order
    kprod_shape = array_ops.concat([shape_slice, [order_prod], [order_prod]], 0)
    return array_ops.reshape(b1_reshape * b2_reshape, kprod_shape)

  sqrtm = op.outputs[0]  # R
  shape = array_ops.shape(sqrtm)
  order = shape[-1]  # m
  matrix_count = math_ops.reduce_prod(shape[0:-2])

  # Get batch of m x m identity matrices
  eye = linalg_ops.eye(order, dtype=sqrtm.dtype)  # m x m identity matrix
  eye_flat = array_ops.reshape(eye, [-1])
  eye_tiled = array_ops.tile(eye_flat, [matrix_count])
  eye_batch = array_ops.reshape(eye_tiled, shape)

  # The transpose of R is taken in the k1 term instead of k2 in
  # order to prevent redundant transposition of R (i.e. (R')' = R)
  sqrtm_transpose = array_ops.matrix_transpose(sqrtm)
  k1 = _KroneckerProduct(eye_batch, sqrtm_transpose)
  k2 = _KroneckerProduct(sqrtm, eye_batch)
  ksum = math_ops.add(k1, k2)

  # Vectorize dA
  shape_slice_size = [math_ops.subtract(array_ops.size(shape), 2)]
  shape_slice = array_ops.slice(shape, [0], shape_slice_size)
  shape_vec_da = array_ops.concat([shape_slice, [order * order], [1]], 0)
  vec_da = array_ops.reshape(array_ops.matrix_transpose(grad), shape_vec_da)

  # Solve for vec(dR)
  vec_dsqrtm = linalg_ops.matrix_solve(ksum, vec_da)

  # Solve for dR by inverse vectorizing vec(dR)
  dsqrtm_transpose = array_ops.reshape(vec_dsqrtm, shape)
  return array_ops.matrix_transpose(dsqrtm_transpose)


@ops.RegisterGradient("LogMatrixDeterminant")
def _LogMatrixDeterminantGrad(op, _, grad_b):
  """Gradient for LogMatrixDeterminant."""
  a = op.inputs[0]
  c = op.outputs[1]
  a_adj_inv = linalg_ops.matrix_inverse(a, adjoint=True)
  multipliers = array_ops.reshape(
      grad_b, array_ops.concat([array_ops.shape(c), [1, 1]], 0))
  return multipliers * a_adj_inv


@ops.RegisterGradient("Cholesky")
def _CholeskyGrad(op, grad):
  """Gradient for Cholesky."""

  # Gradient is l^{-H} @ ((l^{H} @ grad) * (tril(ones)-1/2*eye)) @ l^{-1}
  l = op.outputs[0]
  num_rows = array_ops.shape(l)[-1]
  batch_shape = array_ops.shape(l)[:-2]
  l_inverse = linalg_ops.matrix_triangular_solve(l,
                                                 linalg_ops.eye(
                                                     num_rows,
                                                     batch_shape=batch_shape,
                                                     dtype=l.dtype))

  middle = math_ops.matmul(l, grad, adjoint_a=True)
  middle = array_ops.matrix_set_diag(middle,
                                     0.5 * array_ops.matrix_diag_part(middle))
  middle = array_ops.matrix_band_part(middle, -1, 0)

  grad_a = math_ops.matmul(
      math_ops.matmul(l_inverse, middle, adjoint_a=True), l_inverse)

  grad_a += _linalg.adjoint(grad_a)
  return grad_a * 0.5


@ops.RegisterGradient("Qr")
def _QrGrad(op, dq, dr):
  """Gradient for Qr."""
  q, r = op.outputs
  if q.dtype.is_complex:
    raise NotImplementedError("QrGrad not implemented for dtype: %s" % q.dtype)
  if (r.shape.ndims is None or r.shape.as_list()[-2] is None or
      r.shape.as_list()[-1] is None):
    raise NotImplementedError("QrGrad not implemented with dynamic shapes.")
  if r.shape.dims[-2].value != r.shape.dims[-1].value:
    raise NotImplementedError("QrGrad not implemented when ncols > nrows "
                              "or full_matrices is true and ncols != nrows.")

  qdq = math_ops.matmul(q, dq, adjoint_a=True)
  qdq_ = qdq - _linalg.adjoint(qdq)
  rdr = math_ops.matmul(r, dr, adjoint_b=True)
  rdr_ = rdr - _linalg.adjoint(rdr)
  tril = array_ops.matrix_band_part(qdq_ + rdr_, -1, 0)

  def _TriangularSolve(x, r):
    """Equiv to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri."""
    return _linalg.adjoint(
        linalg_ops.matrix_triangular_solve(
            r, _linalg.adjoint(x), lower=False, adjoint=False))

  grad_a = math_ops.matmul(q, dr + _TriangularSolve(tril, r))
  grad_b = _TriangularSolve(dq - math_ops.matmul(q, qdq), r)
  return grad_a + grad_b


@ops.RegisterGradient("MatrixSolve")
def _MatrixSolveGrad(op, grad):
  """Gradient for MatrixSolve."""
  a = op.inputs[0]
  adjoint_a = op.get_attr("adjoint")
  c = op.outputs[0]
  grad_b = linalg_ops.matrix_solve(a, grad, adjoint=not adjoint_a)
  if adjoint_a:
    grad_a = -math_ops.matmul(c, grad_b, adjoint_b=True)
  else:
    grad_a = -math_ops.matmul(grad_b, c, adjoint_b=True)
  return (grad_a, grad_b)


@ops.RegisterGradient("MatrixSolveLs")
def _MatrixSolveLsGrad(op, grad):
  """Gradients for MatrixSolveLs."""

  # TODO(rmlarsen): The implementation could be more efficient:
  #   a) Output the Cholesky factorization from forward op instead of
  #      recomputing it here.
  #   b) Implement a symmetric rank-k update op instead of computing
  #      x*z + transpose(x*z). This pattern occurs other places in TensorFlow.

  def _Overdetermined(op, grad):
    """Gradients for the overdetermined case of MatrixSolveLs.

    This is the backprop for the solution to the normal equations of the first
    kind:
       X = F(A, B) = (A^T * A + lambda * I)^{-1} * A^T * B
    which solve the least squares problem
       min ||A * X - B||_F^2 + lambda ||X||_F^2.
    """
    a = op.inputs[0]
    b = op.inputs[1]
    x = op.outputs[0]
    l2_regularizer = math_ops.cast(op.inputs[2], a.dtype.base_dtype)
    # pylint: disable=protected-access
    chol = linalg_ops._RegularizedGramianCholesky(
        a, l2_regularizer=l2_regularizer, first_kind=True)
    # pylint: enable=protected-access
    # Temporary z = (A^T * A + lambda * I)^{-1} * grad.
    z = linalg_ops.cholesky_solve(chol, grad)
    xzt = math_ops.matmul(x, z, adjoint_b=True)
    zx_sym = xzt + array_ops.matrix_transpose(xzt)
    grad_a = -math_ops.matmul(a, zx_sym) + math_ops.matmul(b, z, adjoint_b=True)
    grad_b = math_ops.matmul(a, z)
    return (grad_a, grad_b, None)

  def _Underdetermined(op, grad):
    """Gradients for the underdetermined case of MatrixSolveLs.

    This is the backprop for the solution to the normal equations of the second
    kind:
      X = F(A, B) = A * (A*A^T + lambda*I)^{-1} * B
    that (for lambda=0) solve the least squares problem
      min ||X||_F subject to A*X = B.
    """
    a = op.inputs[0]
    b = op.inputs[1]
    l2_regularizer = math_ops.cast(op.inputs[2], a.dtype.base_dtype)
    # pylint: disable=protected-access
    chol = linalg_ops._RegularizedGramianCholesky(
        a, l2_regularizer=l2_regularizer, first_kind=False)
    # pylint: enable=protected-access
    grad_b = linalg_ops.cholesky_solve(chol, math_ops.matmul(a, grad))
    # Temporary tmp = (A * A^T + lambda * I)^{-1} * B.
    tmp = linalg_ops.cholesky_solve(chol, b)
    a1 = math_ops.matmul(tmp, a, adjoint_a=True)
    a1 = -math_ops.matmul(grad_b, a1)
    a2 = grad - math_ops.matmul(a, grad_b, adjoint_a=True)
    a2 = math_ops.matmul(tmp, a2, adjoint_b=True)
    grad_a = a1 + a2
    return (grad_a, grad_b, None)

  fast = op.get_attr("fast")
  if fast is False:
    raise ValueError("Gradient not defined for fast=False")
  matrix_shape = op.inputs[0].get_shape()[-2:]
  if matrix_shape.is_fully_defined():
    if matrix_shape[-2] >= matrix_shape[-1]:
      return _Overdetermined(op, grad)
    else:
      return _Underdetermined(op, grad)
  else:
    # We have to defer determining the shape to runtime and use
    # conditional execution of the appropriate graph.
    matrix_shape = array_ops.shape(op.inputs[0])[-2:]
    return control_flow_ops.cond(matrix_shape[-2] >= matrix_shape[-1],
                                 lambda: _Overdetermined(op, grad),
                                 lambda: _Underdetermined(op, grad))


@ops.RegisterGradient("MatrixTriangularSolve")
def _MatrixTriangularSolveGrad(op, grad):
  """Gradient for MatrixTriangularSolve."""
  a = op.inputs[0]
  adjoint_a = op.get_attr("adjoint")
  lower_a = op.get_attr("lower")
  c = op.outputs[0]
  grad_b = linalg_ops.matrix_triangular_solve(
      a, grad, lower=lower_a, adjoint=not adjoint_a)
  if adjoint_a:
    grad_a = -math_ops.matmul(c, grad_b, adjoint_b=True)
  else:
    grad_a = -math_ops.matmul(grad_b, c, adjoint_b=True)
  if lower_a:
    grad_a = array_ops.matrix_band_part(grad_a, -1, 0)
  else:
    grad_a = array_ops.matrix_band_part(grad_a, 0, -1)
  return (grad_a, grad_b)


@ops.RegisterGradient("SelfAdjointEigV2")
def _SelfAdjointEigV2Grad(op, grad_e, grad_v):
  """Gradient for SelfAdjointEigV2."""
  e = op.outputs[0]
  compute_v = op.get_attr("compute_v")
  # a = op.inputs[0], which satisfies
  # a[...,:,:] * v[...,:,i] = e[...,i] * v[...,i]
  with ops.control_dependencies([grad_e, grad_v]):
    if compute_v:
      v = op.outputs[1]
      # Construct the matrix f(i,j) = (i != j ? 1 / (e_i - e_j) : 0).
      # Notice that because of the term involving f, the gradient becomes
      # infinite (or NaN in practice) when eigenvalues are not unique.
      # Mathematically this should not be surprising, since for (k-fold)
      # degenerate eigenvalues, the corresponding eigenvectors are only defined
      # up to arbitrary rotation in a (k-dimensional) subspace.
      f = array_ops.matrix_set_diag(
          math_ops.reciprocal(
              array_ops.expand_dims(e, -2) - array_ops.expand_dims(e, -1)),
          array_ops.zeros_like(e))
      grad_a = math_ops.matmul(
          v,
          math_ops.matmul(
              array_ops.matrix_diag(grad_e) +
              f * math_ops.matmul(v, grad_v, adjoint_a=True),
              v,
              adjoint_b=True))
    else:
      _, v = linalg_ops.self_adjoint_eig(op.inputs[0])
      grad_a = math_ops.matmul(v,
                               math_ops.matmul(
                                   array_ops.matrix_diag(grad_e),
                                   v,
                                   adjoint_b=True))
    # The forward op only depends on the lower triangular part of a, so here we
    # symmetrize and take the lower triangle
    grad_a = array_ops.matrix_band_part(grad_a + _linalg.adjoint(grad_a), -1, 0)
    grad_a = array_ops.matrix_set_diag(grad_a,
                                       0.5 * array_ops.matrix_diag_part(grad_a))
    return grad_a


@ops.RegisterGradient("Svd")
def _SvdGrad(op, grad_s, grad_u, grad_v):
  """Gradient for the singular value decomposition."""

  # The derivation for the compute_uv=False case, and most of
  # the derivation for the full_matrices=True case, are in
  # Giles' paper (see reference at top of file).  A derivation for
  # the full_matrices=False case is available at
  # https://j-towns.github.io/papers/svd-derivative.pdf
  a = op.inputs[0]
  a_shape = a.get_shape().with_rank_at_least(2)
  grad_s_mat = array_ops.matrix_diag(grad_s)

  if not op.get_attr("compute_uv"):
    s, u, v = linalg_ops.svd(a, compute_uv=True)
    grad_a = math_ops.matmul(u, math_ops.matmul(grad_s_mat, v, adjoint_b=True))
    grad_a.set_shape(a_shape)
    return grad_a

  full_matrices = op.get_attr("full_matrices")

  # TODO(rmlarsen): Make this work with complex types.
  if a.dtype.is_complex:
    raise NotImplementedError(
        "SVD gradient is not implemented for complex types and "
        "compute_uv=True.")
  grad_u_shape = grad_u.get_shape().with_rank_at_least(2)
  grad_v_shape = grad_v.get_shape().with_rank_at_least(2)
  m = a_shape.dims[-2].merge_with(grad_u_shape[-2])
  n = a_shape.dims[-1].merge_with(grad_v_shape[-2])
  batch_shape = a_shape[:-2].merge_with(grad_u_shape[:-2]).merge_with(
      grad_v_shape[:-2])
  a_shape = batch_shape.concatenate([m, n])

  m = a_shape.dims[-2].value
  n = a_shape.dims[-1].value
  # TODO(rmlarsen): Make this work with placeholders.
  if m is None or n is None:
    raise NotImplementedError(
        "SVD gradient has not been implemented for input with unknown "
        "inner matrix shape.")

  s = op.outputs[0]
  u = op.outputs[1]
  v = op.outputs[2]

  use_adjoint = False
  if m > n:
    # Compute the gradient for A^H = V * S^T * U^H, and (implicitly) take the
    # Hermitian transpose of the gradient at the end.
    use_adjoint = True
    m, n = n, m
    u, v = v, u
    grad_u, grad_v = grad_v, grad_u

  with ops.control_dependencies([grad_s, grad_u, grad_v]):
    if full_matrices and abs(m - n) > 1:
      raise NotImplementedError(
          "svd gradient is not implemented for abs(m - n) > 1 "
          "when full_matrices is True")
    s_mat = array_ops.matrix_diag(s)
    s2 = math_ops.square(s)

    # NOTICE: Because of the term involving f, the gradient becomes
    # infinite (or NaN in practice) when singular values are not unique.
    # Mathematically this should not be surprising, since for (k-fold)
    # degenerate singular values, the corresponding singular vectors are
    # only defined up a (k-dimensional) subspace. In practice, this can
    # lead to numerical instability when singular values are close but not
    # exactly equal.
    # Also, even with distinct singular values, the diagonal of f can have Inf
    # values before setting to zero, which hurt when differentiating through
    # this op. To avoid that, we add eye to the matrix before taking
    # the reciprocal.
    s_shape = array_ops.shape(s)
    eye = _linalg.eye(s_shape[-1], batch_shape=s_shape[:-1], dtype=s.dtype)
    f = array_ops.matrix_set_diag(
        math_ops.reciprocal(
            array_ops.expand_dims(s2, -2) - array_ops.expand_dims(s2, -1) +
            eye), array_ops.zeros_like(s))
    s_inv_mat = array_ops.matrix_diag(math_ops.reciprocal(s))

    v1 = v[..., :, :m]
    grad_v1 = grad_v[..., :, :m]

    u_gu = math_ops.matmul(u, grad_u, adjoint_a=True)
    v_gv = math_ops.matmul(v1, grad_v1, adjoint_a=True)

    f_u = f * u_gu
    f_v = f * v_gv

    term1_nouv = (
        grad_s_mat + math_ops.matmul(f_u + _linalg.adjoint(f_u), s_mat) +
        math_ops.matmul(s_mat, f_v + _linalg.adjoint(f_v)))

    term1 = math_ops.matmul(u, math_ops.matmul(term1_nouv, v1, adjoint_b=True))

    if m == n:
      grad_a_before_transpose = term1
    else:
      gv1t = array_ops.matrix_transpose(grad_v1)
      gv1t_v1 = math_ops.matmul(gv1t, v1)
      term2_nous = gv1t - math_ops.matmul(gv1t_v1, v1, adjoint_b=True)

      if full_matrices:
        v2 = v[..., :, m:n]
        grad_v2 = grad_v[..., :, m:n]

        v1t_gv2 = math_ops.matmul(v1, grad_v2, adjoint_a=True)
        term2_nous -= math_ops.matmul(v1t_gv2, v2, adjoint_b=True)

      u_s_inv = math_ops.matmul(u, s_inv_mat)
      term2 = math_ops.matmul(u_s_inv, term2_nous)

      grad_a_before_transpose = term1 + term2

    if use_adjoint:
      grad_a = array_ops.matrix_transpose(grad_a_before_transpose)
    else:
      grad_a = grad_a_before_transpose

    grad_a.set_shape(a_shape)
    return grad_a

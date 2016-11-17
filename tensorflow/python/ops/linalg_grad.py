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

Useful reference for derivative formulas is
An extended collection of matrix derivative results for forward and reverse
mode algorithmic differentiation by Mike Giles:
http://eprints.maths.ox.ac.uk/1079/1/NA-08-01.pdf

A detailed derivation of formulas for backpropagating through spectral layers
(SVD and Eig) by Ionescu, Vantzos & Sminchisescu:
https://arxiv.org/pdf/1509.07838v4.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


@ops.RegisterGradient("MatrixInverse")
def _MatrixInverseGrad(op, grad):
  """Gradient for MatrixInverse."""
  ainv = op.outputs[0]
  return -math_ops.batch_matmul(
      ainv, math_ops.batch_matmul(
          grad, ainv, adj_y=True), adj_x=True)


@ops.RegisterGradient("MatrixDeterminant")
def _MatrixDeterminantGrad(op, grad):
  """Gradient for MatrixDeterminant."""
  a = op.inputs[0]
  c = op.outputs[0]
  a_adj_inv = linalg_ops.matrix_inverse(a, adjoint=True)
  multipliers = array_ops.reshape(
      grad * c, array_ops.concat(0, [array_ops.shape(c), [1, 1]]))
  return multipliers * a_adj_inv


@ops.RegisterGradient("Cholesky")
def _CholeskyGrad(op, grad):
  """Gradient for Cholesky."""
  return linalg_ops.cholesky_grad(op.outputs[0], grad)


@ops.RegisterGradient("MatrixSolve")
def _MatrixSolveGrad(op, grad):
  """Gradient for MatrixSolve."""
  a = op.inputs[0]
  adjoint_a = op.get_attr("adjoint")
  c = op.outputs[0]
  grad_b = linalg_ops.matrix_solve(a, grad, adjoint=not adjoint_a)
  if adjoint_a:
    grad_a = -math_ops.batch_matmul(c, grad_b, adj_y=True)
  else:
    grad_a = -math_ops.batch_matmul(grad_b, c, adj_y=True)
  return (grad_a, grad_b)


@ops.RegisterGradient("MatrixSolveLs")
def _MatrixSolveLsGrad(op, grad):
  """Gradients for MatrixSolveLs."""

  # TODO(rmlarsen): The implementation could be more efficient:
  #   a) Output the Cholesky factorization from forward op instead of
  #      recomputing it here.
  #   b) Implement a symmetric rank-k update op instead of computing
  #      x*z + transpose(x*z). This pattern occurs other places in TensorFlow.

  def _overdetermined(op, grad):
    """Gradients for the overdetermined case of MatrixSolveLs.

    This is the backprop for the solution to the normal equations of the first
    kind:
       X = F(A, B) = (A^T * A + lambda * I)^{-1} * A^T * B
    which solve the least squares problem
       min ||A * X - B||_F^2 + lambda ||X||_F^2.
    """
    a = op.inputs[0]
    b = op.inputs[1]
    l2_regularizer = math_ops.cast(op.inputs[2], a.dtype.base_dtype)
    x = op.outputs[0]
    a_shape = array_ops.shape(a)
    batch_shape = a_shape[:-2]
    n = a_shape[-1]

    identity = linalg_ops.eye(n, batch_shape=batch_shape, dtype=a.dtype)
    gramian = math_ops.batch_matmul(
        a, a, adj_x=True) + l2_regularizer * identity
    chol = linalg_ops.cholesky(gramian)
    # Temporary z = (A^T * A + lambda * I)^{-1} * grad.
    z = linalg_ops.cholesky_solve(chol, grad)
    xzt = math_ops.batch_matmul(x, z, adj_y=True)
    zx_sym = xzt + array_ops.matrix_transpose(xzt)
    grad_a = -math_ops.batch_matmul(a, zx_sym) + math_ops.batch_matmul(
        b, z, adj_y=True)
    grad_b = math_ops.batch_matmul(a, z)
    return (grad_a, grad_b, None)

  def _underdetermined(op, grad):
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
    a_shape = array_ops.shape(a)
    batch_shape = a_shape[:-2]
    m = a_shape[-2]

    identity = linalg_ops.eye(m, batch_shape=batch_shape, dtype=a.dtype)
    gramian = math_ops.batch_matmul(
        a, a, adj_y=True) + l2_regularizer * identity
    chol = linalg_ops.cholesky(gramian)
    grad_b = linalg_ops.cholesky_solve(chol, math_ops.batch_matmul(a, grad))
    # Temporary tmp = (A * A^T + lambda * I)^{-1} * B.
    tmp = linalg_ops.cholesky_solve(chol, b)
    a1 = math_ops.batch_matmul(tmp, a, adj_x=True)
    a1 = -math_ops.batch_matmul(grad_b, a1)
    a2 = grad - math_ops.batch_matmul(a, grad_b, adj_x=True)
    a2 = math_ops.batch_matmul(tmp, a2, adj_y=True)
    grad_a = a1 + a2
    return (grad_a, grad_b, None)

  fast = op.get_attr("fast")
  if fast is False:
    raise ValueError("Gradient not defined for fast=False")
  matrix_shape = op.inputs[0].get_shape()[-2:]
  if matrix_shape.is_fully_defined():
    if matrix_shape[-2] >= matrix_shape[-1]:
      return _overdetermined(op, grad)
    else:
      return _underdetermined(op, grad)
  else:
    # We have to defer determining the shape to runtime and use
    # conditional execution of the appropriate graph.
    matrix_shape = array_ops.shape(op.inputs[0])[-2:]
    return control_flow_ops.cond(matrix_shape[-2] >= matrix_shape[-1],
                                 lambda: _overdetermined(op, grad),
                                 lambda: _underdetermined(op, grad))


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
    grad_a = -math_ops.batch_matmul(c, grad_b, adj_y=True)
  else:
    grad_a = -math_ops.batch_matmul(grad_b, c, adj_y=True)
  if lower_a:
    grad_a = array_ops.matrix_band_part(grad_a, -1, 0)
  else:
    grad_a = array_ops.matrix_band_part(grad_a, 0, -1)
  return (grad_a, grad_b)


@ops.RegisterGradient("SelfAdjointEigV2")
def _SelfAdjointEigV2Grad(op, grad_e, grad_v):
  """Gradient for SelfAdjointEigV2."""
  e = op.outputs[0]
  v = op.outputs[1]
  # a = op.inputs[0], which satisfies
  # a[...,:,:] * v[...,:,i] = e[...,i] * v[...,i]
  with ops.control_dependencies([grad_e.op, grad_v.op]):
    if grad_v is not None:
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
      grad_a = math_ops.batch_matmul(
          v,
          math_ops.batch_matmul(
              array_ops.matrix_diag(grad_e) + f * math_ops.batch_matmul(
                  v, grad_v, adj_x=True),
              v,
              adj_y=True))
    else:
      grad_a = math_ops.batch_matmul(
          v,
          math_ops.batch_matmul(
              array_ops.matrix_diag(grad_e), v, adj_y=True))
    # The forward op only depends on the lower triangular part of a, so here we
    # symmetrize and take the lower triangle
    grad_a = array_ops.matrix_band_part(
        grad_a + array_ops.matrix_transpose(grad_a), -1, 0)
    grad_a = array_ops.matrix_set_diag(grad_a,
                                       0.5 * array_ops.matrix_diag_part(grad_a))
    return grad_a

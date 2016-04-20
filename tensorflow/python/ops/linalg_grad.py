# Copyright 2015 Google Inc. All Rights Reserved.
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
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


@ops.RegisterGradient("MatrixInverse")
def _MatrixInverseGrad(op, grad):
  """Gradient for MatrixInverse."""
  ainv = op.outputs[0]
  return -math_ops.matmul(ainv,
                          math_ops.matmul(grad,
                                          ainv,
                                          transpose_b=True),
                          transpose_a=True)


@ops.RegisterGradient("BatchMatrixInverse")
def _BatchMatrixInverseGrad(op, grad):
  """Gradient for BatchMatrixInverse."""
  ainv = op.outputs[0]
  return -math_ops.batch_matmul(ainv,
                                math_ops.batch_matmul(grad,
                                                      ainv,
                                                      adj_y=True),
                                adj_x=True)


@ops.RegisterGradient("MatrixDeterminant")
def _MatrixDeterminantGrad(op, grad):
  """Gradient for MatrixDeterminant."""
  a = op.inputs[0]
  c = op.outputs[0]
  a_adj_inv = linalg_ops.matrix_inverse(a, adjoint=True)
  return grad * c * a_adj_inv


@ops.RegisterGradient("BatchMatrixDeterminant")
def _BatchMatrixDeterminantGrad(op, grad):
  """Gradient for BatchMatrixDeterminant."""
  a = op.inputs[0]
  c = op.outputs[0]
  a_adj_inv = linalg_ops.batch_matrix_inverse(a, adjoint=True)
  multipliers = array_ops.reshape(
      grad * c, c.get_shape().concatenate(tensor_shape.TensorShape([1, 1])))
  return multipliers * a_adj_inv


@ops.RegisterGradient("Cholesky")
def _cholesky_grad(op, grad):
  """Gradient for Cholesky."""
  return linalg_ops.cholesky_grad(op.outputs[0], grad)


@ops.RegisterGradient("MatrixSolve")
def _MatrixSolveGrad(op, grad):
  """Gradients for MatrixSolve."""
  a = op.inputs[0]
  adjoint_a = op.get_attr("adjoint")
  c = op.outputs[0]
  grad_b = linalg_ops.matrix_solve(a, grad, adjoint=not adjoint_a)
  if adjoint_a:
    grad_a = -math_ops.matmul(c, grad_b, transpose_b=True)
  else:
    grad_a = -math_ops.matmul(grad_b, c, transpose_b=True)
  return (grad_a, grad_b)


@ops.RegisterGradient("BatchMatrixSolve")
def _BatchMatrixSolveGrad(op, grad):
  """Gradient for BatchMatrixSolve."""
  a = op.inputs[0]
  adjoint_a = op.get_attr("adjoint")
  c = op.outputs[0]
  grad_b = linalg_ops.batch_matrix_solve(a, grad, adjoint=not adjoint_a)
  if adjoint_a:
    grad_a = -math_ops.batch_matmul(c, grad_b, adj_y=True)
  else:
    grad_a = -math_ops.batch_matmul(grad_b, c, adj_y=True)
  return (grad_a, grad_b)


@ops.RegisterGradient("MatrixTriangularSolve")
def _MatrixTriangularSolveGrad(op, grad):
  """Gradients for MatrixTriangularSolve."""
  a = op.inputs[0]
  adjoint_a = op.get_attr("adjoint")
  lower_a = op.get_attr("lower")
  c = op.outputs[0]
  grad_b = linalg_ops.matrix_triangular_solve(a,
                                              grad,
                                              lower=lower_a,
                                              adjoint=not adjoint_a)
  if adjoint_a:
    grad_a = -math_ops.matmul(c, grad_b, transpose_b=True)
  else:
    grad_a = -math_ops.matmul(grad_b, c, transpose_b=True)
  if lower_a:
    grad_a = array_ops.batch_matrix_band_part(grad_a, -1, 0)
  else:
    grad_a = array_ops.batch_matrix_band_part(grad_a, 0, -1)
  return (grad_a, grad_b)


@ops.RegisterGradient("BatchMatrixTriangularSolve")
def _BatchMatrixTriangularSolveGrad(op, grad):
  """Gradient for BatchMatrixTriangularSolve."""
  a = op.inputs[0]
  adjoint_a = op.get_attr("adjoint")
  lower_a = op.get_attr("lower")
  c = op.outputs[0]
  grad_b = linalg_ops.batch_matrix_triangular_solve(a,
                                                    grad,
                                                    lower=lower_a,
                                                    adjoint=not adjoint_a)
  if adjoint_a:
    grad_a = -math_ops.batch_matmul(c, grad_b, adj_y=True)
  else:
    grad_a = -math_ops.batch_matmul(grad_b, c, adj_y=True)
  if lower_a:
    grad_a = array_ops.batch_matrix_band_part(grad_a, -1, 0)
  else:
    grad_a = array_ops.batch_matrix_band_part(grad_a, 0, -1)
  return (grad_a, grad_b)

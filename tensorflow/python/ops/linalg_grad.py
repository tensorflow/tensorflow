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
  """Gradient for MatrixDeterminant.

  Returns:
    gradient
  Args:
    op: op
    grad: grad
  """
  a = op.inputs[0]
  c = op.outputs[0]
  ainv = linalg_ops.matrix_inverse(a)
  return grad * c * array_ops.transpose(ainv)

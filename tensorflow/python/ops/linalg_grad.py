"""Gradients for operators defined in linalg_ops.py."""
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
  return -math_ops.matmul(
      ainv,
      math_ops.matmul(grad, ainv, transpose_b=True),
      transpose_a=True)

@ops.RegisterGradient("BatchMatrixInverse")
def _BatchMatrixInverseGrad(op, grad):
  """Gradient for BatchMatrixInverse."""
  ainv = op.outputs[0]
  return -math_ops.batch_matmul(
      ainv,
      math_ops.batch_matmul(grad, ainv, adj_y=True),
      adj_x=True)

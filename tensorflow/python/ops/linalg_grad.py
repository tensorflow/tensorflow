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

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

"""All user ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform
from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *


def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()

@ops.RegisterGradient("TriangularSolve")
def _solve_grad(op, grad):
    """The gradients for `solve`.
    Args:
    op: The `solve` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `solve` op.
    
    Returns:
    Gradients with respect to the input of `solve`.
    """
    if op.get_attr('Case')=='lower':
        outputGrad = triangular_solve( array_ops.transpose( op.inputs[0] ), grad, 'upper' )
    else:
        outputGrad = triangular_solve( array_ops.transpose( op.inputs[0] ), grad, 'lower' )        
    return ( math_ops.matmul( math_ops.neg( outputGrad ), array_ops.transpose(op.outputs[0]) ), outputGrad )

@ops.RegisterShape("TriangularSolve")
def _solve_shape(op):
  """Shape function for the Solve op.
  produces an output
  with the same shape as its second input.
  """
  return [op.inputs[1].get_shape()]

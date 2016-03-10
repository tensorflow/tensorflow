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

from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python import ops


def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()

@ops.RegisterGradient("Cholesky")
def _cholesky_grad(op, grad):
  return ( cholesky_grad( op.outputs[0] , grad ) )

@ops.RegisterShape("CholeskyGrad")
def _cholesky_grad_shape(op):
  return [op.inputs[0].get_shape()]  
  
@ops.RegisterGradient("Triangle")
def _solve_grad(op, grad):
  if op.get_attr('Case')=='lower':
    outputGrad = triangle(grad,'lower')
  else:
    outputGrad = triangle(grad,'upper')       
  return ( outputGrad )
    
@ops.RegisterShape("Triangle")
def _triangle_shape(op):
    return [ op.inputs[0].get_shape() ]

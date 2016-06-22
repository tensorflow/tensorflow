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
from tensorflow.python.framework import ops


def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()

def trial(a):
  return gen_user_ops._trial(a)

@ops.RegisterShape("Trial")
def _trial_shape(op):
  return [op.inputs[0].get_shape()]

def lookahead(x1, x2):
  return gen_user_ops._lookahead(x1, x2)

def lookaheadgpu(x1, x2):
  return gen_user_ops._lookaheadgpu(x1, x2)

def lookaheadgrad(x1, x2, x3):
  return gen_user_ops._lookaheadgrad(x1, x2, x3)

def lookaheadgradgpu(x1, x2, x3):
  return gen_user_ops._lookaheadgradgpu(x1, x2, x3)

@ops.RegisterShape("Lookahead")
def _lookahead(op):
  inputs_shape = op.inputs[0].get_shape().with_rank(3)
  return [inputs_shape]

@ops.RegisterGradient("Lookahead")
def _lookahead_grad(op, grad):
  """
  Args:
    op: the lookahead op.
    grad: the output grad
  Returns:
    the input grad and the filter grad
  """
  return lookaheadgrad(
              op.inputs[0],op.inputs[1],grad)

@ops.RegisterGradient("Lookaheadgpu")
def _lookaheadgpu_grad(op, grad):
  """
  Args:
    op: the lookahead op.
    grad: the output grad
  Returns:
    the input grad and the filter grad
  """
  return lookaheadgradgpu(
              op.inputs[0],op.inputs[1],grad)

@ops.RegisterShape("Lookaheadgpu")
def _lookaheadgpu(op):
  inputs_shape = op.inputs[0].get_shape().with_rank(3)
  return [inputs_shape]

@ops.RegisterShape("Lookaheadgrad")
def _lookaheadgrad(op):
  inputs_shape1 = op.inputs[0].get_shape().with_rank(3)
  inputs_shape2 = op.inputs[1].get_shape().with_rank(2)
  return [inputs_shape1, inputs_shape2]

@ops.RegisterShape("Lookaheadgradgpu")
def _lookaheadgradgpu(op):
  inputs_shape1 = op.inputs[0].get_shape().with_rank(3)
  inputs_shape2 = op.inputs[1].get_shape().with_rank(2)
  return [inputs_shape1, inputs_shape2]

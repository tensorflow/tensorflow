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
"""Python wrapper for the Repeat Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

_repeat_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_repeat_op.so"))
repeat = _repeat_so.repeat

# TODO: use tf.reduce_slice_sum #9063
@ops.RegisterGradient("Repeat")
def _RepeatGrad(op, grad):
  """Sum reduces grads along the repeated dimension"""
  input_raw_shape = op.inputs[0].get_shape().as_list()
  input_raw_shape_tensor = array_ops.shape(op.inputs[0])
  if len(input_raw_shape) == 0:
    input_shape = [1]
    input_shape_tensor = constant_op.constant([1])
  else:
    input_shape = input_raw_shape
    input_shape_tensor = input_raw_shape_tensor
  repeats = op.inputs[1]
  axis = op.get_attr("axis")
  if axis < 0:
    axis += len(input_shape)
  
  # Collapse the outer and inner dimensions
  outer_dim = 1
  for i in xrange(axis):
    outer_dim *= input_shape_tensor[i]
  inner_dim = 1
  for i in xrange(axis+1, len(input_shape)):
    inner_dim *= input_shape_tensor[i]
  out_grad = array_ops.reshape(grad, [outer_dim, -1, inner_dim])
  
  
  # Handle the scalar case of `repeats`
  if repeats.get_shape().as_list() == []:
    repeats = array_ops.reshape(repeats,[-1])
  if repeats.get_shape().num_elements() == 1:
    repeats = array_ops.tile(repeats, [input_shape[axis]])
  
  start = constant_op.constant([0])
  input_grads = []
  for i in xrange(0, repeats.get_shape().num_elements()):
    end = start + repeats[i]
    input_grads.append(
        math_ops.reduce_sum(out_grad[:, start[0]:end[0], :], 1))
    start = end
  in_grad = array_ops.reshape(array_ops.stack(input_grads, 1), input_raw_shape_tensor)
  return [in_grad, None]

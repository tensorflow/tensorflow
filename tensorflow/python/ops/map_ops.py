# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Use zero_out ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_map_ops
from tensorflow.python.ops.gen_map_ops import *

#zero_out_ops = load_library.load_op_library(
#    resource_loader.get_path_to_datafile('_zero_out_ops.so'))
#zero_out = zero_out_ops.zero_out 

def empty_tensor_map(element_shape,
                      element_dtype,
                      max_num_elements=None,
                      name=None):
  if max_num_elements is None:
    max_num_elements = -1

  return gen_map_ops.empty_tensor_map(element_shape, 
      element_dtype,
      max_num_elements,
      name)

def zero_out(to_zero):
    print("Hello World - Python Op")
    return gen_map_ops.zero_out(to_zero)

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
    """The gradients for `zero_out`.

    Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

    Returns:
    Gradients with respect to the input of `zero_out`.
    """
    to_zero = op.inputs[0]
    shape = array_ops.shape(to_zero)
    index = array_ops.zeros_like(shape)
    first_grad = array_ops.reshape(grad, [-1])[0]
    to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
    return [to_zero_grad]  # List of one Tensor, since we have one input

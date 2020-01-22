# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Gradients for operators defined in random_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops


def add_leading_unit_dimensions(x, num_dimensions):
  new_shape = array_ops.concat(
      [array_ops.ones([num_dimensions], dtype=dtypes.int32),
       array_ops.shape(x)], axis=0)
  return array_ops.reshape(x, new_shape)


@ops.RegisterGradient("RandomGamma")
def _RandomGammaGrad(op, grad):  # pylint: disable=invalid-name
  """Returns the gradient of a Gamma sample w.r.t. alpha.

  The gradient is computed using implicit differentiation, see
  "Implicit Reparameterization Gradients" (https://arxiv.org/abs/1805.08498).

  Args:
    op: A `RandomGamma` operation. We assume that the inputs to the operation
      are `shape` and `alpha` tensors, and the output is the `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
      `op.outputs[0]`.

  Returns:
    A `Tensor` with derivatives `dloss / dalpha`
  """
  shape = op.inputs[0]
  alpha = op.inputs[1]
  sample = op.outputs[0]

  with ops.control_dependencies([grad]):
    # Make the parameters alpha broadcastable with samples by appending
    # unit dimensions.
    num_sample_dimensions = array_ops.shape(shape)[0]
    alpha_broadcastable = add_leading_unit_dimensions(
        alpha, num_sample_dimensions)
    partial_a = gen_random_ops.random_gamma_grad(alpha_broadcastable, sample)

    # The first input is shape; the second input is alpha.
    return (None, math_ops.reduce_sum(
        grad * partial_a, axis=math_ops.range(num_sample_dimensions)))

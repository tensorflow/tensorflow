# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Gradients accumulation for making use of bigger batch size on
   limited memory GPUs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops


def grad_accumulation(grads_and_vars,
                      iter_size=None,
                      average=False):
  """Construct subgraph for gradient accumulation.

  The accumulated gradient is a neural network framework feature. It is a
  workaround to enable big batches on limited memory GPUs. Instead of
  back-propagating for every batch feed-forward, gradients across multiple
  batches are accumulated. After multiple feed forwards, the accumulated
  gradient is back-propagated through the network layer. This gives the
  illusion of using big batches on limited memory GPUs.

  Args:
    grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
    iter_size: Number of local iteration before network weights are updated.
    average: If True, compute the average over all.

  Returns:
    (accum_grad_ops, reset_grad_ops, accumulated_grads_and_vars)
  """
  accum_grads = []
  accum_vars = []
  accum_grad_ops = []
  reset_grad_ops = []

  for grad, var in grads_and_vars:
    accum_vars.append(var)
    with ops.device(var.device):
      if grad is None:
        accum_grads.append(None)
        continue
      assert isinstance(grad, (ops.Tensor, ops.IndexedSlices)),\
          ("Gradient ", grad, " is neither a tensor nor IndexedSlices.")
      accum_grad = resource_variable_ops.ResourceVariable(
          lambda: array_ops.zeros(var.get_shape(), grad.dtype),
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="grad_accumulator")
      reset_grad_op = state_ops.assign(
          accum_grad,
          array_ops.zeros(var.get_shape(), grad.dtype),
          name="reset_grad_accumulator")
      if isinstance(grad, ops.Tensor):
        accum_grad_op = accum_grad.assign_add(grad, name="grad_accumulation")
      else:
        accum_grad_op = state_ops.scatter_update(
            accum_grad, grad.indices, grad.values, name="sparse_grad_accum")

      accum_grads.append(accum_grad)
      accum_grad_ops.append(accum_grad_op)
      reset_grad_ops.append(reset_grad_op)
  if average:
    accum_grads_mean = [math_ops.multiply(v, 1.0 / iter_size)
                        if v is not None else None
                        for v in accum_grads]
    accumulated_grads_and_vars = zip(accum_grads_mean, accum_vars)
  else:
    accumulated_grads_and_vars = zip(accum_grads, accum_vars)

  return accum_grad_ops, reset_grad_ops, accumulated_grads_and_vars

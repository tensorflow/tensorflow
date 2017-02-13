# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Gradients for operators defined in data_flow_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops


@ops.RegisterGradient("DynamicPartition")
def _DynamicPartitionGrads(op, *grads):
  """Gradients for DynamicPartition."""
  data = op.inputs[0]
  indices = op.inputs[1]
  num_partitions = op.get_attr("num_partitions")

  prefix_shape = array_ops.shape(indices)
  original_indices = array_ops.reshape(
      math_ops.range(math_ops.reduce_prod(prefix_shape)), prefix_shape)
  partitioned_indices = data_flow_ops.dynamic_partition(
      original_indices, indices, num_partitions)
  reconstructed = data_flow_ops.dynamic_stitch(partitioned_indices, grads)
  reconstructed = array_ops.reshape(reconstructed, array_ops.shape(data))
  return [reconstructed, None]


@ops.RegisterGradient("DynamicStitch")
def _DynamicStitchGrads(op, grad):
  """Gradients for DynamicStitch."""

  num_values = len(op.inputs) // 2
  indices_grad = [None] * num_values

  def AsInt32(x):
    return (x if op.inputs[0].dtype == dtypes.int32 else
            math_ops.cast(x, dtypes.int32))
  idxs = [AsInt32(array_ops.reshape(op.inputs[i], (-1,)))
          for i in xrange(num_values)]
  if isinstance(grad, ops.IndexedSlices):
    output_shape = array_ops.shape(op.outputs[0])
    output_rows = output_shape[0]
    grad = math_ops.unsorted_segment_sum(grad.values, grad.indices, output_rows)

  values_grad = []
  zeros = array_ops.zeros_like(grad)
  idx_zeros = [zeros[:array_ops.shape(x)[0]] for x in idxs]
  grad_range = math_ops.range(array_ops.shape(grad)[0])
  for i in range(num_values):
    if i == num_values - 1:
      v_grad = grad
    else:
      v_grad = data_flow_ops.dynamic_stitch(
        [grad_range] + idxs[i+1:], [grad] + idx_zeros[i+1:])
    v_grad = array_ops.gather(v_grad, AsInt32(op.inputs[i]))
    values_grad += [v_grad]

  return indices_grad + values_grad


ops.NotDifferentiable("Queue")
ops.NotDifferentiable("QueueEnqueue")
ops.NotDifferentiable("QueueEnqueueMany")
ops.NotDifferentiable("QueueDequeue")
ops.NotDifferentiable("QueueDequeueMany")
ops.NotDifferentiable("QueueDequeueUpTo")
ops.NotDifferentiable("QueueClose")
ops.NotDifferentiable("QueueSize")

ops.NotDifferentiable("Stack")
ops.NotDifferentiable("StackPush")
ops.NotDifferentiable("StackPop")
ops.NotDifferentiable("StackClose")

ops.NotDifferentiable("GetSessionHandle")
ops.NotDifferentiable("GetSessionTensor")
ops.NotDifferentiable("DeleteSessionTensor")

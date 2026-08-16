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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
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
  reconstructed = data_flow_ops.parallel_dynamic_stitch(partitioned_indices,
                                                        grads)
  reconstructed = array_ops.reshape(reconstructed, array_ops.shape(data))
  return [reconstructed, None]


@ops.RegisterGradient("DynamicStitch")
@ops.RegisterGradient("ParallelDynamicStitch")
def _DynamicStitchGrads(op, grad):
  """Gradients for DynamicStitch and ParallelDynamicStitch."""

  num_values = len(op.inputs) // 2
  indices_grad = [None] * num_values

  def AsInt32(x):
    return (x if op.inputs[0].dtype == dtypes.int32 else
            math_ops.cast(x, dtypes.int32))

  inputs = [AsInt32(op.inputs[i]) for i in range(num_values)]
  if isinstance(grad, indexed_slices.IndexedSlices):
    output_shape = array_ops.shape(op.outputs[0])
    output_rows = output_shape[0]
    grad = math_ops.unsorted_segment_sum(grad.values, grad.indices, output_rows)

  ids = []
  current_size = array_ops.zeros([], dtype=dtypes.int32)
  for inp in inputs:
    num_elements = math_ops.cast(array_ops.size(inp), current_size.dtype)
    flat_id = math_ops.range(current_size, current_size + num_elements)
    ids.append(array_ops.reshape(flat_id, array_ops.shape(inp)))
    current_size += num_elements

  stitch_op = (
      data_flow_ops.parallel_dynamic_stitch
      if op.type == "ParallelDynamicStitch"
      else data_flow_ops.dynamic_stitch
  )

  stitched_ids = stitch_op(inputs, ids)

  values_grad = []
  num_inner_dims = array_ops.rank(grad) - 1
  for inp, single_id in zip(inputs, ids):
    value_grad = array_ops.gather(grad, inp)
    winning_ids = array_ops.gather(stitched_ids, inp)
    is_winner = math_ops.equal(winning_ids, single_id)
    mask = math_ops.cast(is_winner, value_grad.dtype)
    winner_shape = array_ops.shape(is_winner)
    mask_shape = array_ops.concat(
        [
            winner_shape,
            array_ops.ones([num_inner_dims], dtype=winner_shape.dtype),
        ],
        axis=0,
    )
    values_grad.append(value_grad * array_ops.reshape(mask, mask_shape))

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
ops.NotDifferentiable("GetSessionHandleV2")
ops.NotDifferentiable("GetSessionTensor")
ops.NotDifferentiable("DeleteSessionTensor")

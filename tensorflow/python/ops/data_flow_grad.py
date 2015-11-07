"""Gradients for operators defined in data_flow_ops.py."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops


@ops.RegisterGradient("DynamicStitch")
def _DynamicStitchGrads(op, grad):
  """Gradients for DynamicStitch."""

  num_values = len(op.inputs) / 2
  indices_grad = [None] * num_values

  def AsInt32(x):
    return (x if op.inputs[0].dtype == types.int32 else
            math_ops.cast(x, types.int32))
  inputs = [AsInt32(op.inputs[i]) for i in range(num_values)]
  if isinstance(grad, ops.IndexedSlices):
    output_shape = array_ops.shape(op.outputs[0])
    output_rows = output_shape[0]
    grad = math_ops.unsorted_segment_sum(grad.values, grad.indices, output_rows)
  values_grad = [array_ops.gather(grad, inp) for inp in inputs]
  return indices_grad + values_grad


ops.NoGradient("Queue")
ops.NoGradient("QueueEnqueue")
ops.NoGradient("QueueEnqueueMany")
ops.NoGradient("QueueDequeue")
ops.NoGradient("QueueDequeueMany")
ops.NoGradient("QueueClose")
ops.NoGradient("QueueSize")

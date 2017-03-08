from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.platform import resource_loader
# from tensorflow.contrib.naturali.python.ops import lookahead_grad_ops

_warpctc_ops_so = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_warpctc_ops.so"))
assert _warpctc_ops_so, "Could not load _warpctc_ops.so."


# NOTE(ebrevdo): We redefine CTCLoss from gen_ctc_ops to only return
# the first output. The second output is only used for the gradient.
# pylint: disable=protected-access, invalid-name
def warp_ctc_loss(inputs, labels, sequence_length,
                  preprocess_collapse_repeated=False, ctc_merge_repeated=True):
  if not isinstance(labels, ops.SparseTensor):
    raise TypeError("Expected labels to be a SparseTensor")

  tmps1 = tf.slice(inputs, [0, 0, 0], [-1, -1, int(tf.Tensor.get_shape(inputs)[2] - 1)])
  tmps2 = tf.slice(inputs, [0, 0, int(tf.Tensor.get_shape(inputs)[2] - 1)], [-1, -1, 1])
  inputs = tf.concat(2, [tmps2, tmps1])

  value_1 = tf.ones(tf.shape(labels.values),dtype=tf.int32)
  new_values = tf.add(labels.values, value_1)
  loss, _ = _warpctc_ops_so._warp_ctc_loss(
      inputs,
      labels.indices,
      new_values,
      sequence_length,
      preprocess_collapse_repeated=preprocess_collapse_repeated,
      ctc_merge_repeated=ctc_merge_repeated)

  return loss


# pylint: disable=unused-argument
@ops.RegisterGradient("WarpCtcLoss")
def _CTCLossGrad(op, grad_loss, _):
  """The derivative provided by CTC Loss.
  Args:
     op: the CTCLoss op.
     grad_loss: The backprop for cost.
  Returns:
     The CTC Loss gradient.
  """
  # Outputs are: loss, grad
  grad = op.outputs[1]
  # Return gradient for inputs and None for
  # labels_indices, labels_values and sequence_length
  return [_BroadcastMul(grad_loss, grad), None, None, None]


@ops.RegisterShape("WarpCtcLoss")
def _CTCLossShape(op):
  """Shape function for the CTCLoss op."""
  # inputs, label_indices, label_values, sequence_length
  inputs_shape = op.inputs[0].get_shape().with_rank(3)
  sequence_length_shape = op.inputs[3].get_shape().with_rank(1)
  # merge batch_size
  sequence_length_shape[0].merge_with(inputs_shape[1])
  inputs_shape[1].merge_with(sequence_length_shape[0])
  batch_size = inputs_shape[1]
  labels_index_shape = op.inputs[1].get_shape().with_rank(2)
  labels_value_shape = op.inputs[2].get_shape().with_rank(1)
  labels_value_shape[0].merge_with(labels_index_shape[0])
  # loss, gradient
  return [tensor_shape.vector(batch_size), inputs_shape]

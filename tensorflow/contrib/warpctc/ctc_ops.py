# Copyright 2016 Google Inc. All Rights Reserved.
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

# pylint: disable=unused-import
"""CTC (Connectionist Temporal Classification) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import gen_warpctc_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul


# NOTE(ebrevdo): We redefine CTCLoss from gen_ctc_ops to only return
# the first output. The second output is only used for the gradient.
# pylint: disable=protected-access, invalid-name
def warp_ctc_loss(labels, inputs, sequence_length,
                  preprocess_collapse_repeated=False, ctc_merge_repeated=True):
  if not isinstance(labels, sparse_tensor.SparseTensor):
    raise TypeError("Expected labels (first argument) to be a SparseTensor")

  tmps1 = tf.slice(inputs, [0, 0, 0], [-1, -1, int(tf.Tensor.get_shape(inputs)[2] - 1)])
  tmps2 = tf.slice(inputs, [0, 0, int(tf.Tensor.get_shape(inputs)[2] - 1)], [-1, -1, 1])
  inputs = tf.concat([tmps2, tmps1], 2)

  value_1 = tf.ones(tf.shape(labels.values),dtype=tf.int32)
  new_values = tf.add(labels.values, value_1)
  loss, _ = gen_warpctc_ops._warp_ctc_loss(
      inputs,
      labels.indices,
      new_values,
      sequence_length,
      preprocess_collapse_repeated=preprocess_collapse_repeated,
      ctc_merge_repeated=ctc_merge_repeated)

  return loss


# pylint: disable=unused-argument
@ops.RegisterGradient("WarpCTCLoss")
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


@ops.RegisterShape("WarpCTCLoss")
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


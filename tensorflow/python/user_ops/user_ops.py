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

"""All user ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *
from tensorflow.python.framework import ops

import sys
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul



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
  loss, _ = gen_user_ops._warp_ctc_loss(
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


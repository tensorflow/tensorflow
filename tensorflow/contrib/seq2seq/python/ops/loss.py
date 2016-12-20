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

"""Seq2seq loss operations for use in neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest 


__all__ = ["seq2seq_loss",
					 "sequence_loss_by_example",
					 "sequence_loss"]


def seq2seq_loss(*args, **kwargs):
  pass


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: [batch_size, num_steps, num_decoder_symbols]. 
            if softmax_loss_function is not None then here is [batch_size, num_steps, emb_dim], actually is just outputs from dynamic_rnn 
    targets: [batch_size, num_steps]
    weights: [batch_size, num_steps]
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.
  """
  with ops.name_scope(name, "sequence_loss_by_example",
                      [logits, targets, weights]):
    if softmax_loss_function is None:
      #croosents [batch_size, num_steps]
      crossents = nn_ops.sparse_softmax_cross_entropy_with_logits(logits, targets)
    else:
      shape_ = array_ops.shape(logits)
      batch_size = shape_[0]
      emb_dim = shape_[-1]
      #need reshape because unlike sparse_softmax_cross_entropy_with_logits, 
      #tf.nn.sampled_softmax_loss now only accept [batch_size, dim] as logtis input
      logits = array_ops.reshape(logits, [-1, emb_dim])
      targets = array_ops.reshape(targets, [-1, 1])
      #croosents [batch_size * num_steps]
      crossents = softmax_loss_function(logits, targets)
      #croosents [batch_size, num_steps]
      crossents = array_ops.reshape(crossents, [batch_size, -1])

    log_perps = math_ops.reduce_sum(math_ops.mul(crossents, weights), 1)

    if average_across_timesteps:
      total_size = math_ops.reduce_sum(weights, 1)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps

def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: [batch_size, num_steps, num_decoder_symbols]. 
            if softmax_loss_function is not None then here is [batch_size, num_steps, emb_dim], actually is just outputs from dynamic_rnn 
    targets: [batch_size, num_steps]
    weights: [batch_size, num_steps]
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).
  """
  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets)[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost

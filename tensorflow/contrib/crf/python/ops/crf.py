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
"""Module for constructing a linear-chain CRF.

The following snippet is an example of a CRF layer on top of a batched sequence
of unary scores (logits for every word). This example also decodes the most
likely sequence at test time:

log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
    unary_scores, gold_tags, sequence_lengths)
loss = tf.reduce_mean(-log_likelihood)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

tf_unary_scores, tf_sequence_lengths, tf_transition_params, _ = session.run(
    [unary_scores, sequence_lengths, transition_params, train_op])
for tf_unary_scores_, tf_sequence_length_ in zip(tf_unary_scores,
                                                 tf_sequence_lengths):
# Remove padding.
tf_unary_scores_ = tf_unary_scores_[:tf_sequence_length_]

# Compute the highest score and its tag sequence.
viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
    tf_unary_scores_, tf_transition_params)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn

__all__ = [
    "crf_sequence_score", "crf_log_norm", "crf_log_likelihood",
    "crf_unary_score", "crf_binary_score", "viterbi_decode"
]


def crf_log_likelihood(inputs, targets, transition_params,
                       sequence_lengths=None):
  """Compute loglikelihood of crf
  Args:
    inputs: a [batch_size, seq_len, num_tags] tensor of unary potentials
        to use as input to CRF layer.
    targets: a [batch_size, seq_len] matrix of tag indices for which we
        compute the log-likelihood.
    transition_params: a [num_tags + 1, num_tags + 1] transition matrix.
    sequence_lengths: a [batch_size] vector of sequence lengths.
  """
  sequence_scores = crf_sequence_score(inputs, targets, transition_params,
                                       sequence_lengths)
  log_norm = crf_log_norm(inputs, transition_params, sequence_lengths)

  log_likelihood = sequence_scores - log_norm
  return log_likelihood


def crf_sequence_score(inputs, targets, transition_params, sequence_lengths):
  """Compute the unnormalized score for a tag sequence

  Args:
    inputs: a [batch_size, seq_len, num_tags] tensor of unary potentials
        to use as input to CRF layer
    targets: a [batch_size, seq_len] matrix of tag indices for which we compute
        the unnormalized score.
    transition_params: a [num_tags + 1, num_tags + 1] transition matrix.
    sequence_lengths: a [batch_size] vector of sequence lengths.
  """

  unary_scores = crf_unary_score(targets, sequence_lengths, inputs)
  binary_scores = crf_binary_score(targets, sequence_lengths, transition_params)

  sequence_scores = unary_scores + binary_scores
  return sequence_scores


def crf_log_norm(inputs, transition_params, sequence_lengths):
  """Compute the normalization for a CRF.

  Args:
    inputs: a [batch_size, seq_len, num_tags] tensor of unary potentials to
        use as input to CRF layer.
    transition_params: a [num_tags + 1, num_tags + 1] matrix of transition
        matrix.
    sequence_lengths: a [batch_size] vector of sequence lengths.
  """
  shape = array_ops.shape(inputs)
  batch_size, seq_len, num_tags = shape[0], shape[1], shape[2]
  inputs = array_ops.unpack(inputs, axis=1)
  # extract ROI of transition_params, and expand dims, resulting in shape
  # [num_tags, num_tags, 1]
  expand_transition_params = array_ops.expand_dims(transition_params[1:, :-1],
                                                   0)

  # calculate the normalization value starts from the special start symbol
  # and the first observation step, resulting in shape [batch_size, num_tags]
  prev = array_ops.expand_dims(transition_params[0, :-1], 0) + inputs[0]
  prevs = [prev]

  for ins in inputs[1:]:
    prev = array_ops.expand_dims(prev, -1)
    prev = math_ops.reduce_logsumexp(prev + expand_transition_params, 1) + ins
    prevs.append(prev)

  if sequence_lengths is None:
    prev = prevs[-1]
  else:
    alphas = array_ops.pack(prevs, axis=1)
    # retrieve the last alpha according to sequence_lengths
    # prev = tf.gather_nd(alphas,
    #                     tf.pack([tf.range(batch_size), sequence_lengths],
    #                             axis=1))
    # TODO the following should be replaced by tf.gather_nd, when its gradient
    # is implemented.
    offset = array_ops.expand_dims(math_ops.range(batch_size) * seq_len, 1)
    flattened_indices = gen_array_ops.reshape(
        offset + array_ops.expand_dims(sequence_lengths-1, -1), [-1])
    flattened_alphas = gen_array_ops.reshape(alphas, [-1, num_tags])
    prev = gen_array_ops.reshape(
        gen_array_ops.gather(flattened_alphas, flattened_indices),
        [batch_size, num_tags])

  # calculate the normalization value starts from the last observation step
  # to the special end symbol
  last = math_ops.reduce_logsumexp(
      array_ops.expand_dims(transition_params[1:, -1], 0) + prev, 1)
  return last


def crf_unary_score(targets, sequence_lengths, inputs):
  """Computes the unary scores of tag sequences.

  Args:
    targets: a [batch_size, seq_len] matrix of tag indices.
    masks: a [batch_size, seq_len] matrix of masks representing the first
        length positions of each row.
    sequence_lengths: a [batch_size] vector of squence lengths.
    inputs: a [batch_size, seq_len, num_tags] tensor of unary potentials.
  """
  shape = array_ops.shape(inputs)
  batch_size, seq_len, num_tags = shape[0], shape[1], shape[2]

  # the input data is flattened and offset is calculated on the flattened data.
  flat_input = gen_array_ops.reshape(inputs, [-1])
  offset = math_ops.range(batch_size * seq_len) * num_tags + \
          gen_array_ops.reshape(targets, [-1])

  # simply gather the unary scores of the flattened data and the offset.
  unary_scores = gen_array_ops.reshape(gen_array_ops.gather(flat_input, offset),
                                       [batch_size, seq_len])

  if sequence_lengths is None:
    unary_scores = math_ops.reduce_sum(unary_scores, 1)
  else:
    masks = array_ops.sequence_mask(sequence_lengths, seq_len,
                                    dtype=dtypes.float32)
    unary_scores = math_ops.reduce_sum(unary_scores * masks, 1)
  return unary_scores

def crf_binary_score(targets, sequence_lengths, transition_params):
  """Computes the binary scores of tag sequences.

  Args:
    targets: a [batch_size, seq_len] matrix of tag indices.
    masks: a [batch_size, seq_len] matrix of masks representing the first
        length positions of each row.
    transition_params: a [num_tags + 1, num_tags + 1] matrix of transition
        matrix.
  """
  # get shape meta info
  shape = array_ops.shape(targets)
  batch_size, seq_len = shape[0], shape[1]
  num_tags = array_ops.shape(transition_params)[0]

  # encode the indices
  start_indices = array_ops.concat_v2(
      [array_ops.zeros([batch_size, 1], dtype=dtypes.int32),
       array_ops.slice(targets, [0, 0], [-1, seq_len-1]) + 1], 1)
  end_indices = array_ops.slice(targets, [0, 0], [-1, seq_len])

  flat_transition_indices = start_indices * num_tags + end_indices
  binary_scores = gen_array_ops.gather(
      gen_array_ops.reshape(transition_params, [-1]), flat_transition_indices)

  if sequence_lengths is None:
    binary_scores = math_ops.reduce_sum(binary_scores, 1)
    sequence_lengths = gen_array_ops.fill([batch_size], seq_len)
  else:
    masks = array_ops.sequence_mask(sequence_lengths, seq_len,
                                    dtype=dtypes.float32)
    # truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = math_ops.reduce_sum(binary_scores * masks, 1)

  # get the transition score of the last step and the added special symbol
  # which denote 'end of states'
  # last_state_ids = tf.gather_nd(targets, tf.pack([tf.range(batch_size),
  #                                                 sequence_lengths], axis=1))
  offset = array_ops.expand_dims(math_ops.range(batch_size) * seq_len, 1)
  flattened_indices = gen_array_ops.reshape(
      offset + array_ops.expand_dims(sequence_lengths-1, -1), [-1])
  flattened_states = gen_array_ops.reshape(targets, [-1])
  last_state_ids = gen_array_ops.gather(flattened_states, flattened_indices)

  # last_trans_inds = tf.pack(
  #    [last_state_ids + 1, tf.fill([batch_size], num_tags-1)], 1)
  # last_trans_score = tf.gather_nd(transition_params, last_trans_inds)
  #
  flattened_indices = (last_state_ids + 1) * num_tags + (num_tags - 1)
  flattened_transition_params = gen_array_ops.reshape(transition_params, [-1])
  last_trans_score = gen_array_ops.gather(flattened_transition_params,
                                          flattened_indices)

  binary_scores += last_trans_score
  return binary_scores


def viterbi_decode(inputs, transition_params, sequence_lengths, name=None):
  """Viterbi decode given observations and transition params.

  Args:
    inputs: a [batch_size, seq_len, num_tags] matrix of unary potentials.
    transition_params: a [num_tags + 1, num_tags + 1] matrix of transition
        matrix.
    sequence_lengths: a [batch_size] vector of sequence lengths.
  """
  num_tags = inputs.get_shape()[1].value,

  first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
  first_input = array_ops.squeeze(first_input, [1])
  rest_of_inputs = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])

  # calcuate the first step score, that is scores starts from beginning symbol
  # to each state of the first step, which resulting in [batch_size, num_tags]
  # tensor
  first_alphas = array_ops.expand_dims(transition_params[0, :-1], 0) \
      + first_input

  viterbi_forward_cell = ViterbiForwardRnnCell(transition_params)
  backtracks, last_alphas = rnn.dynamic_rnn(
      cell=viterbi_forward_cell,
      inputs=rest_of_inputs,
      sequence_length=sequence_lengths - 1,
      initial_state=first_alphas,
      dtype=dtypes.int64)

  # calculate the score from the each state of the last step to the endding
  # symbol, which resulting in [batch_size, num_tags] vector
  final_alpha = last_alphas + array_ops.expand_dims(transition_params[1:, -1],
                                                    0)
  last_state_ids = math_ops.argmax(final_alpha, 1)

  # reverse sequence, sequence_lengths should minus 1, since the backtracks
  # always one-step behind
  backtracks = gen_array_ops.reverse_sequence(backtracks, sequence_lengths-1,
                                              seq_dim=1)

  viterbi_backtrack_cell = ViterbiBacktrackRnnCell(num_tags)
  viterbis, _ = rnn.dynamic_rnn(
      cell=viterbi_backtrack_cell,
      inputs=backtracks,
      sequence_length=sequence_lengths - 1,
      initial_state=array_ops.expand_dims(last_state_ids, -1),
      dtype=dtypes.int64)
  viterbis = array_ops.concat_v2([array_ops.expand_dims(last_state_ids, -1),
                                  array_ops.squeeze(viterbis, [2])], 1)
  viterbis = gen_array_ops.reverse_sequence(viterbis, sequence_lengths, 1)
  return math_ops.to_int32(viterbis, name=name)


class ViterbiForwardRnnCell(core_rnn_cell.RNNCell):
  """Computes forward score using viterbi in a linear-chain CRF.
  """
  def __init__(self, transition_params):
    """Initialize ViterbiForwardRnnCell.

    Args:
      transition_params: A [num_tags + 1, num_tags + 1] matrix of binary
          potentials.
    """
    self.transition_params = array_ops.expand_dims(transition_params[1:, :-1],
                                                   0)
    self.num_tags = transition_params.get_shape()[0].value - 1

  @property
  def state_size(self):
    return self.num_tags

  @property
  def output_size(self):
    return self.num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the ViterbiForwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.
    """
    state = array_ops.expand_dims(state, -1)
    transition_scores = state + self.transition_params

    new_state = inputs + math_ops.reduce_max(transition_scores,
                                             reduction_indices=[1])
    new_state_ids = math_ops.argmax(transition_scores, 1)

    return new_state_ids, new_state


class ViterbiBacktrackRnnCell(core_rnn_cell.RNNCell):
  """Compute optimal state at each step using backtrack in viterbi decoding.
  """
  def __init__(self, num_tags):
    """Initialize ViterbiBacktrackRnnCell

    Args:
      num_tags: num of tags of the linear-chain CRF
    """
    self.num_tags = num_tags

  @property
  def state_size(self):
    return 1

  @property
  def output_size(self):
    return 1

  def __call__(self, inputs, state, scope=None):
    """Construct viterbi backtracking

    Args:
        inputs: [batch_size, num_tags]
        state: [batch_size] vector
    """
    state = array_ops.squeeze(math_ops.to_int32(state), [1])
    batch_size = array_ops.shape(inputs)[0]
    offset = math_ops.range(batch_size) * self.num_tags
    flattened_indices = gen_array_ops.reshape(offset + state, [-1])
    flattened_inputs = gen_array_ops.reshape(inputs, [-1])
    new_state_ids = array_ops.expand_dims(
        gen_array_ops.gather(flattened_inputs, flattened_indices), -1)
    return new_state_ids, new_state_ids


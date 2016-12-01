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

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs

__all__ = ["crf_sequence_score", "crf_log_norm", "crf_log_likelihood",
           "crf_unary_score", "crf_binary_score", "CrfForwardRnnCell",
           "viterbi_decode"]


def _lengths_to_masks(lengths, max_length):
  """Creates a binary matrix that can be used to mask away padding.

  Args:
    lengths: A vector of integers representing lengths.
    max_length: An integer indicating the maximum length. All values in
      lengths should be less than max_length.
  Returns:
    masks: Masks that can be used to get rid of padding.
  """
  tiled_ranges = array_ops.tile(
      array_ops.expand_dims(math_ops.range(max_length), 0),
      [array_ops.shape(lengths)[0], 1])
  lengths = array_ops.expand_dims(lengths, 1)
  masks = math_ops.to_float(
      math_ops.to_int64(tiled_ranges) < math_ops.to_int64(lengths))
  return masks


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
  """Computes the unnormalized score for a tag sequence.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the unnormalized score.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.
  Returns:
    sequence_scores: A [batch_size] vector of unnormalized sequence scores.
  """
  # Compute the scores of the given tag sequence.
  unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
  binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                   transition_params)
  sequence_scores = unary_scores + binary_scores
  return sequence_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
  """Computes the normalization for a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.
  Returns:
    log_norm: A [batch_size] vector of normalizers for a CRF.
  """
  # Split up the first and rest of the inputs in preparation for the forward
  # algorithm.
  first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
  first_input = array_ops.squeeze(first_input, [1])
  rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])

  # Compute the alpha values in the forward algorithm in order to get the
  # partition function.
  forward_cell = CrfForwardRnnCell(transition_params)
  _, alphas = rnn.dynamic_rnn(
      cell=forward_cell,
      inputs=rest_of_input,
      sequence_length=sequence_lengths - 1,
      initial_state=first_input,
      dtype=dtypes.float32)
  log_norm = math_ops.reduce_logsumexp(alphas, [1])
  return log_norm


def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
  """Computes the log-likelihood of tag sequences in a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the log-likelihood.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix, if available.
  Returns:
    log_likelihood: A scalar containing the log-likelihood of the given sequence
        of tag indices.
    transition_params: A [num_tags, num_tags] transition matrix. This is either
        provided by the caller or created in this function.
  """
  # Get shape information.
  num_tags = inputs.get_shape()[2].value

  # Get the transition matrix if not provided.
  if transition_params is None:
    transition_params = vs.get_variable("transitions", [num_tags, num_tags])

  sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
                                       transition_params)
  log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

  # Normalize the scores to get the log-likelihood.
  log_likelihood = sequence_scores - log_norm
  return log_likelihood, transition_params


def crf_unary_score(tag_indices, sequence_lengths, inputs):
  """Computes the unary scores of tag sequences.

  Args:
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
  Returns:
    unary_scores: A [batch_size] vector of unary scores.
  """
  batch_size = array_ops.shape(inputs)[0]
  max_seq_len = array_ops.shape(inputs)[1]
  num_tags = array_ops.shape(inputs)[2]

  flattened_inputs = array_ops.reshape(inputs, [-1])

  offsets = array_ops.expand_dims(
      math_ops.range(batch_size) * max_seq_len * num_tags, 1)
  offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * num_tags, 0)
  flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])

  unary_scores = array_ops.reshape(
      array_ops.gather(flattened_inputs, flattened_tag_indices),
      [batch_size, max_seq_len])

  masks = _lengths_to_masks(sequence_lengths, array_ops.shape(tag_indices)[1])

  unary_scores = math_ops.reduce_sum(unary_scores * masks, 1)
  return unary_scores


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
  """Computes the binary scores of tag sequences.

  Args:
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    binary_scores: A [batch_size] vector of binary scores.
  """
  # Get shape information.
  num_tags = transition_params.get_shape()[0]
  num_transitions = array_ops.shape(tag_indices)[1] - 1

  # Truncate by one on each side of the sequence to get the start and end
  # indices of each transition.
  start_tag_indices = array_ops.slice(tag_indices, [0, 0],
                                      [-1, num_transitions])
  end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])

  # Encode the indices in a flattened representation.
  flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
  flattened_transition_params = array_ops.reshape(transition_params, [-1])

  # Get the binary scores based on the flattened representation.
  binary_scores = array_ops.gather(flattened_transition_params,
                                   flattened_transition_indices)

  masks = _lengths_to_masks(sequence_lengths, array_ops.shape(tag_indices)[1])
  truncated_masks = array_ops.slice(masks, [0, 1], [-1, -1])
  binary_scores = math_ops.reduce_sum(binary_scores * truncated_masks, 1)
  return binary_scores


class CrfForwardRnnCell(rnn_cell.RNNCell):
  """Computes the alpha values in a linear-chain CRF.

  See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
  """

  def __init__(self, transition_params):
    """Initialize the CrfForwardRnnCell.

    Args:
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
    """
    self._transition_params = array_ops.expand_dims(transition_params, 0)
    self._num_tags = transition_params.get_shape()[0].value

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the CrfForwardRnnCell.

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.

    Returns:
      new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
          values containing the new alpha values.
    """
    state = array_ops.expand_dims(state, 2)

    # This addition op broadcasts self._transitions_params along the zeroth
    # dimension and state along the second dimension. This performs the
    # multiplication of previous alpha values and the current binary potentials
    # in log space.
    transition_scores = state + self._transition_params
    new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])

    # Both the state and the output of this RNN cell contain the alphas values.
    # The output value is currently unused and simply satisfies the RNN API.
    # This could be useful in the future if we need to compute marginal
    # probabilities, which would require the accumulated alpha values at every
    # time step.
    return new_alphas, new_alphas


def viterbi_decode(score, transition_params):
  """Decode the highest scoring sequence of tags outside of TensorFlow.

  This should only be used at test time.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indicies.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score

# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Helper functions for decoding Viterbi sequences outside of Tensorflow.

viterbi_decode provides known-tested snippets for Viterbi decoding in log and
standard space for use outside of a Tensorflow graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def decode(score,
           transition_params=None,
           allowed_transitions=None,
           use_log_space=True,
           use_start_and_end_states=False):
  """Decode the highest scoring sequence of tags.

  This function uses numpy instead of Tensorflow ops, and so cannot be used
  inside a Tensorflow graph or function.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    allowed_transitions: A [num_tags, num_tags] matrix where FALSE indicates
      a transition that cannot be taken.
    use_log_space: Whether to perform the Viterbi calculation in logarithmic
      space.
    use_start_and_end_states: If True, add an implicit 'start' and 'end' state
      to the start and end of the given sequence. If this is True,
      transition_params should contain an extra row and column, representing
      potentials for starting/ending a sequence with a given state. These values
      should occupy the outermost row and column of the transition_params
      matrix.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  if transition_params is None:
    num_tags = score.shape[-1]
    if use_log_space:
      transition_params = np.zeros(num_tags, num_tags)
    else:
      transition_params = np.ones(num_tags, num_tags)

  if allowed_transitions is not None:
    if use_log_space:
      transition_mask = np.where(allowed_transitions, 1, -float("inf"))
    else:
      transition_mask = np.where(allowed_transitions, 1, 0.0)

    transition_params = transition_params * transition_mask

  if use_log_space:
    return _decode_in_log_space(score, transition_params,
                                use_start_and_end_states)
  else:
    return _decode_in_exp_space(score, transition_params,
                                use_start_and_end_states)


def _decode_in_log_space(score, transition_params, use_start_and_end_states):
  """Perform Viterbi decoding in log space."""
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)

  if use_start_and_end_states:
    start_potentials = transition_params[-1, :-1]
    end_potentials = transition_params[:-1, -1]
    transition_potentials = transition_params[:-1, :-1]
  else:
    transition_potentials = transition_params

  # Calculate the start value.
  if use_start_and_end_states:
    trellis[0] = score[0] + start_potentials
  else:
    trellis[0] = score[0]

  # Calculate intermediate values.
  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_potentials
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  # If we are using explicit start and end states, change the final scores
  # based on the final state's potentials.
  if use_start_and_end_states:
    final_scores = trellis[-1] + end_potentials
  else:
    final_scores = trellis[-1]

  viterbi = [np.argmax(final_scores)]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(final_scores)

  return viterbi, viterbi_score


def _decode_in_exp_space(score, transition_params, use_start_and_end_states):
  """Perform Viterbi decoding in exp space."""
  if np.any(transition_params < 0):
    raise ValueError("Transition params must be non-negative in exp space.")
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  max_scores = np.zeros(score.shape[0])

  if use_start_and_end_states:
    start_potentials = transition_params[-1, :-1]
    end_potentials = transition_params[:-1, -1]
    transition_potentials = transition_params[:-1, :-1]
  else:
    transition_potentials = transition_params

  # Calculate the start value.
  if use_start_and_end_states:
    trellis[0] = score[0] * start_potentials
  else:
    trellis[0] = score[0]

  max_scores[0] = np.max(trellis[0])
  trellis[0] = trellis[0] / max_scores[0]

  # Calculate intermediate values.
  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) * transition_potentials
    trellis[t] = score[t] * np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)
    max_scores[t] = np.max(trellis[t])
    trellis[t] = trellis[t] / max_scores[t]

  # If we are using explicit start and end states, change the final scores
  # based on the final state's potentials.
  if use_start_and_end_states:
    final_scores = trellis[-1] * end_potentials
  else:
    final_scores = trellis[-1]

  viterbi = [np.argmax(final_scores)]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(final_scores) * np.prod(max_scores)
  return viterbi, viterbi_score

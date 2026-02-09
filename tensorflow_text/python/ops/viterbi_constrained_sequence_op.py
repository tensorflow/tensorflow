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

"""Bulk Viterbi Constrained Sequence.

Constrains a set of predictions based on a set of legal transitions and/or a
set of transition weights, returning the legal sequence that maximizes the
product of the state scores and the transition weights according to the Viterbi
algorithm.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_constrained_sequence_op = load_library.load_op_library(resource_loader.get_path_to_datafile('_constrained_sequence_op.so'))


def viterbi_constrained_sequence(scores,
                                 sequence_length=None,
                                 allowed_transitions=None,
                                 transition_weights=None,
                                 use_log_space=False,
                                 use_start_and_end_states=True,
                                 name=None):
  """Performs greedy constrained sequence on a batch of examples.

  Constrains a set of predictions based on a set of legal transitions
  and/or a set of transition weights, returning the legal sequence that
  maximizes the product of the state scores and the transition weights
  according to the Viterbi algorithm. If `use_log_space` is True, the Viterbi
  calculation will be performed in log space (with sums); if it is False,
  the Viterbi calculation will be performed in exp space (with normalized
  products).

  This op also takes a parameter `use_start_and_end_states`, which when true
  will add an implicit start and end state to each sequence. These implicit
  states allow the user to specify additional weights and permitted transitions
  to start and end a sequence (so, for instance, if you wanted to forbid your
  output from ending in a certain set of states you could do so).

  Inputs to this op can take one of three forms: a single TensorFlow tensor
  of scores with no sequence lengths, a TensorFlow tensor of scores along
  with a TensorFlow tensor of sequence lengths, or a RaggedTensor. If only the
  scores tensor is passed, this op will assume that the sequence lengths are
  equal to the size of the tensor (and so use all the data provided). If a
  scores tensor and sequence_lengths tensor is provided, the op will only
  use the data in the scores tensor as specified by the sequence_lengths tensor.
  Finally, if a RaggedTensor is provided, the sequence_lengths will be ignored
  and the variable length sequences in the RaggedTensor will be used.

  >>> scores = np.array([[10.0, 12.0, 6.0, 4.0],
  ...                    [13.0, 12.0, 11.0, 10.0]], dtype=np.float32)
  >>> sequence_length = np.array([2])
  >>> transition_weights = np.array([[ .1,  .2,  .3,  .4],
  ...                                [ .5,  .6,  .7,  .8],
  ...                                [ .9,  .1, .15,  .2],
  ...                                [.25, .35, .45, .55]], dtype=np.float32)
  >>> allowed_transitions = np.array([[True,  True,  True,  True],
  ...                                 [True,  True,  True,  True],
  ...                                 [True, False,  True, False],
  ...                                 [True,  True,  True,  True]])
  >>> viterbi_constrained_sequence(
  ...      scores=scores,
  ...      sequence_length=sequence_length,
  ...      allowed_transitions=allowed_transitions,
  ...      transition_weights=transition_weights,
  ...      use_log_space=False,
  ...      use_start_and_end_states=False)
  <tf.RaggedTensor [[1, 3]]>

  Args:
    scores: `<float32> [batch_size, num_steps, |num_states|]`
      A tensor of scores, where `scores[b, t, s]` is the predicted score for
      transitioning to state `s` at step `t` for batch `b`. The |num_states|
      dimension must correspond to the num_states attribute for this op. This
      input may be ragged; if it is ragged, the ragged tensor should have the
      same structure [b, t, s] and only axis 1 should be ragged.

    sequence_length: `<{int32, int64}>[batch_size]`
      A rank-1 tensor representing the length of the output sequence. If None,
      and the 'scores' input is not ragged, sequence lengths will be assumed
      to be the length of the score tensor.

    allowed_transitions:
      if use_start_and_end_states is TRUE:
        `<bool>[num_states+1, num_states+1]`
      if use_start_and_end_states is FALSE:
        `<bool>[num_states, num_states]`
      A rank-2 tensor representing allowed transitions.
      - allowed_transitions[i][j] is true if the transition from state i to
          state j is allowed for i and j in 0...(num_states).
      - allowed_transitions[num_states][num_states] is ignored.
      If use_start_and_end_states is TRUE:
        - allowed_transitions[num_states][j] is true if the sequence is allowed
            to start from state j.
        - allowed_transitions[i][num_states] is true if the sequence is allowed
            to end on state i.
      Default - An empty tensor. This allows all sequence states to transition
        to all other sequence states.

    transition_weights:
      if use_start_and_end_states is TRUE:
        `<float32>[num_states+1, num_states+1]`
      if use_start_and_end_states is FALSE:
        `<float32>[num_states, num_states]`
      A rank-2 tensor representing transition weights.
      - transition_weights[i][j] is the coefficient that a candidate transition
          score will be multiplied by if that transition is from state i to
          state j.
      - transition_weights[num_states][num_states] is ignored.
      If use_start_and_end_states is TRUE:
        - transition_weights[num_states][j] is the coefficient that will be used
            if the transition starts with state j.
        - transition_weights[i][num_states] is the coefficient that will be used
            if the final state in the sequence is state i.
      Default - An empty tensor. This assigns a wieght of 1.0 all transitions

    use_log_space: Whether to use log space for the calculation. If false,
      calculations will be done in exp-space.

    use_start_and_end_states: If True, sequences will have an implicit start
      and end state added.

    name: The name scope within which this op should be constructed.

  Returns:
    An <int32>[batch_size, (num_steps)] ragged tensor containing the appropriate
    sequence of transitions. If a sequence is impossible, the value of the
    RaggedTensor for that and all following transitions in that sequence shall
    be '-1'.
  """
  with ops.name_scope(
      name, "BulkViterbiConstrainedSequence",
      [scores, sequence_length, allowed_transitions, transition_weights]):
    if allowed_transitions is None:
      allowed_transitions = []

    if transition_weights is None:
      transition_weights = []

    score_data = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        scores, name="score_data")

    if isinstance(score_data, ragged_tensor.RaggedTensor):
      # TODO(momernick): Extend the generated op to support ragged tensors.
      dense_scores = score_data.to_tensor(default_value=0)
      sequence_lengths = score_data.row_lengths(axis=1)
    else:
      dense_scores = score_data
      # In this case, the core input was a dense tensor.
      if sequence_length is not None:
        sequence_lengths = ops.convert_to_tensor(sequence_length)
      else:
        batch_size = array_ops.shape(dense_scores)[0]
        dense_length = array_ops.shape(dense_scores)[-2]
        sequence_lengths = array_ops.ones([batch_size],
                                          dtype=dtypes.int32) * dense_length

    transition_weights = ops.convert_to_tensor(transition_weights)
    allowed_transitions = ops.convert_to_tensor(
        allowed_transitions, dtype=dtypes.bool)

    output, output_splits = gen_constrained_sequence_op.constrained_sequence(
        scores=dense_scores,
        sequence_lengths=sequence_lengths,
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_viterbi=True,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    return ragged_tensor.RaggedTensor.from_row_splits(
        values=output, row_splits=output_splits)

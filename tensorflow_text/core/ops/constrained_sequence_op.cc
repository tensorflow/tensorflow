// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("ConstrainedSequence")
    .Attr("Tin: {int32, int64}")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Attr("use_viterbi: bool")
    .Attr("use_log_space: bool")
    .Attr("use_start_and_end_states: bool")
    .Input("scores: float")
    .Input("sequence_lengths: Tin")
    .Input("allowed_transitions: bool")
    .Input("transition_weights: float")
    .Output("states: int32")
    .Output("states_splits: Tsplits")

    // TODO(b/122968457): Implement a shape function.
    .Doc(R"doc(
Constrains a set of predictions based on a set of legal transitions and/or a
set of transition weights, returning the legal sequence that maximizes the
product of the state scores and the transition weights using the chained
conditional random field algorithm. (In case of a tie, the state with a higher
index will be chosen.)

This op takes in a set of scores and outputs the most likely legal sequence
for each batch element, where the most likely legal sequence is determined by
the optional 'allowed_transitions' and 'transition_weights' tensors.

The 'allowed_transition' tensor may be omitted; if it is, all sequence states
will be allowed to transition to all other sequence states. If the tensor is
provided it must be of the size [num_states+1][num_states+1].

allowed_transitions[i][j] is true if the transition from state i to state
j is allowed for i and j in 0...(num_states).
allowed_transitions[num_states][j] is true if the sequence is allowed to
start from state j.
allowed_transitions[i][num_states] is true if the sequence is allowed to
end on state i.
allowed_transitions[num_states][num_states] is ignored.

The 'transition_weights' tensor may be omitted; if it is, all transitions will
be weighted with a value of 1.0. If the tensor is provided it must be of the
size [num_states+1][num_states+1].

transition_weights[i][j] is the coefficient that a candidate transition score
will be multiplied by if that transition is from state i to state j.
transition_weights[num_states][j] is the coefficient that will be used
if the transition starts with state j.
transition_weights[i][num_states] is the coefficient that will be used
if the final state in the sequence is state i.
transition_weights[num_states][num_states] is ignored.

This op outputs a RaggedTensor value and splits pair.

scores: <float>[batch_size, num_steps, |num_states|] A tensor of scores, where
        `scores[b, t, s]` is the predicted score for transitioning to state `s`
        at step `t` for batch `b`. The |num_states| dimension must correspond
        to the num_states attribute for this op.
sequence_lengths: <{int32, int64}>[batch_size] A tensor containing the length
        of each sequence in the batch.
allowed_transitions: <bool>[num_states+1, num_states+1] A boolean matrix of
        allowed transitions, or an empty matrix '[]' to allow all transitions.
transition_weights: <float>[num_states+1, num_states+1] A float matrix of score
        coefficients, or an empty matrix '[]' to weight all transitions equally.
states: <int32>[batch_size, max_sequence_length] OR <int32>[total_num_states]
        A set of sequence outputs representing the most likely valid sequences
        for each batch. If `output_ragged_tensor` is false, this will be in
        [batch_size, max_sequence_length] form; if `output_ragged_tensor` is
        true, this will be a RaggedTensor data vector of shape
        [total_num_states].
states_splits: <int64>[batch_size+1] A RaggedTensor splits vector. If
        `output_ragged_tensor` is true, then the state sequence for input `i`
        is stored in `states[states_splits[i]:states_splits[i+1]]`.  If
        `output_ragged_tensor` is false, this tensor will be empty and can be
        ignored.
)doc");

}  // namespace tensorflow

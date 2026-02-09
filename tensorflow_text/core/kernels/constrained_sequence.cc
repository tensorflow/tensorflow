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

#include "tensorflow_text/core/kernels/constrained_sequence.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace text {

// State index to use if the sequence in question requires an impossible
// transition.
constexpr int kErrorState = -1;

ScoreAccessor::ScoreAccessor(const Tensor &score_tensor,
                             const Tensor &lengths_tensor) {
  data_ = score_tensor.flat<float>().data();
  if (lengths_tensor.dtype() == DT_INT64) {
    use_long_lengths_ = true;
    long_lengths_ = lengths_tensor.flat<int64>().data();
  } else {
    use_long_lengths_ = false;
    lengths_ = lengths_tensor.flat<int>().data();
  }
  has_explicit_batch_ = (score_tensor.shape().dims() == 3);
  if (has_explicit_batch_) {
    batch_size_ = score_tensor.shape().dim_size(0);
    num_steps_ = score_tensor.shape().dim_size(1);
    num_scores_ = score_tensor.shape().dim_size(2);
  } else {
    batch_size_ = 1;
    num_steps_ = score_tensor.shape().dim_size(0);
    num_scores_ = score_tensor.shape().dim_size(1);
  }
  batch_offset_ = num_scores_ * num_steps_;
  step_offset_ = num_scores_;
}

// Get a score out of the data tensor.
float ScoreAccessor::GetScore(int batch_idx, int step_idx,
                              int score_idx) const {
  DCHECK_LE(batch_idx, batch_size_);
  DCHECK_LE(step_idx, num_steps_);
  DCHECK_LE(score_idx, num_scores_);
  return data_[batch_offset_ * batch_idx + step_offset_ * step_idx + score_idx];
}

int64 ScoreAccessor::GetLength(int batch_idx) const {
  DCHECK_LE(batch_idx, batch_size_);
  if (use_long_lengths_) {
    return long_lengths_[batch_idx];
  } else {
    return lengths_[batch_idx];
  }
}

int ScoreAccessor::batch_size() const { return batch_size_; }
int ScoreAccessor::num_steps() const { return num_steps_; }
int ScoreAccessor::num_scores() const { return num_scores_; }
bool ScoreAccessor::has_explicit_batch() const { return has_explicit_batch_; }

// Perform Viterbi analysis on a single batch item.
void ViterbiAnalysis(
    const ScoreAccessor &scores,
    const tensorflow::TTypes<const float>::Matrix &transition_weights,
    const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
    const int batch, bool use_log_space, bool use_start_end_states,
    int32 *output_data) {
  VLOG(2) << "Analyzing batch " << batch;
  const bool has_transition_weights = transition_weights.size() != 0;
  const bool has_allowed_transitions = allowed_transitions.size() != 0;
  const int num_states = scores.num_scores();
  const int out_of_bounds_index = num_states;

  int64 num_steps = scores.GetLength(batch);

  // Create two vectors to hold scores. These will be bound to referents later
  // so the names here are somewhat irrelevant.
  std::vector<double> scores_a(num_states,
                               std::numeric_limits<float>::lowest());
  std::vector<double> scores_b(num_states,
                               std::numeric_limits<float>::lowest());

  // Create a chart of backpointers. Include rows for [start] and [end]
  // transitions. By initializing this to kErrorState, we ensure unreachable
  // transitions get marked as errors.
  std::vector<std::vector<int>> backpointers(
      num_steps, std::vector<int>(num_states, kErrorState));

  // Set current and previous references for step 0
  std::vector<double> *previous_scores = &scores_a;
  std::vector<double> *current_scores = &scores_b;

  const bool vlog3 = VLOG_IS_ON(3);

  if (backpointers.empty()) {
    // We're done with this batch if there are no steps to analyze.
    return;
  }
  for (int curr_state = 0; curr_state < num_states; ++curr_state) {
    std::vector<int> &current_bps = backpointers[0];
    if (use_start_end_states) {
      // Initialize the zeroth step BPs to kOutOfBoundsIndex for all states
      // where the OOB->state transition is valid, and set scores as needed.
      if (has_allowed_transitions &&
          !allowed_transitions(out_of_bounds_index, curr_state)) {
        if (vlog3) {
          LOG(INFO) << "(" << batch << ", 0, [START]->" << curr_state
                    << "): disallowed.";
        }
        continue;
      }

      // Because the backpointer vectors are initialized to kErrorState, we
      // need only to set the valid transition paths to have come from the
      // padding state.
      current_bps[curr_state] = out_of_bounds_index;

      // For valid transitions, get the score (and adjust as appropriate).
      const int step = 0;
      float current_score = scores.GetScore(batch, step, curr_state);
      if (has_transition_weights) {
        if (use_log_space) {
          current_score += transition_weights(out_of_bounds_index, curr_state);
        } else {
          current_score *= transition_weights(out_of_bounds_index, curr_state);
        }
      }

      if (vlog3) {
        if (has_transition_weights) {
          LOG(INFO) << "(" << batch << ", " << step << ", [START]->"
                    << curr_state << "): Total score: " << current_score
                    << " (raw: " << scores.GetScore(batch, step, curr_state)
                    << ", tw: "
                    << transition_weights(out_of_bounds_index, curr_state)
                    << ")";
        } else {
          LOG(INFO) << "(" << batch << ", " << step << ", [START]->"
                    << curr_state << "): Total score: " << current_score
                    << " (raw: " << scores.GetScore(batch, step, curr_state)
                    << ")";
        }
      }

      current_scores->at(curr_state) = current_score;
    } else {
      // If we don't have specific start and end states, all bp's are valid
      // and all starting scores are the unadjusted step 0 scores.
      current_bps[curr_state] = out_of_bounds_index;
      const int step = 0;
      current_scores->at(curr_state) = scores.GetScore(batch, step, curr_state);
    }
  }

  // Update the current scores (and normalize if we're not in log space).
  if (!use_log_space) {
    const double max_score =
        *std::max_element(current_scores->begin(), current_scores->end());
    if (max_score > 0) {
      for (double &score : *current_scores) score /= max_score;
    }
  }

  // Swap current and previous score arrays, as we are advancing a step.
  std::vector<double> *tmp = previous_scores;
  previous_scores = current_scores;
  current_scores = tmp;

  // Handle all steps save for the first and last in this loop.
  for (int step = 1; step < num_steps; ++step) {
    const std::vector<int> &previous_bps = backpointers[step - 1];
    std::vector<int> &current_bps = backpointers[step];

    for (int curr_state = 0; curr_state < num_states; ++curr_state) {
      int best_source_state = kErrorState;
      float best_score = std::numeric_limits<float>::lowest();
      for (int prev_state = 0; prev_state < num_states; ++prev_state) {
        // If the previous state was an error state, pass to the next state.
        if (previous_bps[prev_state] == kErrorState) {
          if (vlog3) {
            LOG(INFO) << "(" << batch << ", " << step << ", " << prev_state
                      << "->" << curr_state << "): prev state error.";
          }
          continue;
        }

        // If this is not a permitted transition, continue.
        if (has_allowed_transitions &&
            !allowed_transitions(prev_state, curr_state)) {
          if (vlog3) {
            LOG(INFO) << "(" << batch << ", " << step << ", " << prev_state
                       << "->" << curr_state << "): disallowed.";
          }
          continue;
        }

        float current_score = scores.GetScore(batch, step, curr_state);
        if (use_log_space) {
          current_score += previous_scores->at(prev_state);
        } else {
          current_score *= previous_scores->at(prev_state);
        }
        if (has_transition_weights) {
          if (use_log_space) {
            current_score += transition_weights(prev_state, curr_state);
          } else {
            current_score *= transition_weights(prev_state, curr_state);
          }
        }

        if (vlog3) {
          if (has_transition_weights) {
            LOG(INFO) << "(" << batch << ", " << step << ", " << prev_state
                      << "->" << curr_state
                      << "): Total score: " << current_score
                      << " (prev: " << previous_scores->at(prev_state)
                      << ", raw: " << scores.GetScore(batch, step, curr_state)
                      << ", tw: " << transition_weights(prev_state, curr_state)
                      << ")";
          } else {
            LOG(INFO) << "(" << batch << ", " << step << ", " << prev_state
                      << "->" << curr_state
                      << "): Total score: " << current_score
                      << " (prev: " << previous_scores->at(prev_state)
                      << ", raw: " << scores.GetScore(batch, step, curr_state)
                      << ")";
          }
        }

        if (current_score >= best_score) {
          best_source_state = prev_state;
          best_score = current_score;
        }
      }
      current_bps[curr_state] = best_source_state;
      current_scores->at(curr_state) = best_score;
    }

    // Normalize if we're not in log space.
    if (!use_log_space) {
      const double max_score =
          *std::max_element(current_scores->begin(), current_scores->end());
      if (max_score > 0) {
        for (double &score : *current_scores) score /= max_score;
      }
    }

    // After each step, switch the current scores to the previous scores and
    // use the previous previous scores as the current scores.
    std::vector<double> *tmp = previous_scores;
    previous_scores = current_scores;
    current_scores = tmp;
  }

  // Handle the final transition out of the sequence.
  int final_state = out_of_bounds_index;
  const std::vector<int> &previous_bps = backpointers[num_steps - 1];
  int best_source_state = kErrorState;
  float final_score = std::numeric_limits<float>::lowest();

  for (int prev_state = 0; prev_state < num_states; ++prev_state) {
    // If the previous state was an error state, pass to the next state.
    if (previous_bps[prev_state] == kErrorState) {
      current_scores->at(prev_state) = std::numeric_limits<float>::lowest();
      if (vlog3) {
        LOG(INFO) << "(" << batch << ", " << num_steps << ", " << prev_state
                  << "->[END]): prev state error.";
      }
      continue;
    }

    // If this is not a permitted transition, continue.
    if (has_allowed_transitions && use_start_end_states &&
        !allowed_transitions(prev_state, final_state)) {
      current_scores->at(prev_state) = std::numeric_limits<float>::lowest();
      if (vlog3) {
        LOG(INFO) << "(" << batch << ", " << num_steps << ", " << prev_state
                  << "->[END]): disallowed.";
      }
      continue;
    }

    // Weight the final transition score by the probability of exiting the
    // sequence as well.
    float current_score = previous_scores->at(prev_state);
    if (use_start_end_states) {
      if (has_transition_weights) {
        if (use_log_space) {
          current_score += transition_weights(prev_state, final_state);
        } else {
          current_score *= transition_weights(prev_state, final_state);
        }
      }

      if (vlog3) {
        if (has_transition_weights) {
          LOG(INFO) << "(" << batch << ", " << num_steps << ", " << prev_state
                    << "->[END]): Total score: " << current_score
                    << " (prev: " << previous_scores->at(prev_state)
                    << ", tw: " << transition_weights(prev_state, final_state)
                    << ")";
        } else {
          LOG(INFO) << "(" << batch << ", " << num_steps << ", " << prev_state
                    << "->[END]): Total score: " << current_score
                    << " (prev: " << previous_scores->at(prev_state) << ")";
        }
      }
    }

    current_scores->at(prev_state) = current_score;
    if (current_score >= final_score) {
      best_source_state = prev_state;
      final_score = current_score;
    }
  }

  if (vlog3) {
    LOG(INFO) << "Final score: " << final_score;
  }

  // Calculate the path.
  if (best_source_state == kErrorState) {
    // If the best source is an error state, the path is unknowable. Report
    // error states for the whole sequence.
    for (int64 i = 0; i < scores.GetLength(batch); ++i) {
      output_data[i] = kErrorState;
    }
  } else {
    // If the best source is a 'real' state, report the state path.
    int steps_to_report = scores.GetLength(batch);
    int previous_state = best_source_state;
    for (int64 i = steps_to_report - 1; i >= 0; --i) {
      output_data[i] = previous_state;
      previous_state = backpointers[i][previous_state];
    }
  }
}

void GreedyAnalysis(
    const ScoreAccessor &scores,
    const tensorflow::TTypes<const float>::Matrix &transition_weights,
    const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
    int batch, bool use_log_space, bool use_start_end_states,
    int32 *output_data) {
  const bool has_transition_weights = transition_weights.size() != 0;
  const bool has_allowed_transitions = allowed_transitions.size() != 0;
  const int num_states = scores.num_scores();
  const int out_of_bounds_index = num_states;
  int64 num_steps = scores.GetLength(batch);

  for (int step = 0; step < num_steps; ++step) {
    // Do final step calculations if this is the final step in the sequence
    // and we are calculating based on implicit start and end states.
    bool do_final_step =
        (step == scores.GetLength(batch) - 1) && use_start_end_states;
    VLOG(2) << "is last step: " << do_final_step;

    const int previous_state =
        (step == 0) ? (out_of_bounds_index) : (output_data[step - 1]);

    if (previous_state == kErrorState) {
      // If the previous state is the error state, the current state must
      // also be the error state.
      output_data[step] = kErrorState;
      continue;
    }

    // If no transition is possible, this will stay the error state.
    int best_new_state = kErrorState;
    float best_new_score = std::numeric_limits<float>::lowest();

    for (int state = 0; state < num_states; ++state) {
      float current_score = scores.GetScore(batch, step, state);

      // If we are not using start/end states AND step is 0, then
      // current_score will not be altered.
      if (use_start_end_states || step > 0) {
        if (has_allowed_transitions) {
          // If either the transition from the previous state to this state
          // is disallowed, or we need to analyze the final step and the
          // transition from this state to the final step is not allowed,
          // disallow this transition.
          if (!allowed_transitions(previous_state, state) ||
              (do_final_step &&
               !allowed_transitions(state, out_of_bounds_index))) {
            continue;
          }
        }

        if (has_transition_weights) {
          if (use_log_space) {
            current_score += transition_weights(previous_state, state);
          } else {
            current_score *= transition_weights(previous_state, state);
          }
          // On the last step, also analyze by the weight value of
          // transitioning from this state to the out-of-bounds state.
          if (do_final_step) {
            if (use_log_space) {
              current_score += transition_weights(state, out_of_bounds_index);
            } else {
              current_score *= transition_weights(state, out_of_bounds_index);
            }
          }
        }
      }
      if (current_score >= best_new_score) {
        best_new_state = state;
        best_new_score = current_score;
      }
    }
    output_data[step] = best_new_state;
    VLOG(2) << "Best state for step " << step << " is " << output_data[step]
            << " with score " << best_new_score;
  }
}

}  // namespace text
}  // namespace tensorflow

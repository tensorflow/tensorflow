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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_CONSTRAINED_SEQUENCE_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_CONSTRAINED_SEQUENCE_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace text {

class ScoreAccessor {
 public:
  explicit ScoreAccessor(const Tensor &score_tensor,
                         const Tensor &lengths_tensor);

  // Get a score out of the data tensor.
  float GetScore(int batch_idx, int step_idx, int score_idx) const;

  int64 GetLength(int batch_idx) const;

  int batch_size() const;
  int num_steps() const;
  int num_scores() const;
  bool has_explicit_batch() const;

 private:
  // A pointer into the underlying data of the score tensor. Not owned.
  const float *data_;

  // A pointer into the underlying data of the lengths tensor. Not owned.
  const int *lengths_;
  const int64 *long_lengths_;

  // Whether the passed lengths tensor is int32 or int64.
  bool use_long_lengths_;

  // The batch size associated with the data tensor.
  int batch_size_;

  // The number of steps in the data tensor.
  int num_steps_;

  // The number of scores in the data tensor.
  int num_scores_;

  // The amount to increase the offset within the flat data array if the batch
  // index increases by 1.
  int batch_offset_;

  // The amount to increase the offset within the flat data array if the step
  // index increases by 1.
  int step_offset_;

  // True if the original tensor had an explicit batch dimension (that is,
  // it was of rank 3).
  bool has_explicit_batch_;
};

// Perform Viterbi analysis on a single batch item.
void ViterbiAnalysis(
    const ScoreAccessor &scores,
    const tensorflow::TTypes<const float>::Matrix &transition_weights,
    const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
    const int batch, bool use_log_space, bool use_start_end_states,
    int32 *output_data);

// Perform a greedy analysis on a single batch item.
void GreedyAnalysis(
    const ScoreAccessor &scores,
    const tensorflow::TTypes<const float>::Matrix &transition_weights,
    const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
    int batch, bool use_log_space, bool use_start_end_states,
    int32 *output_data);

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_CONSTRAINED_SEQUENCE_H_

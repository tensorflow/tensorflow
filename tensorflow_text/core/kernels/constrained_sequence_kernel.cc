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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_text/core/kernels/constrained_sequence.h"

namespace tensorflow {

using ::tensorflow::DataType;
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::DT_BOOL;
using ::tensorflow::DT_FLOAT;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::text::GreedyAnalysis;
using ::tensorflow::text::ScoreAccessor;
using ::tensorflow::text::ViterbiAnalysis;

// State index to use if the sequence in question requires an impossible
// transition.
constexpr int kErrorState = -1;

// State index to use when outputting a padded tensor and the sequence in
// question does not have a token for a given step.
constexpr int kPaddingState = -2;

namespace {

// Validate that a given constraint tensor is the proper shape (dimension
// 2, with shape [num_states + 1, num_states + 1].
absl::Status ValidateConstraintTensor(const Tensor &tensor,
                                      const int num_states,
                                      const bool use_start_end_states,
                                      const string &name) {
  if (tensor.shape().dims() != 2) {
    return InvalidArgument(
        tensorflow::strings::StrCat(name, " must be of rank 2"));
  }
  int expected_size = use_start_end_states ? num_states + 1 : num_states;
  if (tensor.shape().dim_size(0) != expected_size) {
    return InvalidArgument(tensorflow::strings::StrCat(
        name, " must have a zeroth dimension of size ", expected_size,
        " when num_states is ", num_states, " and use_start_and_end_states is ",
        use_start_end_states));
  }
  if (tensor.shape().dim_size(1) != expected_size) {
    return InvalidArgument(tensorflow::strings::StrCat(
        name, " must have a first dimension of size ", expected_size,
        " when num_states is ", num_states, " and use_start_and_end_states is ",
        use_start_end_states));
  }
  return absl::OkStatus();
}

}  // namespace

template <typename Tin, typename Tsplits>
class ConstrainedSequence : public OpKernel {
 public:
  explicit ConstrainedSequence(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_viterbi", &use_viterbi_));
    OP_REQUIRES_OK(context, context->GetAttr("use_log_space", &use_log_space_));
    OP_REQUIRES_OK(context, context->GetAttr("use_start_and_end_states",
                                             &use_start_end_states_));
  }

  void Compute(OpKernelContext *context) override {
    const auto &score_tensor = context->input(0);
    OP_REQUIRES(context,
                (score_tensor.shape().dims() == 2) ||
                    (score_tensor.shape().dims() == 3),
                InvalidArgument("The score tensor must be of rank 2 or 3."));
    const auto &lengths_tensor = context->input(1);

    ScoreAccessor scores(score_tensor, lengths_tensor);

    // The scores tensor should be [batch, step, scores].
    const int batch_size = scores.batch_size();
    const int num_steps = scores.num_steps();
    const int num_scores = scores.num_scores();

    OP_REQUIRES(context, lengths_tensor.NumElements() == batch_size,
                InvalidArgument(tensorflow::strings::StrCat(
                    "There should be exactly one length for every batch "
                    "element. Found ",
                    lengths_tensor.NumElements(),
                    " length elements for a batch size of ", batch_size)));

    VLOG(2) << "batch: " << batch_size;
    VLOG(2) << "steps: " << num_steps;
    VLOG(2) << "score: " << num_scores;

    // Make sure there's enough data to advance every sequence.
    int max_length = 0;
    int total_length = 0;
    for (int i = 0; i < batch_size; ++i) {
      int64 length = scores.GetLength(i);
      total_length += length;
      if (length > max_length) {
        max_length = length;
      }
    }

    OP_REQUIRES(
        context, num_steps >= max_length,
        InvalidArgument(
            "The scores tensor is too short for the longest sequence length."));

    // Validate the constraint tensors.
    const auto &allowed_transitions_tensor = context->input(2);
    bool has_allowed_transitions =
        allowed_transitions_tensor.NumElements() != 0;
    VLOG(4) << allowed_transitions_tensor.NumElements();
    if (has_allowed_transitions) {
      OP_REQUIRES_OK(context,
                     ValidateConstraintTensor(allowed_transitions_tensor,
                                              num_scores, use_start_end_states_,
                                              "allowed_transitions"));
    }

    const auto &transition_weights_tensor = context->input(3);

    VLOG(4) << transition_weights_tensor.NumElements();
    bool has_transition_weights = transition_weights_tensor.NumElements() != 0;
    if (has_transition_weights) {
      OP_REQUIRES_OK(context, ValidateConstraintTensor(
                                  transition_weights_tensor, num_scores,
                                  use_start_end_states_, "transition_weights"));

      // If we have transition weights in exp-space, all values must be non-
      // negative.
      if (!use_log_space_) {
        for (int i = 0; i < transition_weights_tensor.NumElements(); ++i) {
          OP_REQUIRES(context, transition_weights_tensor.flat<float>()(i) >= 0,
                      InvalidArgument("The transition weights tensor must not "
                                      "contain negative values."));
        }
      }
    }

    const tensorflow::Tensor empty_float(DT_FLOAT, TensorShape({0, 0}));
    const tensorflow::Tensor empty_bool(DT_BOOL, TensorShape({0, 0}));

    const auto &transition_weights =
        has_transition_weights ? transition_weights_tensor.matrix<float>()
                               : empty_float.matrix<float>();

    const auto &allowed_transitions =
        has_allowed_transitions ? allowed_transitions_tensor.matrix<bool>()
                                : empty_bool.matrix<bool>();

    Tensor *output;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({total_length}), &output));
    int32 *output_data = output->flat<int32>().data();

    Tensor *offsets;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({batch_size + 1}), &offsets));
    Tsplits *offset_data = offsets->flat<Tsplits>().data();
    offset_data[0] = 0;

    for (int batch = 0; batch < batch_size; ++batch) {
      int step_offset = offset_data[batch];
      int64 num_steps = scores.GetLength(batch);
      offset_data[batch + 1] = step_offset + num_steps;
      if (use_viterbi_) {
        DoViterbiAnalysis(transition_weights, allowed_transitions, batch,
                          scores, &output_data[step_offset]);
      } else {
        DoGreedyAnalysis(transition_weights, allowed_transitions, batch, scores,
                         &output_data[step_offset]);
      }
    }
  }

 private:
  // Perform Viterbi analysis on a single batch item.
  void DoViterbiAnalysis(
      const tensorflow::TTypes<const float>::Matrix &transition_weights,
      const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
      const int batch, const ScoreAccessor &scores, int32 *output_data) {
    ViterbiAnalysis(scores, transition_weights, allowed_transitions, batch,
                    use_log_space_, use_start_end_states_, output_data);
  }

  // Perform a greedy analysis on a single batch item.
  void DoGreedyAnalysis(
      const tensorflow::TTypes<const float>::Matrix &transition_weights,
      const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
      int batch, const ScoreAccessor &scores, int32 *output_data) {
    GreedyAnalysis(scores, transition_weights, allowed_transitions, batch,
                   use_log_space_, use_start_end_states_, output_data);
  }

  // True if this op should perform calculations in log-space (using addition).
  // If false, will perform calculations in normalized exp-space (using
  // multiplication).
  bool use_log_space_;

  // True if this op should calculate scores using the Viterbi algorithm. If
  // false, will use a greedy algorithm.
  bool use_viterbi_;

  // True if this op should calculate sequences based on an implicit start
  // and end state.
  bool use_start_end_states_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConstrainedSequence);
};

#define REGISTER_KERNELS(Tin)                                    \
  REGISTER_KERNEL_BUILDER(Name("ConstrainedSequence")            \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<Tin>("Tin")        \
                              .TypeConstraint<int32>("Tsplits"), \
                          ConstrainedSequence<Tin, int32>);      \
  REGISTER_KERNEL_BUILDER(Name("ConstrainedSequence")            \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<Tin>("Tin")        \
                              .TypeConstraint<int64>("Tsplits"), \
                          ConstrainedSequence<Tin, int64>)

REGISTER_KERNELS(int32);
REGISTER_KERNELS(int64);

#undef REGISTER_KERNELS

}  // namespace tensorflow

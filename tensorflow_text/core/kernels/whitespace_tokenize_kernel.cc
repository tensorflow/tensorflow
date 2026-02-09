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

#include <string.h>

#include <vector>

#include "icu4c/source/common/unicode/uchar.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace text {

template <typename SPLITS_TYPE>
class WhitespaceTokenizeWithOffsetsOp : public OpKernel {
 public:
  explicit WhitespaceTokenizeWithOffsetsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  /**
   * Breaks a series of codepoints into individual groups based on the script
   * code.
   *
   * We gain a dimension while tokenizing since a series of integer codepoints
   * is tokenized into different codepoint groups.
   *
   * This accepts two input tensors: a rank 1 tensor of codepoint values and
   * a single rank 1 tensor of splits which determine where each string begins
   * and ends from the provided codepoints.
   */
  void Compute(OpKernelContext* context) override {
    // Get inputs
    const Tensor& input_values_tensor = context->input(0);
    const auto input_values_flat = input_values_tensor.flat<int32>();
    const Tensor& input_splits_tensor = context->input(1);
    const auto input_splits_flat = input_splits_tensor.flat<SPLITS_TYPE>();

    // Since we limit to a 2-D input (flat_values of rank 1 and a single splits
    // tensor), our output dimension will always be 3-D (flat_values of rank 1
    // with two splits - inner for the tokenized values and the outer for those
    // grouped by the original strings).
    // A few things to note:
    // 1) The values and inner splits of the tokenized strings have an unknown
    // length, as well as the offsets, so we allocate them at the end.
    // 2) The outer splits of the tokenized strings matches that of the offset
    // splits. Thus, we will only return one set and use it for all of them.
    // 3) The outer splits shape will match the original input_splits.
    Tensor* output_outer_splits_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output_outer_splits",
                                            input_splits_tensor.shape(),
                                            &output_outer_splits_tensor));
    auto output_outer_splits_flat =
        output_outer_splits_tensor->flat<SPLITS_TYPE>();

    std::vector<int32> output_values;
    std::vector<SPLITS_TYPE> output_values_inner_splits;
    std::vector<int64> output_offset_starts;
    std::vector<int64> output_offset_limits;

    // Loop over the codepoints (a split at a time) and create splits of tokens.
    for (int splits_idx = 0; splits_idx < input_splits_flat.size() - 1;
         splits_idx++) {
      output_outer_splits_flat(splits_idx) = output_offset_starts.size();
      bool token_has_start_set = false;
      int32 curr_skipped_spaces = 0;  // Used when computing the end of a token
      const int curr_word_start_idx = input_splits_flat(splits_idx);
      for (int values_idx = curr_word_start_idx;
           values_idx < input_splits_flat(splits_idx + 1); values_idx++) {
        // End current token if we find whitespace
        if (u_isUWhiteSpace(input_values_flat(values_idx))) {
          if (token_has_start_set) {
            output_offset_limits.push_back(values_idx - curr_word_start_idx -
                                           curr_skipped_spaces);
          }
          token_has_start_set = false;
          ++curr_skipped_spaces;
        } else {
          // Non whitespace. Start a new token if needed, and append the
          // codepoint to our current token.
          if (!token_has_start_set) {
            // Set token start offset relative to current string.
            output_offset_starts.push_back(values_idx - curr_word_start_idx);
            // Set split to indicate start of a new token.
            output_values_inner_splits.push_back(output_values.size());
            token_has_start_set = true;
          }
          output_values.push_back(input_values_flat(values_idx));
          curr_skipped_spaces = 0;
        }
      }
      // Looping through the codepoints for current tokens complete. Now set the
      // last limit of out last token (if we found a start earlier).
      if (token_has_start_set) {
        output_offset_limits.push_back(input_splits_flat(splits_idx + 1) -
                                       curr_word_start_idx -
                                       curr_skipped_spaces);
      }
    }
    // Now set the closing value of our splits.
    output_outer_splits_flat(input_splits_flat.size() - 1) =
        output_offset_starts.size();
    output_values_inner_splits.push_back(output_values.size());

// Allocate output & fill output tensors.
#define DECLARE_ALLOCATE_AND_FILL_OUTPUT_TENSOR(name, dtype)                 \
  int64 name##_size = name.size();                                           \
  Tensor* name##_tensor = nullptr;                                           \
  OP_REQUIRES_OK(context,                                                    \
                 context->allocate_output(#name, TensorShape({name##_size}), \
                                          &name##_tensor));                  \
  auto name##_data = name##_tensor->flat<dtype>().data();                    \
  memcpy(name##_data, name.data(), name##_size * sizeof(dtype));

    DECLARE_ALLOCATE_AND_FILL_OUTPUT_TENSOR(output_values, int32);
    DECLARE_ALLOCATE_AND_FILL_OUTPUT_TENSOR(output_values_inner_splits,
                                            SPLITS_TYPE);
    DECLARE_ALLOCATE_AND_FILL_OUTPUT_TENSOR(output_offset_starts, int64);
    DECLARE_ALLOCATE_AND_FILL_OUTPUT_TENSOR(output_offset_limits, int64);

#undef DECLARE_ALLOCATE_AND_FILL_OUTPUT_TENSOR
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WhitespaceTokenizeWithOffsetsOp);
};

REGISTER_KERNEL_BUILDER(Name("WhitespaceTokenizeWithOffsets")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("Tsplits"),
                        WhitespaceTokenizeWithOffsetsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("WhitespaceTokenizeWithOffsets")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("Tsplits"),
                        WhitespaceTokenizeWithOffsetsOp<int64>);

}  // namespace text
}  // namespace tensorflow

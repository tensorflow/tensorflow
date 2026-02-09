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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow_text/core/kernels/sentencepiece/optimized_decoder.h"
#include "tensorflow_text/core/kernels/sentencepiece/sentencepiece_detokenizer.h"

namespace tensorflow {
namespace text {

template <typename Tsplits>
class TFSentencepieceDetokenizerOp : public tensorflow::OpKernel {
 public:
  explicit TFSentencepieceDetokenizerOp(tensorflow::OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(tensorflow::OpKernelContext* ctx) override {
    const auto& model_tensor = ctx->input(kSPModelIndex);
    const auto& input_values_tensor = ctx->input(kInputIndex);
    const auto input_values_flat =
        input_values_tensor.flat<tensorflow::int32>();
    const auto& input_splits_tensor = ctx->input(kInputSplits);
    const auto input_splits_flat = input_splits_tensor.flat<Tsplits>();
    const int num_of_sentences = input_splits_flat.size() - 1;
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {num_of_sentences}, &output_tensor));
    auto output_flat = output_tensor->flat<tensorflow::tstring>();
    std::vector<int> codes_for_split;
    int input_offset = 0;
    for (int i = 0; i < num_of_sentences; i++) {
      // Create a vector of int32 from input according to spans.
      const int split_size = input_splits_flat(i + 1) - input_splits_flat(i);
      codes_for_split.clear();
      codes_for_split.reserve(split_size);
      for (int j = 0; j < split_size; ++j) {
        codes_for_split.push_back(input_values_flat(input_offset++));
      }
      const auto res = sentencepiece::DecodeString(
          codes_for_split, model_tensor.data());
      OP_REQUIRES(ctx, res.type == sentencepiece::DecoderResultType::SUCCESS,
                  absl::Status(static_cast<absl::StatusCode>(
                                   absl::StatusCode::kInternal),
                               "Sentencepiece conversion failed"));
      output_flat(i) = res.decoded;
    }
  }
};
}  // namespace text
}  // namespace tensorflow

REGISTER_KERNEL_BUILDER(
    Name("TFText>FastSentencepieceDetokenize")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<tensorflow::int32>("Tsplits"),
    tensorflow::text::TFSentencepieceDetokenizerOp<tensorflow::int32>);
REGISTER_KERNEL_BUILDER(
    Name("TFText>FastSentencepieceDetokenize")
        .Device(tensorflow::DEVICE_CPU)
        .TypeConstraint<tensorflow::int64>("Tsplits"),
    tensorflow::text::TFSentencepieceDetokenizerOp<tensorflow::int64>);

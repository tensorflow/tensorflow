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

#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow_text/core/kernels/sentencepiece/optimized_encoder.h"
#include "tensorflow_text/core/kernels/sentencepiece/sentencepiece_tokenizer.h"

namespace tensorflow {
namespace text{

class TFSentencepieceOp : public tensorflow::OpKernel {
 public:
  explicit TFSentencepieceOp(tensorflow::OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(tensorflow::OpKernelContext* ctx) override {
    const auto& model_tensor = ctx->input(kSPModelIndex);
    const auto& input_values_tensor = ctx->input(kInputIndex);
    const auto input_values_flat =
        input_values_tensor.flat<tensorflow::tstring>();
    const int64_t num_of_input_values = input_values_flat.size();

    const auto& add_bos_tensor = ctx->input(kAddBOSInput);
    const bool add_bos = add_bos_tensor.scalar<bool>()();
    const auto& add_eos_tensor = ctx->input(kAddEOSInput);
    const bool add_eos = add_eos_tensor.scalar<bool>()();
    const auto& reverse_tensor = ctx->input(kReverseInput);
    const bool reverse = reverse_tensor.scalar<bool>()();

    std::vector<int32> encoded;
    std::vector<int32> splits;
    for (int i = 0; i < num_of_input_values; ++i) {
      const auto res = sentencepiece::EncodeString(
          input_values_flat(i), model_tensor.data(), add_bos, add_eos, reverse);
      OP_REQUIRES(ctx, res.type == sentencepiece::EncoderResultType::SUCCESS,
                  absl::Status(static_cast<absl::StatusCode>(
                                   absl::StatusCode::kInternal),
                               "Sentencepiece conversion failed"));
      std::copy(res.codes.begin(), res.codes.end(),
                std::back_inserter(encoded));
      splits.emplace_back(encoded.size());
    }
    tensorflow::Tensor* output_values_tensor = nullptr;
    tensorflow::Tensor* output_splits_tensor = nullptr;
    OP_REQUIRES(ctx, encoded.size() < std::numeric_limits<int32_t>::max(),
                errors::InvalidArgument(
                    "Encoded input must contain less than 2^31 characters."));
    OP_REQUIRES(
        ctx, splits.size() + 1 < std::numeric_limits<int32_t>::max(),
        errors::InvalidArgument("Splits tensor is limited to 2^31-1 values."));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {static_cast<int32_t>(encoded.size())},
                                  &output_values_tensor));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, {static_cast<int32_t>(splits.size()) + 1},
                                  &output_splits_tensor));

    auto values_tensor_flat = output_values_tensor->vec<int32>();
    auto splits_tensor_flat = output_splits_tensor->vec<int32>();
    for (int32_t i = 0; i < encoded.size(); ++i) {
      values_tensor_flat(i) = encoded[i];
    }
    splits_tensor_flat(0) = 0;
    for (int32_t i = 0; i < splits.size(); ++i) {
      splits_tensor_flat(i + 1) = splits[i];
    }
  }
};

}  // namespace text
}  // namespace tensorflow
REGISTER_KERNEL_BUILDER(
    Name("TFText>FastSentencepieceTokenize").Device(tensorflow::DEVICE_CPU),
    tensorflow::text::TFSentencepieceOp);

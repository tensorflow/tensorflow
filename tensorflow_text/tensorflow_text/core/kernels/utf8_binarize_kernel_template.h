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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_UTF8_BINARIZE_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_UTF8_BINARIZE_KERNEL_TEMPLATE_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/utf8_binarize.h"

namespace tensorflow {
namespace text {

template <tflite::shim::Runtime Rt>
class Utf8BinarizeOp : public tflite::shim::OpKernelShim<Utf8BinarizeOp, Rt> {
 private:
  enum Inputs { kInputTokens = 0 };
  enum Outputs { kOutputBinarizations = 0 };

  using typename tflite::shim::OpKernelShim<Utf8BinarizeOp, Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<Utf8BinarizeOp, Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<Utf8BinarizeOp,
                                            Rt>::ShapeInferenceContext;

 public:
  Utf8BinarizeOp() = default;
  static constexpr char kOpName[] = "TFText>Utf8Binarize";
  static constexpr char kDoc[] = R"doc(
      Decode a UTF-8 string into Unicode code points
      and return their bitwise little-endian representations
      (see the [RetVec paper](https://arxiv.org/abs/2302.09207)).
      )doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs();

  // Inputs declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Outputs declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context);

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);

 private:
  inline static constexpr absl::string_view kMaxCharsAttr = "word_length";
  inline static constexpr absl::string_view kBitsPerCharAttr = "bits_per_char";
  inline static constexpr absl::string_view kReplacementCharAttr =
      "replacement_char";

  int64_t word_length_;
  int64_t bits_per_char_;
  int64_t replacement_char_;
};

template <tflite::shim::Runtime Rt>
std::vector<std::string> Utf8BinarizeOp<Rt>::Attrs() {
  return {absl::StrCat(kMaxCharsAttr, ": int"),
          absl::StrCat(kBitsPerCharAttr, ": int"),
          absl::StrCat(kReplacementCharAttr, ": int")};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> Utf8BinarizeOp<Rt>::Inputs() {
  return {"input_tokens: string"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> Utf8BinarizeOp<Rt>::Outputs() {
  return {"output_binarizations: float"};
}

template <tflite::shim::Runtime Rt>
absl::Status Utf8BinarizeOp<Rt>::Init(InitContext* context) {
  // Attrs
  SH_RETURN_IF_ERROR(
      context->GetAttr(std::string(kMaxCharsAttr), &word_length_));
  SH_RETURN_IF_ERROR(
      context->GetAttr(std::string(kBitsPerCharAttr), &bits_per_char_));
  SH_RETURN_IF_ERROR(
      context->GetAttr(std::string(kReplacementCharAttr), &replacement_char_));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status Utf8BinarizeOp<Rt>::ShapeInference(ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto input_tokens_shape_status = c->GetInputShape(kInputTokens);
  if (!input_tokens_shape_status.ok()) {
    return input_tokens_shape_status.status();
  }
  const Shape& input_tokens_shape = *input_tokens_shape_status;

  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  if (!input_tokens_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", input_tokens_shape.ToString()));
  }

  int64_t word_length;
  SH_RETURN_IF_ERROR(
      c->GetAttr(std::string(kMaxCharsAttr), &word_length));
  int64_t bits_per_char;
  SH_RETURN_IF_ERROR(c->GetAttr(std::string(kBitsPerCharAttr), &bits_per_char));

  const int num_tokens = input_tokens_shape.Dim(0);
  const int bits_per_token = word_length * bits_per_char;
  const Shape output_shape{num_tokens, bits_per_token};
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputBinarizations, output_shape));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status Utf8BinarizeOp<Rt>::Invoke(InvokeContext* context) {
  // Attrs
  const int word_length = word_length_;
  const int bits_per_char = bits_per_char_;
  const int replacement_char = replacement_char_;
  const int bits_per_token = word_length * bits_per_char;

  // Inputs
  const auto tokens_statusor = context->GetInput(kInputTokens);
  if (!tokens_statusor.ok()) {
    return tokens_statusor.status();
  }
  const auto tokens = (*tokens_statusor)->template As<tensorflow::tstring, 1>();
  const int num_tokens = tokens.Dim(0);

  // Outputs
  auto binarizations_statusor =
      context->GetOutput(kOutputBinarizations, {num_tokens, bits_per_token});
  if (!binarizations_statusor.ok()) {
    return binarizations_statusor.status();
  }
  auto binarizations = (*binarizations_statusor)->template As<float, 2>();

  // Iterate through all the token strings and binarize them.
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    float* row_start = &binarizations(token_idx, 0);
    absl::Span<float> output_binarization(row_start, bits_per_token);
    Utf8Binarize(tokens(token_idx),
                 /*word_length=*/word_length,
                 /*bits_per_char=*/bits_per_char,
                 /*replacement=*/replacement_char,
                 /*result=*/output_binarization);
  }

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_UTF8_BINARIZE_KERNEL_TEMPLATE_H_

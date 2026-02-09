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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_V2_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_V2_KERNEL_TEMPLATE_H_


#include <iostream>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/sentence_fragmenter_v2.h"

namespace tensorflow {
namespace text {

template <tflite::shim::Runtime Rt>
class SentenceFragmenterV2Op
    : public tflite::shim::OpKernelShim<SentenceFragmenterV2Op, Rt> {
 private:
  enum Inputs {
    kInputValues = 0
  };
  enum Outputs {
    kFragmentStart = 0,
    kFragmentEnd,
    kFragmentProperties,
    kTerminalPuncToken,
    kOutputRowLengths
  };

  using typename tflite::shim::OpKernelShim<SentenceFragmenterV2Op,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<SentenceFragmenterV2Op,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<SentenceFragmenterV2Op,
                                            Rt>::ShapeInferenceContext;

 public:
  SentenceFragmenterV2Op() = default;
  static constexpr char kOpName[] = "SentenceFragmentsV2";
  static constexpr char kDoc[] = R"doc(
      Splits a string into sentence fragments
      )doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() { return {}; }

  // Inputs declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Outputs declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context) { return absl::OkStatus(); }

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

template <tflite::shim::Runtime Rt>
std::vector<std::string> SentenceFragmenterV2Op<Rt>::Inputs() {
  return {"doc: string"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> SentenceFragmenterV2Op<Rt>::Outputs() {
  return {"fragment_start: int64", "fragment_end: int64",
      "fragment_properties: int64", "terminal_punc_token: int64",
      "output_row_lengths: int64"};
}

template <tflite::shim::Runtime Rt>
absl::Status SentenceFragmenterV2Op<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto rank_1_shape = Shape({Shape::kUnknownDim});

  SH_ASSIGN_OR_RETURN(const Shape& input_values_shape,
                      c->GetInputShape(kInputValues));
  if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", input_values_shape.ToString()));
  }

  SH_RETURN_IF_ERROR(c->SetOutputShape(kFragmentStart, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kFragmentEnd, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kFragmentProperties, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kTerminalPuncToken, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputRowLengths, rank_1_shape));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status SentenceFragmenterV2Op<Rt>::Invoke(InvokeContext* context) {
  // Inputs
  SH_ASSIGN_OR_RETURN(const auto input_values, context->GetInput(kInputValues));
  const auto document = input_values->template As<tensorflow::tstring, 1>();

  // Outputs
  std::vector<int64> fragment_start;
  std::vector<int64> fragment_end;
  std::vector<int64> fragment_properties;
  std::vector<int64> terminal_punc_token;
  std::vector<int64> output_row_lengths;

  // Iterate through all the documents and find fragments.
  for (int i = 0; i < document.Dim(0); ++i) {
    // Find fragments.
    SentenceFragmenterV2 fragmenter(document(i));
    std::vector<SentenceFragment> frags;

    SH_RETURN_IF_ERROR(fragmenter.FindFragments(&frags));

    for (const auto& f : frags) {
      fragment_start.push_back(f.start);
      fragment_end.push_back(f.limit);
      fragment_properties.push_back(f.properties);
      terminal_punc_token.push_back(f.terminal_punc_token);
    }
    output_row_lengths.push_back(frags.size());
  }

  // Allocate output & fill output tensors.
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int64_t, int64_t>(
      fragment_start, kFragmentStart, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int64_t, int64_t>(
      fragment_end, kFragmentEnd, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int64_t, int64_t>(
      fragment_properties, kFragmentProperties, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int64_t, int64_t>(
      terminal_punc_token, kTerminalPuncToken, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int64_t, int64_t>(
      output_row_lengths, kOutputRowLengths, context));

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_FRAGMENTER_V2_KERNEL_TEMPLATE_H_

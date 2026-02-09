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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_KERNEL_TEMPLATE_H_

#include <iostream>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer.h"

namespace tensorflow {
namespace text {

template <tflite::shim::Runtime Rt>
class WhitespaceTokenizeWithOffsetsV2Op
    : public tflite::shim::OpKernelShim<WhitespaceTokenizeWithOffsetsV2Op, Rt> {
 private:
  enum Inputs {
    kInputValues = 0,
    kInputConfig
  };
  enum Outputs {
    kOutputTokens = 0,
    kOutputRowSplits,
    kOutputStartOffsets,
    kOutputEndOffsets
  };

  using typename tflite::shim::OpKernelShim<WhitespaceTokenizeWithOffsetsV2Op,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<WhitespaceTokenizeWithOffsetsV2Op,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<WhitespaceTokenizeWithOffsetsV2Op,
                                            Rt>::ShapeInferenceContext;

 public:
  WhitespaceTokenizeWithOffsetsV2Op() = default;
  static constexpr char kOpName[] = "TFText>WhitespaceTokenizeWithOffsetsV2";
  static constexpr char kDoc[] = R"doc(
    Splits a string into tokens based off of Unicode whitespaces. It also returns
    the relative byte offsets for each token.

    ### Example:

    ```python
    >>> splitter = WhitespaceTokenizer()
    >>> tokens, starts, ends = splitter.tokenize_with_offsets("a bb ccc")
    >>> print(tokens.numpy(), starts.numpy(), ends.numpy())
    [b'a' b'bb' b'ccc'] [0 2 5] [1 4 8]
    ```

    Args:
      input_values: 1D Tensor of strings to tokenize.
      input_config: A string representing a WhitespaceTokenizerConfig.

    Returns:
      * output_tokens: 1D tensor containing the tokens for all input strings.
        A 2D RaggedTensor can be constructed from this and output_row_splits.
      * output_row_splits: 1D int tensor with the row splits that allow us to
        build RaggedTensors from output_tokens, output_start_offsets, and
        output_end_offsets.
      * output_start_offsets: 1D tensor containing the inclusive start byte offset
        for each token in all input strings.  Corresponds 1:1 with output_tokens.
        A 2D RaggedTensor can be constructed from this and output_row_splits.
      * output_end_offsets: 1D tensor containing the exclusive end byte offset for
        each token in all input strings.  Corresponds 1:1 with output_tokens.
        A 2D RaggedTensor can be constructed from this and output_row_splits.
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
std::vector<std::string> WhitespaceTokenizeWithOffsetsV2Op<Rt>::Inputs() {
  return {"input_values: string", "input_config: string"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> WhitespaceTokenizeWithOffsetsV2Op<Rt>::Outputs() {
  return {"output_tokens: string", "output_row_splits: int64",
          "output_start_offsets: int32", "output_end_offsets: int32"};
}

template <tflite::shim::Runtime Rt>
absl::Status WhitespaceTokenizeWithOffsetsV2Op<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto input_values_shape_status = c->GetInputShape(kInputValues);
  if (!input_values_shape_status.ok()) {
    return input_values_shape_status.status();
  }
  const Shape& input_values_shape = *input_values_shape_status;

  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputTokens, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputStartOffsets, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputEndOffsets, rank_1_shape));
  const int num_splits = Shape::AddDims(1, input_values_shape.Dim(0));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputRowSplits, Shape({num_splits})));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
    absl::Status WhitespaceTokenizeWithOffsetsV2Op<Rt>
        ::Invoke(InvokeContext* context) {
  // Inputs
  const auto values_statusor = context->GetInput(kInputValues);
  if (!values_statusor.ok()) {
    return values_statusor.status();
  }
  const auto values = (*values_statusor)->template As<tensorflow::tstring, 1>();

  const auto cfg_statusor = context->GetInput(kInputConfig);
  if (!cfg_statusor.ok()) {
    return cfg_statusor.status();
  }
  const absl::string_view config =
      (*cfg_statusor)->template AsScalar<tensorflow::tstring>();
  WhitespaceTokenizer tokenizer(config);

  // Outputs
  std::vector<std::string> tokens;
  std::vector<int64_t> row_splits;
  std::vector<int32_t> start_offsets;
  std::vector<int32_t> end_offsets;

  // Iterate through all the values and wordpiece tokenize them.
  row_splits.push_back(0);
  for (int i = 0; i < values.Dim(0); ++i) {
    // Tokenize into subwords and record the offset locations.
    const int orig_num_tokens = tokens.size();
    tokenizer.Tokenize(values(i), &tokens, &start_offsets, &end_offsets);
    const int delta_num_tokens = tokens.size() - orig_num_tokens;
    // Record the row splits.
    row_splits.push_back(delta_num_tokens + row_splits.back());
  }

  // Allocate output & fill output tensors.
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<std::string,
                                                      tensorflow::tstring>(
      tokens, kOutputTokens, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int64_t, int64_t>(
      row_splits, kOutputRowSplits, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int32_t, int32_t>(
      start_offsets, kOutputStartOffsets, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int32_t, int32_t>(
      end_offsets, kOutputEndOffsets, context));

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_KERNEL_TEMPLATE_H_

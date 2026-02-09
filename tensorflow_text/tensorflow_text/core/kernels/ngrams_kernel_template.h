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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TEXT_CORE_KERNELS_NGRAMS_KERNEL_TEMPLATE_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_NGRAMS_KERNEL_TEMPLATE_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"

namespace tensorflow {
namespace text {

// text.ngrams op kernel. See `kDoc` for more info.
template <tflite::shim::Runtime Rt>
class NgramsStringJoin : public tflite::shim::OpKernelShim<NgramsStringJoin,
                                                           Rt> {
 protected:
  using Shape = tflite::shim::Shape;

 public:
  using typename tflite::shim::OpKernelShim<NgramsStringJoin,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<NgramsStringJoin,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<NgramsStringJoin,
                                            Rt>::ShapeInferenceContext;

  NgramsStringJoin() = default;
  static constexpr char kOpName[] = "TFText>NgramsStringJoin";
  static constexpr char kDoc[] = R"doc(
    Create a tensor of n-grams based on the string input data.

    Args:
      input_values: A string tensor, or a ragged string tensor (a 1D string value
          tensor and one or more 1D int64 row_split tensors).
      row_splits: List of integer tensors representing the splits of the
          input_values
      width:             scalar integer
          The width of the ngram window.
      axis:              scalar integer
          The axis to create ngrams along.  Currently, it must be -1.
      string_separator:  scalar string
          The separator string used to join tokens together.

    Returns:
      output_values: A string tensor that matches the rank of 'data'.  Will be a
          ragged tensor if 'data' is a ragged tensor.
      output_row_splits: Splits of above.
    )doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration
  static std::vector<std::string> Attrs() {
    return {"width: int",
            "axis: int",
            "string_separator: string",
            "RAGGED_RANK: int >= 0",
            "Tsplits: {int64} = DT_INT64"};
  }
  // Input tensors declaration
  static std::vector<std::string> Inputs() {
    return {"input_values: string", "input_row_splits: RAGGED_RANK * Tsplits"};
  }
  // Output tensors declaration
  static std::vector<std::string> Outputs() {
    return {"output_values: string",
            "output_row_splits: RAGGED_RANK * Tsplits"};
  }

  // Initializes the op
  absl::Status Init(InitContext* ctx) {
    int64_t axis;
    SH_RETURN_IF_ERROR(ctx->GetAttr("axis", &axis));
    if (axis != -1) {
      return absl::InternalError(absl::StrCat("axis != -1: ", axis));
    }
    SH_RETURN_IF_ERROR(ctx->GetAttr("width", &width_));
    absl::string_view string_separator;
    SH_RETURN_IF_ERROR(ctx->GetAttr("string_separator", &string_separator));
    string_separator_ = std::string(string_separator);
    return absl::OkStatus();
  }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
    if (ctx->NumOutputs() == 1) {
      // Tensor Output
      SH_ASSIGN_OR_RETURN(const auto input_shape, ctx->GetInputShape(kValues));
      int64_t width;
      SH_RETURN_IF_ERROR(ctx->GetAttr("width", &width));
      SH_RETURN_IF_ERROR(ctx->SetOutputShape(
          kValues, OutputValuesTensorShape(input_shape, width)));
    } else {
      // RaggedTensor Output
      SH_ASSIGN_OR_RETURN(const auto input_shape, ctx->GetInputShape(kValues));
      Shape output_shape(input_shape);
      const int last_dim = output_shape->size() - 1;
      if (last_dim != -1) {
        (*output_shape)[last_dim] = output_shape.kUnknownDim;
      }
      SH_RETURN_IF_ERROR(ctx->SetOutputShape(kValues, output_shape));

      // The row_splits tensors maintain their shape, because only the
      // innermost dimension will change.
      for (int i = kRowSplitsStart; i < ctx->NumOutputs(); ++i) {
        SH_ASSIGN_OR_RETURN(const Shape input_row_splits_shape,
                            ctx->GetInputShape(i));
        if (input_row_splits_shape.Rank() != 1) {
          return absl::InvalidArgumentError(
              absl::StrCat("expected rank == 1 for input index: ", i));
        }
        SH_RETURN_IF_ERROR(ctx->SetOutputShape(i, input_row_splits_shape));
      }
    }
    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* ctx) {
    using Tsplits = int64_t;
    // Storage for the dummy input and output row_splits used in the tensor
    // case.
    std::vector<Tsplits> tensor_input_row_splits;
    std::vector<Tsplits> tensor_output_row_splits;

    const Tsplits* input_row_splits;
    Tsplits* output_row_splits;
    int n_row_splits = 0;

    SH_ASSIGN_OR_RETURN(const auto input_values, ctx->GetInput(kValues));
    const Shape input_values_shape(input_values->Shape());

    // Tensor output
    if (ctx->NumOutputs() == 1) {
      // Generate mock input and output innermost row_splits.
      int64_t total_tokens =
          input_values->template Data<tensorflow::tstring>().size();
      int64_t tokens_per_element =
          input_values_shape->at(input_values_shape->size() - 1);
      tensor_output_row_splits.resize(total_tokens / tokens_per_element + 1);
      for (int64_t i = 0; i <= total_tokens; i += tokens_per_element) {
        tensor_input_row_splits.push_back(i);
      }
      input_row_splits = tensor_input_row_splits.data();
      output_row_splits = tensor_output_row_splits.data();
      n_row_splits = tensor_input_row_splits.size();
    } else {
      // RaggedTensor output
      int index = 0;
      const int num_row_splits = ctx->NumInputs() - kRowSplitsStart;
      // Copy all input splits except for innermost into output splits.
      while (index < num_row_splits - 1) {
        SH_ASSIGN_OR_RETURN(const auto input_tensor_row_splits,
                            ctx->GetInput(kRowSplitsStart + index));
        SH_ASSIGN_OR_RETURN(
            const auto output_tensor_row_splits,
            ctx->GetOutput(kRowSplitsStart + index,
                           Shape(input_tensor_row_splits->Shape())));
        const auto input_buffer =
            input_tensor_row_splits->template Data<Tsplits>();
        const auto output_buffer =
            output_tensor_row_splits->template Data<Tsplits>();
        std::memcpy(output_buffer.data(), input_buffer.data(),
                    input_buffer.size() * sizeof(Tsplits));
        ++index;
      }
      // Set row splits variables to the innermost
      SH_ASSIGN_OR_RETURN(const auto input_tensor_row_splits,
                          ctx->GetInput(kRowSplitsStart + index));
      SH_ASSIGN_OR_RETURN(
          const auto output_tensor_row_splits,
          ctx->GetOutput(kRowSplitsStart + index,
                         Shape(input_tensor_row_splits->Shape())));
      input_row_splits =
          input_tensor_row_splits->template Data<Tsplits>().data();
      output_row_splits =
          output_tensor_row_splits->template Data<Tsplits>().data();
      n_row_splits = input_tensor_row_splits->Shape().at(0);
    }

    const auto input_values_data =
        input_values->template Data<tensorflow::tstring>();

    // Create ngrams by looping through the innermost input splits.
    std::vector<std::string> buffer;
    for (int i = 0; i < n_row_splits - 1; ++i) {
      // Set output splits using current number of created output values.
      output_row_splits[i] = buffer.size();
      std::vector<tensorflow::tstring> tokens;
      for (int j = input_row_splits[i]; j < input_row_splits[i + 1]; ++j) {
        tokens.emplace_back(input_values_data.at(j));
        if (tokens.size() < width_) continue;
        tokens.erase(tokens.begin(), tokens.begin() + tokens.size() - width_);
        buffer.push_back(absl::StrJoin(tokens, string_separator_));
      }
    }
    output_row_splits[n_row_splits - 1] = buffer.size();

    // Set output values from the generated buffer.
    tflite::shim::TensorViewOr output_values_or;
    if (ctx->NumOutputs() == 1) {
      output_values_or = ctx->GetOutput(
          kValues, OutputValuesTensorShape(input_values_shape, width_));
    } else {
      output_values_or =
          ctx->GetOutput(kValues, Shape({static_cast<int>(buffer.size())}));
    }
    if (!output_values_or.ok()) return output_values_or.status();
    auto& output_buffer =
        output_values_or.value()->template Data<tensorflow::tstring>();
    int i = 0;
    for (const auto& v : buffer) output_buffer[i++] = v;
    return absl::OkStatus();
  }

 protected:
  inline static Shape OutputValuesTensorShape(const Shape& input_values_shape,
                                              const int64_t width) {
    // If the input shape is unknown, so is the output shape.
    if (input_values_shape.Rank() == input_values_shape.kUnknownRank)
      return input_values_shape;

    Shape output_shape(input_values_shape);
    const int last_dim = output_shape->size() - 1;
    if (input_values_shape->at(last_dim) == input_values_shape.kUnknownDim)
      return output_shape;
    (*output_shape)[last_dim] =
        std::max(0, output_shape->at(last_dim) - static_cast<int>(width) + 1);
    return output_shape;
  }

  // Both the input and output tensors use the same indices.
  static constexpr int kValues = 0;
  static constexpr int kRowSplitsStart = 1;

  int64_t width_;
  std::string string_separator_;
};

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_NGRAMS_KERNEL_TEMPLATE_H_

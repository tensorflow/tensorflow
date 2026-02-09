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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_SPLITTER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_SPLITTER_KERNEL_TEMPLATE_H_

#include <iostream>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/byte_splitter.h"

namespace tensorflow {
namespace text {

template <tflite::shim::Runtime Rt>
class ByteSplitterWithOffsetsOp
    : public tflite::shim::OpKernelShim<ByteSplitterWithOffsetsOp, Rt> {
 private:
  enum Inputs {
    kInputValues = 0
  };
  enum Outputs {
    kOutputBytes = 0,
    kOutputRowSplits,
    kOutputStartOffsets,
    kOutputEndOffsets
  };

  using typename tflite::shim::OpKernelShim<ByteSplitterWithOffsetsOp,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<ByteSplitterWithOffsetsOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<ByteSplitterWithOffsetsOp,
                                            Rt>::ShapeInferenceContext;

 public:
  ByteSplitterWithOffsetsOp() = default;
  static constexpr char kOpName[] = "TFText>ByteSplitWithOffsets";
  static constexpr char kDoc[] = R"doc(
  Splits a string into bytes
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
std::vector<std::string> ByteSplitterWithOffsetsOp<Rt>::Inputs() {
  return {"input_values: string"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> ByteSplitterWithOffsetsOp<Rt>::Outputs() {
  return {"output_bytes: uint8", "output_row_splits: int64",
          "output_start_offsets: int32", "output_end_offsets: int32"};
}

template <tflite::shim::Runtime Rt>
absl::Status ByteSplitterWithOffsetsOp<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto rank_1_shape = Shape({Shape::kUnknownDim});

  SH_ASSIGN_OR_RETURN(const Shape& input_values_shape,
                      c->GetInputShape(kInputValues));
  if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Input values shape must be rank 1: ",
        input_values_shape.ToString()));
  }

  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputBytes, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputStartOffsets, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputEndOffsets, rank_1_shape));
  const int num_splits = Shape::AddDims(1, input_values_shape.Dim(0));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputRowSplits, Shape({num_splits})));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
    absl::Status ByteSplitterWithOffsetsOp<Rt>
        ::Invoke(InvokeContext* context) {
  // Inputs
  SH_ASSIGN_OR_RETURN(const auto values_view, context->GetInput(kInputValues));
  const auto values = values_view->template As<tensorflow::tstring, 1>();

  ByteSplitter splitter;

  // Outputs
  std::vector<unsigned char> bytes;
  std::vector<int64_t> row_splits;
  std::vector<int32_t> start_offsets;
  std::vector<int32_t> end_offsets;

  // Iterate through all the string values and split them.
  row_splits.push_back(0);
  for (int i = 0; i < values.Dim(0); ++i) {
    // Split into bytes and record the offset locations.
    const int orig_num_bytes = bytes.size();
    splitter.Split(values(i), &bytes, &start_offsets, &end_offsets);
    const int delta_num_bytes = bytes.size() - orig_num_bytes;
    // Record the row splits.
    row_splits.push_back(delta_num_bytes + row_splits.back());
  }

  // Allocate output & fill output tensors.
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<unsigned char, uint8_t>(
      bytes, kOutputBytes, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int64_t, int64_t>(
      row_splits, kOutputRowSplits, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int32_t, int32_t>(
      start_offsets, kOutputStartOffsets, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int32_t, int32_t>(
      end_offsets, kOutputEndOffsets, context));

  return absl::OkStatus();
}


template <tflite::shim::Runtime Rt>
class ByteSplitByOffsetsOp
    : public tflite::shim::OpKernelShim<ByteSplitByOffsetsOp, Rt> {
 private:
  enum Inputs {
    kInputValues = 0,
    kInputStartOffsets,
    kInputEndOffsets,
    kInputRowSplits
  };
  enum Outputs {
    kOutputValues = 0,
    kOutputRowSplits,
  };

  using typename tflite::shim::OpKernelShim<ByteSplitByOffsetsOp,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<ByteSplitByOffsetsOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<ByteSplitByOffsetsOp,
                                            Rt>::ShapeInferenceContext;

 public:
  ByteSplitByOffsetsOp() = default;
  static constexpr char kOpName[] = "TFText>ByteSplitByOffsets";
  static constexpr char kDoc[] = R"doc(
      Splits a string into bytes using the given start and end offsets.
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
std::vector<std::string> ByteSplitByOffsetsOp<Rt>::Inputs() {
  return {"input_values: string", "input_start_offsets: int32",
          "input_end_offsets: int32", "input_row_splits: int64"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> ByteSplitByOffsetsOp<Rt>::Outputs() {
  return {"output_values: string", "output_row_splits: int64"};
}

template <tflite::shim::Runtime Rt>
absl::Status ByteSplitByOffsetsOp<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  // input values shape
  SH_ASSIGN_OR_RETURN(const Shape& input_values_shape,
                      c->GetInputShape(kInputValues));
  if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Input values shape must be rank 1: ",
        input_values_shape.ToString()));
  }
  // input starts shape
  SH_ASSIGN_OR_RETURN(const Shape& input_starts_shape,
                      c->GetInputShape(kInputStartOffsets));
  if (!input_starts_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Input start offsets shape must be rank 1: ",
        input_starts_shape.ToString()));
  }
  // input ends shape
  SH_ASSIGN_OR_RETURN(const Shape& input_ends_shape,
                      c->GetInputShape(kInputEndOffsets));
  if (!input_ends_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Input end offsets shape must be rank 1: ",
        input_ends_shape.ToString()));
  }
  // input row splits shape
  SH_ASSIGN_OR_RETURN(const Shape& input_row_splits_shape,
                      c->GetInputShape(kInputRowSplits));
  if (!input_row_splits_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Input row splits shape must be rank 1: ",
        input_row_splits_shape.ToString()));
  }

  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputValues, input_starts_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputRowSplits,
                                       input_row_splits_shape));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
    absl::Status ByteSplitByOffsetsOp<Rt>
        ::Invoke(InvokeContext* context) {
  // Inputs
  SH_ASSIGN_OR_RETURN(const auto input_values_view,
                      context->GetInput(kInputValues));
  const auto input_values =
      input_values_view->template As<tensorflow::tstring, 1>();
  SH_ASSIGN_OR_RETURN(const auto starts_view,
                      context->GetInput(kInputStartOffsets));
  const auto starts = starts_view->template As<int32_t, 1>();
  SH_ASSIGN_OR_RETURN(const auto ends_view,
                      context->GetInput(kInputEndOffsets));
  const auto ends = ends_view->template As<int32_t, 1>();
  SH_ASSIGN_OR_RETURN(const auto in_splits_view,
                      context->GetInput(kInputRowSplits));
  const auto in_splits = in_splits_view->template As<int64_t, 1>();

  ByteSplitter splitter;

  // Outputs
  std::vector<absl::string_view> output_values;
  std::vector<int32_t> out_splits;

  // Iterate through all the string values and split them.
  out_splits.push_back(0);
  for (int i = 0; i < input_values.Dim(0); ++i) {
    SH_ASSIGN_OR_RETURN(auto batch,
        splitter.SplitByOffsets(
            input_values(i),
            absl::MakeSpan(starts.Ptr() + in_splits(i),
                          in_splits(i+1) - in_splits(i)),
            absl::MakeSpan(ends.Ptr() + in_splits(i),
                          in_splits(i+1) - in_splits(i))));
    output_values.insert(output_values.end(), batch.begin(), batch.end());
    out_splits.push_back(batch.size() + out_splits.back());
  }

  // Allocate output & fill output tensors.
  SH_RETURN_IF_ERROR(
      this->template FillOutputTensor<absl::string_view, tensorflow::tstring>(
          output_values, kOutputValues, context));
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<int32_t, int64_t>(
      out_splits, kOutputRowSplits, context));

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_SPLITTER_KERNEL_TEMPLATE_H_

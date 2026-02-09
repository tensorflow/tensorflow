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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BOISE_OFFSET_CONVERTER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BOISE_OFFSET_CONVERTER_KERNEL_TEMPLATE_H_

#include <cstdint>
#include <iostream>
#include <ostream>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/boise_offset_converter.h"

namespace tensorflow {
namespace text {

template <tflite::shim::Runtime Rt>
class OffsetsToBoiseTagsOp
    : public tflite::shim::OpKernelShim<OffsetsToBoiseTagsOp, Rt> {
 private:
  enum Inputs {
    kInputTokenBeginOffsets = 0,
    kInputTokenEndOffsets,
    kInputSpanBeginOffsets,
    kInputSpanEndOffsets,
    kInputSpanType,
    kInputTokenBeginRowSplits,
    kInputTokenEndRowSplits,
    kInputSpanBeginRowSplits,
    kInputSpanEndRowSplits,
    kInputSpanTypeRowSplits,
    kInputUseStrictBoundaryMode
  };
  enum Outputs { kOutputBoiseTags = 0 };

  using typename tflite::shim::OpKernelShim<OffsetsToBoiseTagsOp,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<OffsetsToBoiseTagsOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<OffsetsToBoiseTagsOp,
                                            Rt>::ShapeInferenceContext;

 public:
  OffsetsToBoiseTagsOp() = default;
  static constexpr char kOpName[] = "TFText>OffsetsToBoiseTags";
  static constexpr char kDoc[] = R"doc(
  Converts token/span begin/end offsets into BOISE tags.
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

////////////////////////// Implementation

template <tflite::shim::Runtime Rt>
std::vector<std::string> OffsetsToBoiseTagsOp<Rt>::Inputs() {
  return {"input_token_begin_offsets: int32",
          "input_token_end_offsets: int32",
          "input_span_begin_offsets: int32",
          "input_span_end_offsets: int32",
          "input_span_type: string",
          "input_token_begin_row_splits: int64",
          "input_token_end_row_splits: int64",
          "input_span_begin_row_splits: int64",
          "input_span_end_row_splits: int64",
          "input_span_type_row_splits: int64",
          "input_use_strict_boundary_mode: bool"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> OffsetsToBoiseTagsOp<Rt>::Outputs() {
  return {"output_boise_tags: string"};
}

template <tflite::shim::Runtime Rt>
absl::Status OffsetsToBoiseTagsOp<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto rank_1_shape = Shape({Shape::kUnknownDim});

  SH_ASSIGN_OR_RETURN(const Shape input_token_begin_shape,
                      c->GetInputShape(kInputTokenBeginOffsets));
  if (!input_token_begin_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_token_begin_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_token_end_shape,
                      c->GetInputShape(kInputTokenEndOffsets));
  if (!input_token_end_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_token_end_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_span_begin_shape,
                      c->GetInputShape(kInputSpanBeginOffsets));
  if (!input_span_begin_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_span_begin_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_span_end_shape,
                      c->GetInputShape(kInputSpanEndOffsets));
  if (!input_span_end_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_span_end_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_span_type_shape,
                      c->GetInputShape(kInputSpanType));
  if (!input_span_type_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_span_type_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_token_begin_rs_shape,
                      c->GetInputShape(kInputTokenBeginRowSplits));
  if (!input_token_begin_rs_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_token_begin_rs_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_token_end_rs_shape,
                      c->GetInputShape(kInputTokenEndRowSplits));
  if (!input_token_end_rs_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_token_end_rs_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_span_begin_rs_shape,
                      c->GetInputShape(kInputSpanBeginRowSplits));
  if (!input_span_begin_rs_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_span_begin_rs_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_span_end_rs_shape,
                      c->GetInputShape(kInputSpanEndRowSplits));
  if (!input_span_end_rs_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_span_end_rs_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_span_type_rs_shape,
                      c->GetInputShape(kInputSpanTypeRowSplits));
  if (!input_span_type_rs_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_span_type_rs_shape.ToString()));
  }

  const int num_offsets = input_token_begin_shape.Dim(0);
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputBoiseTags, Shape({num_offsets})));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status OffsetsToBoiseTagsOp<Rt>::Invoke(InvokeContext* context) {
  // Inputs
  SH_ASSIGN_OR_RETURN(const auto input_token_begin_offsets,
                      context->GetInput(kInputTokenBeginOffsets));
  const auto& input_token_begin_offsets_vec =
      input_token_begin_offsets->template As<int, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_token_end_offsets,
                      context->GetInput(kInputTokenEndOffsets));
  const auto& input_token_end_offsets_vec =
      input_token_end_offsets->template As<int, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_span_begin_offsets,
                      context->GetInput(kInputSpanBeginOffsets));
  const auto& input_span_begin_offsets_vec =
      input_span_begin_offsets->template As<int, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_span_end_offsets,
                      context->GetInput(kInputSpanEndOffsets));
  const auto& input_span_end_offsets_vec =
      input_span_end_offsets->template As<int, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_span_type,
                      context->GetInput(kInputSpanType));
  const auto& input_span_type_vec =
      input_span_type->template As<tensorflow::tstring, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_token_begin_row_splits,
                      context->GetInput(kInputTokenBeginRowSplits));
  const auto& input_token_begin_row_splits_vec =
      input_token_begin_row_splits->template As<int64_t, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_token_end_row_splits,
                      context->GetInput(kInputTokenEndRowSplits));
  const auto& input_token_end_row_splits_vec =
      input_token_end_row_splits->template As<int64_t, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_span_begin_row_splits,
                      context->GetInput(kInputSpanBeginRowSplits));
  const auto& input_span_begin_row_splits_vec =
      input_span_begin_row_splits->template As<int64_t, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_span_end_row_splits,
                      context->GetInput(kInputSpanEndRowSplits));
  const auto& input_span_end_row_splits_vec =
      input_span_end_row_splits->template As<int64_t, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_span_type_row_splits,
                      context->GetInput(kInputSpanTypeRowSplits));
  const auto& input_span_type_row_splits_vec =
      input_span_type_row_splits->template As<int64_t, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_use_strict_boundary_mode,
                      context->GetInput(kInputUseStrictBoundaryMode));
  const bool input_use_strict_boundary_mode_value =
      input_use_strict_boundary_mode->template AsScalar<bool>();

  // Check token begin and end offsets match in size.
  // Check span begin/end offsets, span type match in size.
  if (input_token_begin_offsets_vec.Dim(0) !=
          input_token_end_offsets_vec.Dim(0) ||
      input_span_begin_offsets_vec.Dim(0) !=
          input_span_end_offsets_vec.Dim(0) ||
      input_span_begin_offsets_vec.Dim(0) != input_span_type_vec.Dim(0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Token begin/end offsets must have the same size. Span begin/end "
        "offsets and span type must have the same size.",
        " Token begin offsets shape: ", input_token_begin_offsets_vec.Dim(0),
        " Token end offsets shape: ", input_token_end_offsets_vec.Dim(0),
        " Span begin offsets shape: ", input_span_begin_offsets_vec.Dim(0),
        " Span end offsets shape: ", input_span_end_offsets_vec.Dim(0),
        " Span type shape: ", input_span_type_vec.Dim(0)));
  }

  // Check row splits are the same for token begin, end offsets.
  if (input_token_begin_row_splits_vec.Dim(0) !=
          input_token_end_row_splits_vec.Dim(0) ||
      input_span_begin_row_splits_vec.Dim(0) !=
          input_span_begin_row_splits_vec.Dim(0) ||
      input_span_begin_row_splits_vec.Dim(0) !=
          input_span_end_row_splits_vec.Dim(0) ||
      input_span_begin_row_splits_vec.Dim(0) !=
          input_span_type_row_splits_vec.Dim(0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Row splits must have the same size for token and span. ",
        " Token begin row splits shape: ",
        input_token_begin_row_splits_vec.Dim(0),
        " Token end row splits shape: ", input_token_end_row_splits_vec.Dim(0),
        " Span begin row splits shape: ",
        input_span_begin_row_splits_vec.Dim(0), " Span end row splits shape: ",
        input_span_end_row_splits_vec.Dim(0), " Span type row splits shape: ",
        input_span_type_row_splits_vec.Dim(0)));
  }

  for (int i = 0; i < input_token_begin_row_splits_vec.Dim(0) - 1; ++i) {
    if (input_token_begin_row_splits_vec(i) !=
        input_token_end_row_splits_vec(i)) {
      return absl::InvalidArgumentError(
          "Row splits must be the same for token begin and end offsets.");
    }
  }

  // Check row splits are the same for span begin, end offsets and span type.
  for (int i = 0; i < input_span_begin_row_splits_vec.Dim(0) - 1; ++i) {
    if (input_span_begin_row_splits_vec(i) !=
            input_span_end_row_splits_vec(i) ||
        input_span_begin_row_splits_vec(i) !=
            input_span_type_row_splits_vec(i)) {
      return absl::InvalidArgumentError(
          "Row splits must be the same for span begin, end offsets and span "
          "type.");
    }
  }

  // Outputs
  std::vector<std::string> boise_tags;
  std::vector<int32_t> input_token_begin_offsets_vec_i;
  std::vector<int32_t> input_token_end_offsets_vec_i;
  std::vector<int32_t> input_span_begin_offsets_vec_i;
  std::vector<int32_t> input_span_end_offsets_vec_i;
  std::vector<std::string> input_span_type_vec_i;

  // Iterate through all the input values and split them.
  for (int i = 0; i < input_token_begin_row_splits_vec.Dim(0) - 1; ++i) {
    int token_start_index = input_token_begin_row_splits_vec(i);
    int token_end_index = input_token_begin_row_splits_vec(i + 1);
    int span_start_index = input_span_begin_row_splits_vec(i);
    int span_end_index = input_span_begin_row_splits_vec(i + 1);

    input_token_begin_offsets_vec_i.clear();
    input_token_end_offsets_vec_i.clear();
    input_span_begin_offsets_vec_i.clear();
    input_span_end_offsets_vec_i.clear();
    input_span_type_vec_i.clear();

    for (int j = token_start_index; j < token_end_index; ++j) {
      input_token_begin_offsets_vec_i.push_back(
          input_token_begin_offsets_vec(j));
      input_token_end_offsets_vec_i.push_back(input_token_end_offsets_vec(j));
    }
    for (int j = span_start_index; j < span_end_index; ++j) {
      input_span_begin_offsets_vec_i.push_back(input_span_begin_offsets_vec(j));
      input_span_end_offsets_vec_i.push_back(input_span_end_offsets_vec(j));
      input_span_type_vec_i.push_back(input_span_type_vec(j));
    }

    SH_ASSIGN_OR_RETURN(
        std::vector<std::string> boise_tags_i,
        OffsetsToBoiseTags(
            input_token_begin_offsets_vec_i, input_token_end_offsets_vec_i,
            input_span_begin_offsets_vec_i, input_span_end_offsets_vec_i,
            input_span_type_vec_i, input_use_strict_boundary_mode_value));

    for (int j = 0; j < boise_tags_i.size(); ++j) {
      boise_tags.push_back(boise_tags_i[j]);
    }
  }

  // Allocate output & fill output tensors.
  SH_RETURN_IF_ERROR(this->template FillOutputTensor<std::string,
                                                     tensorflow::tstring>(
      boise_tags, kOutputBoiseTags, context));

  return absl::OkStatus();
}


template <tflite::shim::Runtime Rt>
class BoiseTagsToOffsetsOp
    : public tflite::shim::OpKernelShim<BoiseTagsToOffsetsOp, Rt> {
 private:
  enum Inputs {
    kInputTokenBeginOffsets = 0,
    kInputTokenEndOffsets,
    kInputBoiseTags,
    kInputTokenBeginRowSplits,
    kInputTokenEndRowSplits,
    kInputBoiseTagsRowSplits,
  };
  enum Outputs {
    kOutputSpanBeginOffsets = 0,
    kOutputSpanEndOffsets,
    kOutputSpanType,
    kOutputRowSplits,
  };

  using typename tflite::shim::OpKernelShim<BoiseTagsToOffsetsOp,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<BoiseTagsToOffsetsOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<BoiseTagsToOffsetsOp,
                                            Rt>::ShapeInferenceContext;

 public:
  BoiseTagsToOffsetsOp() = default;
  static constexpr char kOpName[] = "TFText>BoiseTagsToOffsets";
  static constexpr char kDoc[] = R"doc(
  Converts BOISE tags into span begin/end offsets and span type.
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

 protected:
  template <typename BufferType, typename DType>
  inline absl::Status FillOutputTensor(const std::vector<BufferType>& buffer,
                                       int index, InvokeContext* context);
};

////////////////////////// Implementation

template <tflite::shim::Runtime Rt>
std::vector<std::string> BoiseTagsToOffsetsOp<Rt>::Inputs() {
  return {"input_token_begin_offsets: int32",
          "input_token_end_offsets: int32",
          "input_boise_tags: string",
          "input_token_begin_row_splits: int64",
          "input_token_end_row_splits: int64",
          "input_boise_tags_row_splits: int64"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> BoiseTagsToOffsetsOp<Rt>::Outputs() {
  return {"output_span_begin_offsets: int32", "output_span_end_offsets: int32",
          "output_span_type: string", "output_row_splits: int64"};
}

template <tflite::shim::Runtime Rt>
absl::Status BoiseTagsToOffsetsOp<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto rank_1_shape = Shape({Shape::kUnknownDim});

  SH_ASSIGN_OR_RETURN(const Shape input_token_begin_shape,
                      c->GetInputShape(kInputTokenBeginOffsets));
  if (!input_token_begin_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_token_begin_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_token_end_shape,
                      c->GetInputShape(kInputTokenEndOffsets));
  if (!input_token_end_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_token_end_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_boise_tags_shape,
                      c->GetInputShape(kInputBoiseTags));
  if (!input_boise_tags_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_boise_tags_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_token_begin_rs_shape,
                      c->GetInputShape(kInputTokenBeginRowSplits));
  if (!input_token_begin_rs_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_token_begin_rs_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_token_end_rs_shape,
                      c->GetInputShape(kInputTokenEndRowSplits));
  if (!input_token_end_rs_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_token_end_rs_shape.ToString()));
  }

  SH_ASSIGN_OR_RETURN(const Shape input_boise_tags_rs_shape,
                      c->GetInputShape(kInputBoiseTagsRowSplits));
  if (!input_boise_tags_rs_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_boise_tags_rs_shape.ToString()));
  }

  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputSpanBeginOffsets, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputSpanEndOffsets, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputSpanType, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputRowSplits, rank_1_shape));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status BoiseTagsToOffsetsOp<Rt>::Invoke(InvokeContext* context) {
  // Inputs
  SH_ASSIGN_OR_RETURN(const auto input_token_begin_offsets,
                      context->GetInput(kInputTokenBeginOffsets));
  const auto& input_token_begin_offsets_vec =
      input_token_begin_offsets->template As<int, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_token_end_offsets,
                      context->GetInput(kInputTokenEndOffsets));
  const auto& input_token_end_offsets_vec =
      input_token_end_offsets->template As<int, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_boise_tags,
                      context->GetInput(kInputBoiseTags));
  const auto& input_boise_tags_vec =
      input_boise_tags->template As<tensorflow::tstring, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_token_begin_row_splits,
                      context->GetInput(kInputTokenBeginRowSplits));
  const auto& input_token_begin_row_splits_vec =
      input_token_begin_row_splits->template As<int64_t, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_token_end_row_splits,
                      context->GetInput(kInputTokenEndRowSplits));
  const auto& input_token_end_row_splits_vec =
      input_token_end_row_splits->template As<int64_t, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_boise_tags_row_splits,
                      context->GetInput(kInputBoiseTagsRowSplits));
  const auto& input_boise_tags_row_splits_vec =
      input_boise_tags_row_splits->template As<int64_t, 1>();

  // Check token begin and end offsets, and boise tags match in size.
  if (input_token_begin_offsets_vec.Dim(0) !=
          input_token_end_offsets_vec.Dim(0) ||
      input_token_begin_offsets_vec.Dim(0) != input_boise_tags_vec.Dim(0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Token begin/end offsets and boise tags must have the same size. ",
        " Token begin offsets shape: ", input_token_begin_offsets_vec.Dim(0),
        " Token end offsets shape: ", input_token_end_offsets_vec.Dim(0),
        " BOISE tags shape: ", input_boise_tags_vec.Dim(0)));
  }

  // Check row splits are the same for token begin, end offsets and boise tags.
  // First, check dimensions are the same.
  if (input_token_begin_row_splits_vec.Dim(0) !=
          input_token_end_row_splits_vec.Dim(0) ||
      input_token_begin_row_splits_vec.Dim(0) !=
          input_boise_tags_row_splits_vec.Dim(0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Row splits must have the same size for token begin/end offsets and "
        "BOISE tags. ",
        " Token begin row splits shape: ",
        input_token_begin_row_splits_vec.Dim(0),
        " Token end row splits shape: ", input_token_end_row_splits_vec.Dim(0),
        " BOISE tags row splits shape: ",
        input_boise_tags_row_splits_vec.Dim(0)));
  }
  // Second, check values are the same.
  for (int i = 0; i < input_token_begin_row_splits_vec.Dim(0) - 1; ++i) {
    if (input_token_begin_row_splits_vec(i) !=
            input_token_end_row_splits_vec(i) ||
        input_token_begin_row_splits_vec(i) !=
            input_boise_tags_row_splits_vec(i)) {
      return absl::InvalidArgumentError(
          "Row splits must be the same for token begin/end offsets ad BOISE "
          "tags.");
    }
  }

  // Outputs
  std::vector<int32_t> span_begin_offsets;
  std::vector<int32_t> span_end_offsets;
  std::vector<std::string> span_type;
  std::vector<int64_t> row_splits;

  row_splits.push_back(0);

  // Iterate through all the input values and split them.
  std::vector<int32_t> input_token_begin_offsets_vec_i;
  std::vector<int32_t> input_token_end_offsets_vec_i;
  std::vector<std::string> input_boise_tags_vec_i;
  for (int i = 0; i < input_token_begin_row_splits_vec.Dim(0) - 1; ++i) {
    int token_start_index = input_token_begin_row_splits_vec(i);
    int token_end_index = input_token_begin_row_splits_vec(i + 1);

    input_token_begin_offsets_vec_i.clear();
    input_token_end_offsets_vec_i.clear();
    input_boise_tags_vec_i.clear();

    for (int j = token_start_index; j < token_end_index; ++j) {
      input_token_begin_offsets_vec_i.push_back(
          input_token_begin_offsets_vec(j));
      input_token_end_offsets_vec_i.push_back(input_token_end_offsets_vec(j));
      input_boise_tags_vec_i.push_back(input_boise_tags_vec(j));
    }

    auto [span_begin_offsets_i, span_end_offsets_i, span_type_i] =
        BoiseTagsToOffsets(input_token_begin_offsets_vec_i,
                           input_token_end_offsets_vec_i,
                           input_boise_tags_vec_i)
            .value();

    const int num_span_i = span_type_i.size();
    row_splits.push_back(row_splits.back() + num_span_i);

    for (int j = 0; j < span_type_i.size(); ++j) {
      span_type.push_back(span_type_i[j]);
      span_begin_offsets.push_back(span_begin_offsets_i[j]);
      span_end_offsets.push_back(span_end_offsets_i[j]);
    }
  }

  // Allocate output & fill output tensors.
  SH_RETURN_IF_ERROR(FillOutputTensor<int32_t, int32_t>(
      span_begin_offsets, kOutputSpanBeginOffsets, context));
  SH_RETURN_IF_ERROR(FillOutputTensor<int32_t, int32_t>(
      span_end_offsets, kOutputSpanEndOffsets, context));
  SH_RETURN_IF_ERROR(FillOutputTensor<std::string, tensorflow::tstring>(
      span_type, kOutputSpanType, context));
  SH_RETURN_IF_ERROR(FillOutputTensor<int64_t, int64_t>(
      row_splits, kOutputRowSplits, context));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
template <typename BufferType, typename DType>
absl::Status BoiseTagsToOffsetsOp<Rt>::FillOutputTensor(
    const std::vector<BufferType>& buffer, const int index,
    InvokeContext* context) {
  SH_ASSIGN_OR_RETURN(
      const auto tensorview,
      context->GetOutput(
          index, tflite::shim::Shape({static_cast<int>(buffer.size())})));
  auto data = tensorview->template As<DType, 1>();
  // TODO(broken): investigate using memcpy like previous WST
  for (int i = 0; i < buffer.size(); ++i) data(i) = buffer.at(i);
  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BOISE_OFFSET_CONVERTER_KERNEL_TEMPLATE_H_

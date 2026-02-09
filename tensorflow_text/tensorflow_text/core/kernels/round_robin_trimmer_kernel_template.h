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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_KERNEL_TEMPLATE_H_

#include <cstdint>
#include <iostream>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/round_robin_trimmer.h"

namespace tensorflow {
namespace text {

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
class RoundRobinTrimOp
    : public tflite::shim::OpKernelShim<RoundRobinTrimOp, Rt, T, Tsplits> {
 private:
  enum Inputs {
    kMaxSeqLength = 0,
    kInputValues,
    kInputRowSplits
  };
  enum Outputs {
    kOutputValues = 0,
    kOutputRowSplits
  };
  int64_t number_of_segments_;

  using typename tflite::shim::OpKernelShim<RoundRobinTrimOp, Rt, T,
                                            Tsplits>::InitContext;
  using typename tflite::shim::OpKernelShim<RoundRobinTrimOp, Rt, T,
                                            Tsplits>::InvokeContext;
  using typename tflite::shim::OpKernelShim<RoundRobinTrimOp, Rt, T,
                                            Tsplits>::ShapeInferenceContext;

 public:
  RoundRobinTrimOp() = default;
  static constexpr char kOpName[] = "TFText>RoundRobinTrim";
  static constexpr char kDoc[] = R"doc(
      Trims a tensor.
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
  absl::Status Init(InitContext* context) {
    // Attr
    SH_RETURN_IF_ERROR(context->GetAttr("N", &number_of_segments_));
    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
std::vector<std::string> RoundRobinTrimOp<Rt, T, Tsplits>::Attrs() {
  return {"N: int >= 1", "T: type", "Tsplits: {int32, int64}"};
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
std::vector<std::string> RoundRobinTrimOp<Rt, T, Tsplits>::Inputs() {
  return {"max_sequence_length: int32", "input_values: N * T",
          "input_row_splits: N * Tsplits"};
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
std::vector<std::string> RoundRobinTrimOp<Rt, T, Tsplits>::Outputs() {
  return {"values: N * T", "row_splits: N * Tsplits"};
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
absl::Status RoundRobinTrimOp<Rt, T, Tsplits>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  int64_t num_segments;
  SH_RETURN_IF_ERROR(c->GetAttr("N", &num_segments));

  SH_ASSIGN_OR_RETURN(const Shape& max_seq_shape,
                      c->GetInputShape(kMaxSeqLength));
  if (!max_seq_shape.Compatible(Shape({}))) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be a scalar: ", max_seq_shape.ToString()));
  }

  for (int i = 0; i < num_segments; ++i) {
    SH_ASSIGN_OR_RETURN(
        const Shape& values_shape,
        c->GetInputShape(
          (kInputValues - 1) * num_segments + i + 1));
    if (!values_shape.Compatible(rank_1_shape)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Shape must be rank 1: ", values_shape.ToString()));
    }

    SH_ASSIGN_OR_RETURN(
        const Shape& row_splits_shape,
        c->GetInputShape(
            (kInputRowSplits - 1) * num_segments + i + 1));
    if (!row_splits_shape.Compatible(rank_1_shape)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Shape must be rank 1: ", row_splits_shape.ToString()));
    }

    SH_RETURN_IF_ERROR(c->SetOutputShape(
      kOutputRowSplits * num_segments + i, row_splits_shape));
    SH_RETURN_IF_ERROR(c->SetOutputShape(
      kOutputValues * num_segments + i, rank_1_shape));
  }

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
absl::Status RoundRobinTrimOp<Rt, T, Tsplits>::Invoke(InvokeContext* context) {
  // Inputs
  SH_ASSIGN_OR_RETURN(const auto msl, context->GetInput(kMaxSeqLength));
  const int max_sequence_length = msl->template AsScalar<tensorflow::int32>();

  std::vector<absl::Span<T>> list_of_values(number_of_segments_);
  std::vector<absl::Span<Tsplits>> list_of_splits(number_of_segments_);
  for (int i = 0; i < number_of_segments_; ++i) {
    SH_ASSIGN_OR_RETURN(const auto fv, context->GetInput(kInputValues + i));
    list_of_values[i] = fv->template Data<T>();

    int row_split_idx = kInputRowSplits + number_of_segments_ - 1 + i;
    SH_ASSIGN_OR_RETURN(const auto rs, context->GetInput(row_split_idx));
    list_of_splits[i] = rs->template Data<Tsplits>();
  }

  // Compute
  RoundRobinTrimmer<T, Tsplits> trimmer(max_sequence_length);
  auto [trimmed_vals, trimmed_splits] = trimmer.TrimBatch(
      list_of_values, list_of_splits);

  for (int i = 0; i < number_of_segments_; ++i) {
    // Allocate output & fill output tensors.
    SH_RETURN_IF_ERROR(this->template FillOutputTensor<T, T>(
        trimmed_vals[i], (kOutputValues * number_of_segments_) + i, context));
    SH_RETURN_IF_ERROR(
        this->template FillOutputTensor<Tsplits, Tsplits>(trimmed_splits[i],
        (kOutputRowSplits * number_of_segments_) + i, context));
  }

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
class RoundRobinGenerateMasksOp
    : public tflite::shim::OpKernelShim<RoundRobinGenerateMasksOp, Rt, T,
                                        Tsplits> {
 private:
  enum Inputs {
    kMaxSeqLength = 0,
    kInputValues,
    kInputRowSplits
  };
  enum Outputs {
    kOutputMasks = 0
  };
  int64_t number_of_segments_;

  using typename tflite::shim::OpKernelShim<RoundRobinGenerateMasksOp, Rt, T,
                                            Tsplits>::InitContext;
  using typename tflite::shim::OpKernelShim<RoundRobinGenerateMasksOp, Rt, T,
                                            Tsplits>::InvokeContext;
  using typename tflite::shim::OpKernelShim<RoundRobinGenerateMasksOp, Rt, T,
                                            Tsplits>::ShapeInferenceContext;

 public:
  RoundRobinGenerateMasksOp() = default;
  static constexpr char kOpName[] = "TFText>RoundRobinGenerateMasks";
  static constexpr char kDoc[] = R"doc(
      Generates a mask for trimming a tensor.
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
  absl::Status Init(InitContext* context) {
    // Attr
    SH_RETURN_IF_ERROR(context->GetAttr("N", &number_of_segments_));
    return absl::OkStatus();
  }

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
std::vector<std::string> RoundRobinGenerateMasksOp<Rt, T, Tsplits>::Attrs() {
  return {"N: int >= 1", "T: type", "Tsplits: {int32, int64}"};
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
std::vector<std::string> RoundRobinGenerateMasksOp<Rt, T, Tsplits>::Inputs() {
  // TODO(broken): use templated value
  return {"max_sequence_length: int32", "input_values: N * T",
          "input_row_splits: N * Tsplits"};
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
std::vector<std::string> RoundRobinGenerateMasksOp<Rt, T, Tsplits>::Outputs() {
  return {"masks: N * bool"};
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
absl::Status RoundRobinGenerateMasksOp<Rt, T, Tsplits>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  int64_t num_segments;
  SH_RETURN_IF_ERROR(c->GetAttr("N", &num_segments));

  SH_ASSIGN_OR_RETURN(const Shape& max_seq_shape,
                      c->GetInputShape(kMaxSeqLength));
  if (!max_seq_shape.Compatible(Shape({}))) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be a scalar: ", max_seq_shape.ToString()));
  }

  for (int i = 0; i < num_segments; ++i) {
    SH_ASSIGN_OR_RETURN(
        const Shape& values_shape,
        c->GetInputShape(
          (kInputValues - 1) * num_segments + i + 1));
    if (!values_shape.Compatible(rank_1_shape)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Shape must be rank 1: ", values_shape.ToString()));
    }

    SH_ASSIGN_OR_RETURN(
        const Shape& row_splits_shape,
        c->GetInputShape(
            (kInputRowSplits - 1) * num_segments + i + 1));
    if (!row_splits_shape.Compatible(rank_1_shape)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Shape must be rank 1: ", row_splits_shape.ToString()));
    }

    SH_RETURN_IF_ERROR(c->SetOutputShape(
      kOutputMasks * num_segments + i, values_shape));
  }

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt, typename T, typename Tsplits>
absl::Status RoundRobinGenerateMasksOp<Rt, T, Tsplits>::Invoke(
    InvokeContext* context) {
  // Inputs
  SH_ASSIGN_OR_RETURN(const auto msl, context->GetInput(kMaxSeqLength));
  const int max_sequence_length = msl->template AsScalar<tensorflow::int32>();

  std::vector<absl::Span<Tsplits>> list_of_splits(number_of_segments_);
  for (int i = 0; i < number_of_segments_; ++i) {
    int row_split_idx = kInputRowSplits + number_of_segments_ - 1 + i;
    SH_ASSIGN_OR_RETURN(const auto rs, context->GetInput(row_split_idx));
    list_of_splits[i] = rs->template Data<Tsplits>();
  }

  // Compute
  RoundRobinTrimmer<T, Tsplits> trimmer(max_sequence_length);
  std::vector<std::vector<bool>> masks =
      trimmer.GenerateMasksBatch(list_of_splits);

  for (int i = 0; i < number_of_segments_; ++i) {
    // Allocate output & fill output tensors.
    SH_RETURN_IF_ERROR(this->template FillOutputTensor<bool, bool>(masks[i],
        (kOutputMasks * number_of_segments_) + i, context));
  }

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_KERNEL_TEMPLATE_H_

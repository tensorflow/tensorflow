/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/shlo/ops/compare.h"

#include <functional>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

namespace {

template <DataType storage_type, DataType expressed_type, typename F>
void DequantizeCompare(F&& func, const Tensor& lhs, const Tensor& rhs,
                       Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;
  const DimensionSize num_elements = lhs.NumElements();
  const StorageT lhs_zero_point =
      lhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT lhs_scale =
      lhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const StorageT rhs_zero_point =
      rhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT rhs_scale =
      rhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  bool* output_data = output.GetDataAs<DataType::kI1>();
  for (DimensionSize i = 0; i < num_elements;
       ++i, ++lhs_data, ++rhs_data, ++output_data) {
    const ExpressedT dequantized_lhs =
        Dequantize(*lhs_data, lhs_zero_point, lhs_scale);
    const ExpressedT dequantized_rhs =
        Dequantize(*rhs_data, rhs_zero_point, rhs_scale);
    *output_data = func(dequantized_lhs, dequantized_rhs);
  }
}

}  // namespace

CompareOp Create(CompareOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(CompareOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("compare"), lhs, IsBoolTensor, IsIntTensor,
                          IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("compare"), output, IsBoolTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("compare"), lhs, rhs));
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  return absl::OkStatus();
}

// Huge body because of the type dispatch.
// NOLINTNEXTLINE(google-readability-function-size)
absl::Status Evaluate(CompareOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
#define SHLO_REF_COMPARISON_DIRECTION_CASE(DIRECTION, COMPARISON_OP)         \
  case CompareOp::ComparisonDirection::DIRECTION: {                          \
    if (IsBoolTensor(lhs) || IsIntTensor(lhs) || IsFloatTensor(lhs)) {       \
      DISPATCH_BOOL_INT_FLOAT(detail::EvaluateNoQuantization,                \
                              lhs.tensor_element_type(), COMPARISON_OP, lhs, \
                              rhs, output);                                  \
    } else if (IsQuantizedPerTensorTensor(lhs)) {                            \
      DISPATCH_QUANTIZED(                                                    \
          DequantizeCompare,                                                 \
          lhs.quantized_per_tensor_element_type().StorageType(),             \
          lhs.quantized_per_tensor_element_type().ExpressedType(),           \
          COMPARISON_OP, lhs, rhs, output)                                   \
    }                                                                        \
    break;                                                                   \
  }

  switch (op.attributes.comparison_direction) {
    SHLO_REF_COMPARISON_DIRECTION_CASE(kEq, std::equal_to<void>());
    SHLO_REF_COMPARISON_DIRECTION_CASE(kNe, std::not_equal_to<void>());
    SHLO_REF_COMPARISON_DIRECTION_CASE(kGe, std::greater_equal<void>());
    SHLO_REF_COMPARISON_DIRECTION_CASE(kGt, std::greater<void>());
    SHLO_REF_COMPARISON_DIRECTION_CASE(kLe, std::less_equal<void>());
    SHLO_REF_COMPARISON_DIRECTION_CASE(kLt, std::less<void>());
  }

  return absl::FailedPreconditionError(
      "stablehlo.compare: Unsupported tensor type.");
}

}  // namespace shlo_ref

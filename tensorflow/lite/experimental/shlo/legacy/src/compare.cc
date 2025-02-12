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

#include <cstddef>
#include <type_traits>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/dispatch.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

namespace {

template <typename Value>
bool CompareHelper(ComparisonDirection comparison_direction, Value lhs,
                   Value rhs) {
  switch (comparison_direction) {
    case ComparisonDirection::kEQ:
      return (lhs == rhs);
    case ComparisonDirection::kNE:
      return (lhs != rhs);
    case ComparisonDirection::kGE:
      return (lhs >= rhs);
    case ComparisonDirection::kGT:
      return (lhs > rhs);
    case ComparisonDirection::kLE:
      return (lhs <= rhs);
    case ComparisonDirection::kLT:
      return (lhs < rhs);
  }
}

template <typename Value>
absl::Status CheckParameters(const Value& lhs, const Value& rhs,
                             ComparisonDirection comparison_direction,
                             CompareType compare_type, Tensor& result) {
  if (!(lhs.baseline_element_type() == rhs.baseline_element_type())) {
    return absl::InvalidArgumentError(
        "Constraint violation: baseline_element_type(lhs) = "
        "baseline_element_type(rhs)");
  } else if (!(lhs.shape() == rhs.shape() and lhs.shape() == result.shape())) {
    return absl::InvalidArgumentError(
        "Constraint violation: shape(lhs) = shape(rhs) = shape(rhs)");
  } else if (result.element_type() != ElementType::kI1) {
    return absl::InvalidArgumentError("Expected boolean tensor as result");
  } else {
    auto element_type = lhs.element_type();
    if (!((compare_type == CompareType::kSigned and
           IsSignedInteger(element_type)) or
          (compare_type == CompareType::kUnsigned and
           (IsUnsignedInteger(element_type) or IsBoolean(element_type))) or
          ((compare_type == CompareType::kFloat or
            compare_type == CompareType::kTotalOrder) and
           IsFloat(element_type)))) {
      return absl::InvalidArgumentError(
          "Inconsistent compare type vs element type");
    }
  }
  if (compare_type == CompareType::kTotalOrder) {
    return absl::InvalidArgumentError(
        "CompareType::kTotalOrder is unsupported");
  }

  if constexpr (std::is_same_v<Value, QuantizedTensor>) {
    if (!(lhs.is_per_tensor_quantized() and rhs.is_per_tensor_quantized())) {
      return absl::InvalidArgumentError("Expected per-tensor quantization");
    }
  }

  if (lhs.layout().has_strides() || rhs.layout().has_strides() ||
      result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type, typename Value>
absl::Status Compare(const Value& lhs, const Value& rhs,
                     ComparisonDirection comparison_direction,
                     CompareType compare_type, Tensor& result) {
  if (auto check =
          CheckParameters(lhs, rhs, comparison_direction, compare_type, result);
      !check.ok()) {
    return check;
  }

  using S = Storage<storage_type>;
  const size_t n = result.num_elements();
  auto lhs_buffer = lhs.buffer();
  auto rhs_buffer = rhs.buffer();
  auto result_buffer = result.buffer();

  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != lhs.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }

    for (size_t i = 0; i < n; ++i) {
      auto lhs_value = S::Get(lhs_buffer, i);
      auto rhs_value = S::Get(rhs_buffer, i);
      bool result_value =
          CompareHelper(comparison_direction, lhs_value, rhs_value);
      Storage<ElementType::kI1>::Set(result_buffer, i, result_value);
    }

  } else {
    static_assert(std::is_same_v<Value, QuantizedTensor>);

    const QuantizedParameter& lhs_quant_param =
        lhs.type().element_type().parameters(0);
    const QuantizedParameter& rhs_quant_param =
        rhs.type().element_type().parameters(0);

    for (size_t i = 0; i < n; ++i) {
      auto lhs_storage = S::Get(lhs_buffer, i);
      auto lhs_expressed = Dequantize<storage_type, expressed_type>(
          lhs_storage, lhs_quant_param);

      auto rhs_storage = S::Get(rhs_buffer, i);
      auto rhs_expressed = Dequantize<storage_type, expressed_type>(
          rhs_storage, rhs_quant_param);

      bool result_value =
          CompareHelper(comparison_direction, lhs_expressed, rhs_expressed);
      Storage<ElementType::kI1>::Set(result_buffer, i, result_value);
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status Compare(const Tensor& lhs, const Tensor& rhs,
                     ComparisonDirection comparison_direction,
                     CompareType compare_type, Tensor& result) {
  DISPATCH_BOOL_INT_FLOAT(Compare, lhs.element_type(), lhs, rhs,
                          comparison_direction, compare_type, result);
}

absl::Status Compare(const QuantizedTensor& lhs, const QuantizedTensor& rhs,
                     ComparisonDirection comparison_direction,
                     CompareType compare_type, Tensor& result) {
  DISPATCH_QUANTIZED(Compare, lhs.storage_type(), lhs.expressed_type(), lhs,
                     rhs, comparison_direction, compare_type, result);
}

}  // namespace stablehlo

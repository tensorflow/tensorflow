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

#include <algorithm>
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
absl::Status CheckParameters(const Value& min, const Value& operand,
                             const Value& max, Value& result) {
  if (!(min.rank() == 0 or min.shape() == operand.shape())) {
    return absl::InvalidArgumentError(
        "Constraint violation: rank(min) = 0 or shape(min) = shape(operand)");
  } else if (!(max.rank() == 0 or max.shape() == operand.shape())) {
    return absl::InvalidArgumentError(
        "Constraint violation: rank(max) = 0 or shape(max) = shape(operand)");
  } else if (!(min.baseline_element_type() ==
                   operand.baseline_element_type() and
               min.baseline_element_type() == max.baseline_element_type())) {
    return absl::InvalidArgumentError(
        "Constraint violation: baseline_element_type(min) = "
        "baseline_element_type(operand) = baseline_element_type(max)");
  } else if (!(operand.baseline_type() == result.baseline_type())) {
    return absl::InvalidArgumentError(
        "Constraint violation: baseline_type(operand) = baseline_type(result)");
  }

  if constexpr (std::is_same_v<Value, QuantizedTensor>) {
    if (!(min.is_per_tensor_quantized() and max.is_per_tensor_quantized() and
          operand.is_per_tensor_quantized() and
          result.is_per_tensor_quantized())) {
      return absl::InvalidArgumentError("Expected per-tensor quantization");
    }
  }

  if (operand.layout().has_strides() || result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type, typename Value>
absl::Status Clamp(const Value& min, const Value& operand, const Value& max,
                   Value& result) {
  if (auto check = CheckParameters(min, operand, max, result); !check.ok()) {
    return check;
  }

  using S = Storage<storage_type>;

  const bool min_is_tensor = (min.rank() > 0);
  const bool max_is_tensor = (max.rank() > 0);
  const size_t n = result.num_elements();

  auto operand_buffer = operand.buffer();
  auto min_buffer = min.buffer();
  auto max_buffer = max.buffer();
  auto result_buffer = result.buffer();

  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != result.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }

    typename S::Type min_value;
    typename S::Type max_value;
    for (size_t i = 0; i < n; ++i) {
      if (min_is_tensor || (i == 0)) {
        min_value = S::Get(min_buffer, i);
      }
      if (max_is_tensor || (i == 0)) {
        max_value = S::Get(max_buffer, i);
      }
      auto operand_value = S::Get(operand_buffer, i);
      auto result_value =
          std::min(max_value, std::max(min_value, operand_value));
      S::Set(result_buffer, i, result_value);
    }

  } else {
    static_assert(std::is_same_v<Value, QuantizedTensor>);

    if (storage_type != result.storage_type()) {
      return absl::InvalidArgumentError("Unexpected storage type");
    } else if (expressed_type != result.expressed_type()) {
      return absl::InvalidArgumentError("Unexpected expressed type");
    }

    using ET = typename Storage<expressed_type>::Type;

    const QuantizedParameter& min_quant_param =
        min.type().element_type().parameters(0);
    const QuantizedParameter& max_quant_param =
        max.type().element_type().parameters(0);
    const QuantizedParameter& operand_quant_param =
        operand.type().element_type().parameters(0);
    const QuantizedParameter& result_quant_param =
        result.type().element_type().parameters(0);
    ET result_scale_inv = ET(1.0) / static_cast<ET>(result_quant_param.scale);

    ET min_expressed;
    ET max_expressed;
    for (size_t i = 0; i < n; ++i) {
      if (min_is_tensor || (i == 0)) {
        auto min_storage = S::Get(min_buffer, i);
        min_expressed = Dequantize<storage_type, expressed_type>(
            min_storage, min_quant_param);
      }
      if (max_is_tensor || (i == 0)) {
        auto max_storage = S::Get(max_buffer, i);
        max_expressed = Dequantize<storage_type, expressed_type>(
            max_storage, max_quant_param);
      }

      auto operand_storage = S::Get(operand_buffer, i);
      auto result_storage =
          DequantizeOpQuantizePartial<storage_type, expressed_type>(
              operand_storage, operand_quant_param, result_scale_inv,
              result_quant_param.zero_point, [=](auto x) {
                return std::min(max_expressed, std::max(min_expressed, x));
              });
      S::Set(result_buffer, i, result_storage);
    }

    if (auto status = CompleteQuantization<storage_type>(result);
        !status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status Clamp(const Tensor& min, const Tensor& operand, const Tensor& max,
                   Tensor& result) {
  DISPATCH_INT_FLOAT(Clamp, result.element_type(), min, operand, max, result);
}

absl::Status Clamp(const QuantizedTensor& min, const QuantizedTensor& operand,
                   const QuantizedTensor& max, QuantizedTensor& result) {
  DISPATCH_QUANTIZED(Clamp, result.storage_type(), result.expressed_type(), min,
                     operand, max, result);
}

}  // namespace stablehlo

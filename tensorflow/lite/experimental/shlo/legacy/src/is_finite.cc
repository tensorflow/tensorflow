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

#include <cmath>
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
absl::Status CheckParameters(const Value& operand, Tensor& result) {
  if (operand.shape() != result.shape()) {
    return absl::InvalidArgumentError("Inconsistent input/output shapes");
  } else if (result.element_type() != ElementType::kI1) {
    return absl::InvalidArgumentError("Unexpected output tensor element type");
  }

  if constexpr (std::is_same_v<Value, QuantizedTensor>) {
    if (!operand.is_per_tensor_quantized()) {
      return absl::InvalidArgumentError("Expected per-tensor quantization");
    }
  }

  if (operand.layout().has_strides() || result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type, typename Value>
absl::Status IsFinite(const Value& operand, Tensor& result) {
  if (auto check = CheckParameters(operand, result); !check.ok()) {
    return check;
  }

  using S = Storage<storage_type>;

  auto operand_buffer = operand.buffer();
  auto result_buffer = result.buffer();

  size_t n = operand.num_elements();
  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != operand.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }

    for (size_t i = 0; i < n; ++i) {
      auto x = S::Get(operand_buffer, i);
      auto y = std::isfinite(static_cast<float>(x));
      Storage<ElementType::kI1>::Set(result_buffer, i, y);
    }

  } else {
    static_assert(std::is_same_v<Value, QuantizedTensor>);

    const QuantizedParameter& operand_quant_param =
        operand.type().element_type().parameters(0);

    for (size_t i = 0; i < n; ++i) {
      auto operand_storage = S::Get(operand_buffer, i);
      auto operand_expressed = Dequantize<storage_type, expressed_type>(
          operand_storage, operand_quant_param);
      auto result_expressed =
          std::isfinite(static_cast<float>(operand_expressed));
      Storage<ElementType::kI1>::Set(result_buffer, i, result_expressed);
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status IsFinite(const Tensor& operand, Tensor& result) {
  DISPATCH_FLOAT(IsFinite, operand.element_type(), operand, result);
}

absl::Status IsFinite(const QuantizedTensor& operand, Tensor& result) {
  DISPATCH_QUANTIZED(IsFinite, operand.storage_type(), operand.expressed_type(),
                     operand, result);
}

}  // namespace stablehlo

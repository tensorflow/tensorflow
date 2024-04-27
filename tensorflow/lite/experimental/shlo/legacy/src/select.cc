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
absl::Status CheckParameters(const Tensor& pred, const Value& on_true,
                             const Value& on_false, Value& result) {
  if (!(pred.rank() == 0 or pred.shape() == on_true.shape())) {
    return absl::InvalidArgumentError(
        "Constraint violation: rank(pred) = 0 or shape(pred) = "
        "shape(on_true)");
  } else if (!(on_true.baseline_type() == on_false.baseline_type() and
               on_true.baseline_type() == result.baseline_type())) {
    return absl::InvalidArgumentError(
        "Constraint violation: baseline_type(on_true) = "
        "baseline_type(on_false) = baseline_type(result)");
  } else if (pred.element_type() != ElementType::kI1) {
    return absl::InvalidArgumentError("Expected boolean tensor as predicate");
  }

  if constexpr (std::is_same_v<Value, QuantizedTensor>) {
    if (!(on_true.is_per_tensor_quantized() and
          on_false.is_per_tensor_quantized() and
          result.is_per_tensor_quantized())) {
      return absl::InvalidArgumentError("Expected per-tensor quantization");
    }
  }

  if (pred.layout().has_strides() || on_true.layout().has_strides() ||
      on_false.layout().has_strides() || result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type, typename Value>
absl::Status Select(const Tensor& pred, const Value& on_true,
                    const Value& on_false, Value& result) {
  if (auto check = CheckParameters(pred, on_true, on_false, result);
      !check.ok()) {
    return check;
  }

  using P = Storage<ElementType::kI1>;
  using S = Storage<storage_type>;

  const bool pred_is_tensor = (pred.rank() > 0);
  const size_t n = result.num_elements();

  auto pred_buffer = pred.buffer();
  auto on_true_buffer = on_true.buffer();
  auto on_false_buffer = on_false.buffer();
  auto result_buffer = result.buffer();

  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != result.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }

    bool selection_value;
    for (size_t i = 0; i < n; ++i) {
      if (pred_is_tensor || (i == 0)) {
        selection_value = P::Get(pred_buffer, i);
      }
      auto input_buffer = selection_value ? on_true_buffer : on_false_buffer;
      auto result_value = S::Get(input_buffer, i);
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

    const QuantizedParameter& on_true_quant_param =
        on_true.type().element_type().parameters(0);
    const QuantizedParameter& on_false_quant_param =
        on_false.type().element_type().parameters(0);
    const QuantizedParameter& result_quant_param =
        result.type().element_type().parameters(0);
    ET result_scale_inv = ET(1.0) / static_cast<ET>(result_quant_param.scale);

    bool selection_value;
    for (size_t i = 0; i < n; ++i) {
      if (pred_is_tensor || (i == 0)) {
        selection_value = P::Get(pred_buffer, i);
      }

      const void* input_buffer;
      const QuantizedParameter* input_quant_param;
      if (selection_value) {
        input_buffer = on_true_buffer;
        input_quant_param = &on_true_quant_param;
      } else {
        input_buffer = on_false_buffer;
        input_quant_param = &on_false_quant_param;
      }

      auto input_storage = S::Get(input_buffer, i);
      auto result_storage =
          DequantizeOpQuantizePartial<storage_type, expressed_type>(
              input_storage, *input_quant_param, result_scale_inv,
              result_quant_param.zero_point, [](auto x) { return x; });
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

absl::Status Select(const Tensor& pred, const Tensor& on_true,
                    const Tensor& on_false, Tensor& result) {
  DISPATCH_BOOL_INT_FLOAT(Select, result.element_type(), pred, on_true,
                          on_false, result);
}

absl::Status Select(const Tensor& pred, const QuantizedTensor& on_true,
                    const QuantizedTensor& on_false, QuantizedTensor& result) {
  DISPATCH_QUANTIZED(Select, result.storage_type(), result.expressed_type(),
                     pred, on_true, on_false, result);
}

}  // namespace stablehlo

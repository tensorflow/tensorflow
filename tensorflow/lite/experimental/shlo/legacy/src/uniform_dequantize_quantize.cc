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

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/dispatch.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

namespace {

absl::Status CheckDequantizeParameters(const QuantizedTensor& operand,
                                       Tensor& result) {
  if (operand.shape() != result.shape()) {
    return absl::InvalidArgumentError("Inconsistent input/output shapes");
  } else if (operand.expressed_type() != result.element_type()) {
    return absl::InvalidArgumentError("Inconsistent element types");
  } else if (!operand.is_per_tensor_quantized()) {
    return absl::InvalidArgumentError("Unsupported input quantization");
  }

  if (operand.layout().has_strides() || result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type>
absl::Status UniformDequantize(const QuantizedTensor& operand, Tensor& result) {
  if (auto check = CheckDequantizeParameters(operand, result); !check.ok()) {
    return check;
  }

  const QuantizedParameter& operand_quant_param =
      operand.type().element_type().parameters(0);

  size_t n = operand.num_elements();

  using S = Storage<storage_type>;
  using E = Storage<expressed_type>;
  auto operand_buffer = operand.buffer();
  auto result_buffer = result.buffer();

  for (size_t i = 0; i < n; ++i) {
    auto operand_storage = S::Get(operand_buffer, i);
    auto operand_expressed = Dequantize<storage_type, expressed_type>(
        operand_storage, operand_quant_param);
    auto result_expressed = operand_expressed;
    E::Set(result_buffer, i, result_expressed);
  }

  return absl::OkStatus();
}

absl::Status CheckQuantizeParameters(const Tensor& operand,
                                     QuantizedTensor& result) {
  if (operand.shape() != result.shape()) {
    return absl::InvalidArgumentError("Inconsistent input/output shapes");
  } else if (operand.element_type() != result.expressed_type()) {
    return absl::InvalidArgumentError("Inconsistent element types");
  } else if (!result.is_per_tensor_quantized()) {
    return absl::InvalidArgumentError("Unsupported output quantization");
  }
  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type>
absl::Status UniformQuantize(const Tensor& operand, QuantizedTensor& result) {
  if (auto check = CheckQuantizeParameters(operand, result); !check.ok()) {
    return check;
  }

  const QuantizedParameter& result_quant_param =
      result.type().element_type().parameters(0);

  size_t n = operand.num_elements();

  using S = Storage<storage_type>;
  using E = Storage<expressed_type>;
  auto operand_buffer = operand.buffer();
  auto result_buffer = result.buffer();

  using ET = typename E::Type;
  ET result_scale_inv = ET(1.0) / static_cast<ET>(result_quant_param.scale);

  for (size_t i = 0; i < n; ++i) {
    auto operand_expressed = E::Get(operand_buffer, i);
    auto result_expressed = operand_expressed;
    auto result_storage = QuantizePartial<storage_type, expressed_type>(
        result_expressed, result_scale_inv, result_quant_param.zero_point);
    S::Set(result_buffer, i, result_storage);
  }

  if (auto status = CompleteQuantization<storage_type>(result); !status.ok()) {
    return status;
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status UniformDequantize(const QuantizedTensor& operand, Tensor& result) {
  DISPATCH_QUANTIZED(UniformDequantize, operand.storage_type(),
                     operand.expressed_type(), operand, result);
}

absl::Status UniformQuantize(const Tensor& operand, QuantizedTensor& result) {
  DISPATCH_QUANTIZED(UniformQuantize, result.storage_type(),
                     result.expressed_type(), operand, result);
}

}  // namespace stablehlo

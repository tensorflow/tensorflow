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
#include <iterator>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/dispatch.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

namespace {

bool IsUnique(absl::Span<const DimensionSize> span) {
  std::vector<DimensionSize> temp(span.begin(), span.end());
  auto i = std::unique(temp.begin(), temp.end());
  return std::distance(temp.begin(), i) == span.size();
}

template <typename Value>
absl::Status CheckParameters(
    const Value& operand, absl::Span<const DimensionSize> broadcast_dimensions,
    Value& result) {
  if (!operand.is_per_axis_quantized()) {
    if (!(result.element_type() == operand.element_type())) {
      return absl::InvalidArgumentError(
          "Constraint violation: element_type(result) = element_type(operand) "
          "if !is_per_axis_quantized(operand)");
    }
  }

  if (!(broadcast_dimensions.size() == operand.rank())) {
    return absl::InvalidArgumentError(
        "Constraint violation: size(broadcast_dimensions) = rank(operand)");
  } else if (!(*std::min_element(broadcast_dimensions.begin(),
                                 broadcast_dimensions.end()) >= 0 and
               *std::max_element(broadcast_dimensions.begin(),
                                 broadcast_dimensions.end()) < result.rank())) {
    return absl::InvalidArgumentError(
        "Constraint violation: 0 <= broadcast_dimensions < rank(result)");
  } else if (!(IsUnique(broadcast_dimensions))) {
    return absl::InvalidArgumentError(
        "Constraint violation: is_unique(broadcast_dimensions)");
  } else {
    for (auto d : operand.axes()) {
      if (!(operand.dim(d) == 1 or
            operand.dim(d) == result.dim(broadcast_dimensions[d]))) {
        return absl::InvalidArgumentError(
            "Constraint violation: dim(operand, d) = 1 or dim(operand, d) = "
            "dim(result, broadcast_dimensions[d])");
      }
    }
  }

  if constexpr (std::is_same_v<Value, QuantizedTensor>) {
    if (operand.is_per_axis_quantized()) {
      if (!(operand.is_per_axis_quantized() and
            result.storage_type() == operand.storage_type() and
            result.expressed_type() == operand.expressed_type() and
            result.storage_min() == operand.storage_min() and
            result.storage_max() == operand.storage_max())) {
        return absl::InvalidArgumentError(
            "Constraint violation: element_type(result) = "
            "element_type(operand) with exceptions if "
            "is_per_axis_quantized(operand)");
      }
    }
    if (result.is_per_axis_quantized()) {
      if (!(*result.quantized_dimension() ==
            broadcast_dimensions[*operand.quantized_dimension()])) {
        return absl::InvalidArgumentError(
            "quantization_dimension(result) = "
            "broadcast_dimensions[quantization_dimension(operand)]");
      }
      if (operand.dim(*operand.quantized_dimension()) == 1) {
        auto n = result.dim(*result.quantized_dimension());
        for (auto i = 0; i < n; ++i) {
          if (!(result.scales(i) == operand.scales(0) and
                result.zero_points(i) == operand.zero_points(0))) {
            return absl::InvalidArgumentError(
                "If dim(operand, quantization_dimension(operand)) = 1, then "
                "scales(result)[i] = scales(operand)[0] and "
                "zero_points(result)[i] = zero_points(operand)[0] for i in "
                "range(dim(result, quantization_dimension(result)))");
          }
        }
      }
    }
  }

  if (operand.layout().has_strides() || result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type, typename Value>
absl::Status BroadcastInDim(
    const Value& operand, absl::Span<const DimensionSize> broadcast_dimensions,
    Value& result) {
  if (auto check = CheckParameters(operand, broadcast_dimensions, result);
      !check.ok()) {
    return check;
  }

  using S = Storage<storage_type>;

  auto operand_buffer = operand.buffer();
  auto result_buffer = result.buffer();

  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != operand.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }

    TensorIndex operand_index(operand.shape());
    for (TensorIndexIterator result_index_iter{result.shape()};
         result_index_iter.has_next(); ++result_index_iter) {
      for (auto d = 0; d < operand.rank(); ++d) {
        if (operand.dim(d) == 1) {
          operand_index.set(d, 0);
        } else {
          auto b = broadcast_dimensions[d];
          operand_index.set(d, (*result_index_iter)[b]);
        }
      }
      auto linearized_operand_index = operand_index.linearize();
      auto linearized_result_index = result_index_iter->linearize();
      auto value = S::Get(operand_buffer, linearized_operand_index);
      S::Set(result_buffer, linearized_result_index, value);
    }

  } else if constexpr (std::is_same_v<Value, QuantizedTensor>) {
    if (storage_type != result.storage_type()) {
      return absl::InvalidArgumentError("Unexpected storage type");
    } else if (expressed_type != result.expressed_type()) {
      return absl::InvalidArgumentError("Unexpected expressed type");
    }

    if (!(operand.is_per_tensor_quantized() and
          result.is_per_tensor_quantized())) {
      return absl::InvalidArgumentError(
          "Only per-tensor quantization is currently supported");
    }

    using ET = typename Storage<expressed_type>::Type;

    const QuantizedParameter& operand_quant_param =
        operand.type().element_type().parameters(0);
    const QuantizedParameter& result_quant_param =
        result.type().element_type().parameters(0);
    ET result_scale_inv = ET(1.0) / static_cast<ET>(result_quant_param.scale);

    TensorIndex operand_index(operand.shape());
    for (TensorIndexIterator result_index_iter{result.shape()};
         result_index_iter.has_next(); ++result_index_iter) {
      for (auto d = 0; d < operand.rank(); ++d) {
        if (operand.dim(d) == 1) {
          operand_index.set(d, 0);
        } else {
          auto b = broadcast_dimensions[d];
          operand_index.set(d, (*result_index_iter)[b]);
        }
      }

      auto linearized_operand_index = operand_index.linearize();
      auto linearized_result_index = result_index_iter->linearize();

      auto operand_storage = S::Get(operand_buffer, linearized_operand_index);
      auto result_storage =
          DequantizeOpQuantizePartial<storage_type, expressed_type>(
              operand_storage, operand_quant_param, result_scale_inv,
              result_quant_param.zero_point, [](auto x) { return x; });
      S::Set(result_buffer, linearized_result_index, result_storage);
    }

    if (auto status = CompleteQuantization<storage_type>(result);
        !status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status BroadcastInDim(
    const Tensor& operand, absl::Span<const DimensionSize> broadcast_dimensions,
    Tensor& result) {
  DISPATCH_BOOL_INT_FLOAT(BroadcastInDim, result.element_type(), operand,
                          broadcast_dimensions, result);
}

absl::Status BroadcastInDim(
    const QuantizedTensor& operand,
    absl::Span<const DimensionSize> broadcast_dimensions,
    QuantizedTensor& result) {
  DISPATCH_QUANTIZED(BroadcastInDim, result.storage_type(),
                     result.expressed_type(), operand, broadcast_dimensions,
                     result);
}

}  // namespace stablehlo

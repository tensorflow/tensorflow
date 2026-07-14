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
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/dispatch.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

namespace {

template <typename Value>
absl::Status CheckParameters(absl::Span<const Value*> inputs,
                             DimensionSize dimension, Value& result) {
  for (auto i = 1; i < inputs.size(); ++i) {
    if (!(inputs[i]->element_type() == inputs[0]->element_type())) {
      return absl::InvalidArgumentError(
          "Constraint violation: same(element_type(inputs...))");
    }
  }

  for (size_t ax = 0; ax < inputs[0]->rank(); ++ax) {
    if (ax != dimension) {
      for (auto i = 1; i < inputs.size(); ++i) {
        if (!(inputs[i]->dim(ax) == inputs[0]->dim(ax))) {
          return absl::InvalidArgumentError(
              "Constraint violation: same(shape(inputs...)) except for "
              "dim(inputs..., dimension)");
        }
      }
    }
  }

  if (inputs.empty()) {
    return absl::InvalidArgumentError("Constraint violation: 0 < size(inputs)");
  } else if (!(dimension >= 0 && dimension < inputs[0]->rank())) {
    return absl::InvalidArgumentError(
        "Constraint violation: 0 <= dimension < rank(inputs[0])");
  } else if (!(result.element_type() == inputs[0]->element_type())) {
    return absl::InvalidArgumentError(
        "Constraint violation: element_type(result) = element_type(inputs[0])");
  } else {
    for (size_t ax = 0; ax < result.rank(); ++ax) {
      DimensionSize expected_dim_size = 0;
      if (ax == dimension) {
        for (auto* x : inputs) {
          expected_dim_size += x->dim(ax);
        }
      } else {
        expected_dim_size = inputs[0]->dim(ax);
      }
      if (!(result.dim(ax) == expected_dim_size)) {
        return absl::InvalidArgumentError(
            "Constraint violation: element_type(result) = "
            "element_type(inputs[0])");
      }
    }
  }

  if (std::any_of(inputs.begin(), inputs.end(),
                  [](auto* x) { return x->layout().has_strides(); }) ||
      result.layout().has_strides()) {
    return absl::InvalidArgumentError("Stides not supported yet");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type, typename Value>
absl::Status Concatenate(absl::Span<const Value*> inputs,
                         DimensionSize dimension, Value& result) {
  if (auto check = CheckParameters(inputs, dimension, result); !check.ok()) {
    return check;
  }

  using S = Storage<storage_type>;

  auto result_buffer = result.buffer();

  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != result.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }

    DimensionSize dimension_offset = 0;
    TensorIndex result_index(result.shape());
    for (auto* input : inputs) {
      auto input_buffer = input->buffer();
      for (TensorIndexIterator input_iter{input->shape()};
           input_iter.has_next(); ++input_iter) {
        const TensorIndex& input_index = *input_iter;
        result_index.set(input_index);
        auto new_dim_size = result_index[dimension] + dimension_offset;
        result_index.set(dimension, new_dim_size);

        auto linearized_input_index = input_index.linearize();
        auto linearized_result_index = result_index.linearize();

        auto value = S::Get(input_buffer, linearized_input_index);
        S::Set(result_buffer, linearized_result_index, value);
      }

      dimension_offset += input->dim(dimension);
    }

  } else {
    static_assert(std::is_same_v<Value, QuantizedTensor>);

    if (storage_type != result.storage_type()) {
      return absl::InvalidArgumentError("Unexpected storage type");
    } else if (expressed_type != result.expressed_type()) {
      return absl::InvalidArgumentError("Unexpected expressed type");
    }

    using ET = typename Storage<expressed_type>::Type;
    const QuantizedParameter& result_quant_param =
        result.type().element_type().parameters(0);
    ET result_scale_inv = ET(1.0) / static_cast<ET>(result_quant_param.scale);

    DimensionSize dimension_offset = 0;
    TensorIndex result_index(result.shape());
    for (auto* input : inputs) {
      auto input_buffer = input->buffer();
      const QuantizedParameter& input_quant_param =
          input->type().element_type().parameters(0);
      for (TensorIndexIterator input_iter{input->shape()};
           input_iter.has_next(); ++input_iter) {
        const TensorIndex& input_index = *input_iter;
        result_index.set(input_index);
        auto new_dim_size = result_index[dimension] + dimension_offset;
        result_index.set(dimension, new_dim_size);

        auto linearized_input_index = input_index.linearize();
        auto linearized_result_index = result_index.linearize();

        auto input_storage = S::Get(input_buffer, linearized_input_index);
        auto result_storage =
            DequantizeOpQuantizePartial<storage_type, expressed_type>(
                input_storage, input_quant_param, result_scale_inv,
                result_quant_param.zero_point, [](auto x) { return x; });
        S::Set(result_buffer, linearized_result_index, result_storage);
      }

      dimension_offset += input->dim(dimension);
    }

    if (auto status = CompleteQuantization<storage_type>(result);
        !status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status Concatenate(absl::Span<const Tensor*> inputs,
                         DimensionSize dimension, Tensor& result) {
  DISPATCH_BOOL_INT_FLOAT(Concatenate, result.element_type(), inputs, dimension,
                          result);
}

absl::Status Concatenate(absl::Span<const QuantizedTensor*> inputs,
                         DimensionSize dimension, QuantizedTensor& result) {
  DISPATCH_QUANTIZED(Concatenate, result.storage_type(),
                     result.expressed_type(), inputs, dimension, result);
}

}  // namespace stablehlo

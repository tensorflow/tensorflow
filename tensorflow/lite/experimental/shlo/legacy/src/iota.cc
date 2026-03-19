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

#include <type_traits>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/dispatch.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

namespace {

template <typename Value>
absl::Status CheckParameters(DimensionSize iota_dimension,
                             const Value& result) {
  if (!(0 <= iota_dimension && iota_dimension < result.rank())) {
    return absl::InvalidArgumentError(
        "Constraint violation: 0 <= iota_dimension < rank(result)");
  }

  return absl::OkStatus();
}

template <ElementType storage_type, ElementType expressed_type, typename Value>
absl::Status Iota(DimensionSize iota_dimension, Value& result) {
  if (auto check = CheckParameters(iota_dimension, result); !check.ok()) {
    return check;
  }

  using S = Storage<storage_type>;
  using ST = typename S::Type;

  auto result_buffer = result.buffer();

  if constexpr (std::is_same_v<Value, Tensor>) {
    if (storage_type != result.element_type()) {
      return absl::InvalidArgumentError("Unexpected tensor element type");
    }

    for (TensorIndexIterator iter(result.shape()); iter.has_next(); ++iter) {
      auto& index = *iter;
      ST y = index[iota_dimension];
      S::Set(result_buffer, index.linearize(), y);
    }

  } else {
    static_assert(std::is_same_v<Value, QuantizedTensor>);

    const QuantizedParameter& result_quant_param =
        result.type().element_type().parameters(0);

    using ET = typename Storage<expressed_type>::Type;

    ET result_scale_inv = ET(1.0) / static_cast<ET>(result_quant_param.scale);

    for (TensorIndexIterator iter(result.shape()); iter.has_next(); ++iter) {
      auto& index = *iter;
      ET result_expressed = index[iota_dimension];
      auto result_storage = QuantizePartial<storage_type, expressed_type>(
          result_expressed, result_scale_inv, result_quant_param.zero_point);
      S::Set(result_buffer, index.linearize(), result_storage);
    }

    if (auto status = CompleteQuantization<storage_type>(result);
        !status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status Iota(DimensionSize iota_dimension, Tensor& result) {
  DISPATCH_INT_FLOAT(Iota, result.element_type(), iota_dimension, result);
}

absl::Status Iota(DimensionSize iota_dimension, QuantizedTensor& result) {
  DISPATCH_QUANTIZED(Iota, result.storage_type(), result.expressed_type(),
                     iota_dimension, result);
}

}  // namespace stablehlo

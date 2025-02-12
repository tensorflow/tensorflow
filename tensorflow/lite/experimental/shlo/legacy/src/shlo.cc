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

#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"

#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"

namespace stablehlo {

Axes Shape::axes() const {
  Axes tmp(dims_.size());
  std::iota(tmp.begin(), tmp.end(), 0);
  return tmp;
}

size_t Shape::num_elements() const {
  if (dims_.empty()) {
    return 0;
  }
  return std::accumulate(dims_.begin(), dims_.end(), 1,
                         std::multiplies<size_t>());
}

namespace {

template <ElementType element_type>
struct GetNumBytes {
  size_t operator()() {
    return sizeof(typename stablehlo::Storage<element_type>::Type);
  }
};

template <template <ElementType> typename Op, typename Result>
Result CallTemplatedFunctorWithResult(ElementType element_type) {
  switch (element_type) {
    case ElementType::kI1:
      return Op<ElementType::kI1>()();
    case ElementType::kSI8:
      return Op<ElementType::kSI8>()();
    case ElementType::kSI16:
      return Op<ElementType::kSI16>()();
    case ElementType::kSI32:
      return Op<ElementType::kSI32>()();
    case ElementType::kBF16:
      return Op<ElementType::kBF16>()();
    case ElementType::kF16:
      return Op<ElementType::kF16>()();
    case ElementType::kF32:
      return Op<ElementType::kF32>()();
    default:
      LOG(ERROR) << "Unexpected tensor element type: "
                 << static_cast<int>(element_type);
      return Result();
  }
}

}  // namespace

size_t TensorType::num_bytes() const {
  auto num_bytes_per_element =
      CallTemplatedFunctorWithResult<GetNumBytes, size_t>(element_type());
  return num_bytes_per_element * shape().num_elements();
}

bool Tensor::operator==(const Tensor& other) const {
  return (type() == other.type()) &&
         !std::memcmp(buffer(), other.buffer(), num_bytes());
}

// /////////////////////////////////////////////////////////////////////////////

size_t QuantizedTensorType::num_bytes() const {
  auto num_bytes_per_element =
      CallTemplatedFunctorWithResult<GetNumBytes, size_t>(
          element_type().storage_type());
  return num_bytes_per_element * shape_.num_elements();
}

bool QuantizedTensor::operator==(const QuantizedTensor& other) const {
  return (type() == other.type()) &&
         !std::memcmp(buffer(), other.buffer(), num_bytes());
}

QuantizedTensorType QuantizedTensor::baseline_type() const {
  // For quantized types, we ignore scales and zero points.
  auto shape = type_.shape();

  if (is_per_tensor_quantized()) {
    QuantizedTensorElementType element_type(
        storage_type(), expressed_type(),
        QuantizedParameter{.scale = 1.0, .zero_point = 0},
        type_.element_type().storage_min(), type_.element_type().storage_max());
    return QuantizedTensorType(std::move(shape), std::move(element_type));

  } else {
    auto n = shape.dim(*type_.element_type().quantized_dimension());
    std::vector<QuantizedParameter> parameters(n,
                                               {.scale = 1.0, .zero_point = 0});
    QuantizedTensorElementType element_type(
        storage_type(), expressed_type(), std::move(parameters),
        type_.element_type().storage_min(), type_.element_type().storage_max());
    return QuantizedTensorType(std::move(shape), std::move(element_type));
  }
}

}  // namespace stablehlo

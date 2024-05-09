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

#include "tensorflow/lite/experimental/shlo/tensor.h"

#include <cassert>
#include <cstddef>
#include <variant>

#include "absl/log/absl_check.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/overload.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"

namespace shlo_ref {

TensorElementTypeVariant BaselineType(const TensorElementTypeVariant& type) {
  return std::visit(
      [](auto t) -> TensorElementTypeVariant { return BaselineType(t); }, type);
}

const Shape& Tensor::shape() const {
  return std::visit([](auto& t) -> const Shape& { return t.shape; }, type);
}

Shape& Tensor::shape() {
  return std::visit([](auto& t) -> Shape& { return t.shape; }, type);
}

bool Tensor::IsQuantized() const {
  return IsPerTensorQuantized() || IsPerAxisQuantized();
}

bool Tensor::IsPerAxisQuantized() const {
  return std::holds_alternative<QuantizedPerAxisTensorType>(type);
}
bool Tensor::IsPerTensorQuantized() const {
  return std::holds_alternative<QuantizedPerTensorTensorType>(type);
}

size_t Tensor::Rank() const { return shape().Rank(); }

DataType Tensor::StorageType() const {
  return std::visit(
      shlo_ref::Overload(
          [](const TensorType& t) { return t.element_type; },
          [](const auto& t) { return t.element_type.StorageType(); }),
      type);
}

DimensionSize Tensor::NumElements() const { return shape().NumElements(); }

size_t Tensor::SizeInBytes() const {
  return SizeOf(StorageType()) * NumElements();
}

TensorType& Tensor::tensor_type() {
  ABSL_CHECK(std::holds_alternative<TensorType>(type));
  return std::get<TensorType>(type);
}

const TensorType& Tensor::tensor_type() const {
  ABSL_CHECK(std::holds_alternative<TensorType>(type));
  return std::get<TensorType>(type);
}

QuantizedPerTensorTensorType& Tensor::quantized_per_tensor_type() {
  ABSL_CHECK(std::holds_alternative<QuantizedPerTensorTensorType>(type));
  return std::get<QuantizedPerTensorTensorType>(type);
}

const QuantizedPerTensorTensorType& Tensor::quantized_per_tensor_type() const {
  assert(std::holds_alternative<QuantizedPerTensorTensorType>(type));
  return std::get<QuantizedPerTensorTensorType>(type);
}

QuantizedPerAxisTensorType& Tensor::quantized_per_axis_type() {
  ABSL_CHECK(std::holds_alternative<QuantizedPerAxisTensorType>(type));
  return std::get<QuantizedPerAxisTensorType>(type);
}

const QuantizedPerAxisTensorType& Tensor::quantized_per_axis_type() const {
  assert(std::holds_alternative<QuantizedPerAxisTensorType>(type));
  return std::get<QuantizedPerAxisTensorType>(type);
}

const TensorElementType& Tensor::tensor_element_type() const {
  return tensor_type().element_type;
}

const QuantizedElementTypePerTensor& Tensor::quantized_per_tensor_element_type()
    const {
  return quantized_per_tensor_type().element_type;
}

const QuantizedElementTypePerAxis& Tensor::quantized_per_axis_element_type()
    const {
  return quantized_per_axis_type().element_type;
}

TensorElementTypeVariant Tensor::element_type() const {
  return std::visit(
      [](const auto& t) -> TensorElementTypeVariant { return t.element_type; },
      type);
}

void Tensor::GetNdIndex(
    size_t index, absl::InlinedVector<Axis, kMaxNumDimensions>& indices) const {
  size_t divisor = 1, dim = 0;
  Axis rank = Rank();
  for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i) {
    dim = shape().Dim(i);
    indices[i] = (index / divisor) % dim;
    divisor *= dim;
  }
  return;
}

DimensionSize Tensor::FlattenIndex(
    absl::InlinedVector<Axis, kMaxNumDimensions>& indices) const {
  DimensionSize index = 0;
  if (shape().empty()) {
    return index;
  }
  size_t rank = Rank();
  absl::InlinedVector<Axis, kMaxNumDimensions> strides(rank);
  strides[rank - 1] = 1;
  for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape()[i + 1];
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    index += strides[i] * indices[i];
  }
  return index;
}

bool operator==(const TensorType& lhs, const TensorType& rhs) {
  return lhs.element_type == rhs.element_type && lhs.shape == rhs.shape;
}

bool operator!=(const TensorType& lhs, const TensorType& rhs) {
  return !(lhs == rhs);
}

bool operator==(const QuantizedPerTensorTensorType& lhs,
                const QuantizedPerTensorTensorType& rhs) {
  return lhs.element_type == rhs.element_type && lhs.shape == rhs.shape;
}

bool operator!=(const QuantizedPerTensorTensorType& lhs,
                const QuantizedPerTensorTensorType& rhs) {
  return !(lhs == rhs);
}

bool operator==(const QuantizedPerAxisTensorType& lhs,
                const QuantizedPerAxisTensorType& rhs) {
  return lhs.element_type == rhs.element_type && lhs.shape == rhs.shape;
}

bool operator!=(const QuantizedPerAxisTensorType& lhs,
                const QuantizedPerAxisTensorType& rhs) {
  return !(lhs == rhs);
}

}  // namespace shlo_ref

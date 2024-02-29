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

#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"

namespace shlo_ref {

const Shape& Tensor::shape() const {
  if (IsQuantized()) {
    return quantized_tensor_type().shape;
  } else {
    return tensor_type().shape;
  }
}

Shape& Tensor::shape() {
  if (IsQuantized()) {
    return quantized_tensor_type().shape;
  } else {
    return tensor_type().shape;
  }
}

bool Tensor::IsQuantized() const {
  return std::holds_alternative<QuantizedTensorType>(type);
}

bool Tensor::IsPerAxisQuantized() const {
  return IsQuantized() &&
         std::get<QuantizedTensorType>(type).element_type.IsPerAxisQuantized();
}
bool Tensor::IsPerTensorQuantized() const {
  return IsQuantized() && std::get<QuantizedTensorType>(type)
                              .element_type.IsPerTensorQuantized();
}

size_t Tensor::Rank() const {
  return IsQuantized() ? quantized_tensor_type().shape.Rank()
                       : tensor_type().shape.Rank();
}

DataType Tensor::StorageType() const {
  return IsQuantized() ? quantized_tensor_type().element_type.StorageType()
                       : tensor_type().element_type;
}

DimensionSize Tensor::NumElements() const {
  return IsQuantized() ? quantized_tensor_type().shape.NumElements()
                       : tensor_type().shape.NumElements();
}

size_t Tensor::SizeInBytes() const {
  if (IsQuantized()) {
    return SizeOf(quantized_tensor_type().element_type.StorageType()) *
           quantized_tensor_type().shape.NumElements();
  } else {
    return SizeOf(tensor_type().element_type) *
           tensor_type().shape.NumElements();
  }
}

TensorType& Tensor::tensor_type() {
  assert(std::holds_alternative<TensorType>(type));
  return std::get<TensorType>(type);
}

const TensorType& Tensor::tensor_type() const {
  assert(std::holds_alternative<TensorType>(type));
  return std::get<TensorType>(type);
}

QuantizedTensorType& Tensor::quantized_tensor_type() {
  assert(std::holds_alternative<QuantizedTensorType>(type));
  return std::get<QuantizedTensorType>(type);
}

const QuantizedTensorType& Tensor::quantized_tensor_type() const {
  assert(std::holds_alternative<QuantizedTensorType>(type));
  return std::get<QuantizedTensorType>(type);
}

const TensorElementType& Tensor::tensor_element_type() const {
  return tensor_type().element_type;
}
const QuantizedTensorElementType& Tensor::quantized_tensor_element_type()
    const {
  return quantized_tensor_type().element_type;
}

bool operator==(const TensorType& lhs, const TensorType& rhs) {
  return lhs.element_type == rhs.element_type && lhs.shape == rhs.shape;
}

bool operator!=(const TensorType& lhs, const TensorType& rhs) {
  return !(lhs == rhs);
}

bool operator==(const QuantizedTensorType& lhs,
                const QuantizedTensorType& rhs) {
  return lhs.element_type == rhs.element_type && lhs.shape == rhs.shape;
}

bool operator!=(const QuantizedTensorType& lhs,
                const QuantizedTensorType& rhs) {
  return !(lhs == rhs);
}

}  // namespace shlo_ref

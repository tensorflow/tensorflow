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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_WITH_DATA_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_WITH_DATA_H_

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

// This is a utility class for creating a Tensor with a backing data buffer for
// the tensor data. It is primarily to be used for testing.
class TensorWithData {
 public:
  // Creates a non-quantized tensor.
  template <DataType storage_type>
  static TensorWithData Create(
      Shape shape, absl::Span<const StorageType<storage_type>> data) {
    Tensor tensor{
        TensorType{.shape = std::move(shape), .element_type = storage_type}};
    std::vector<std::byte> buffer(tensor.SizeInBytes());
    std::memcpy(buffer.data(), data.data(), tensor.SizeInBytes());
    return TensorWithData(std::move(tensor), std::move(buffer));
  }

  // Creates a per-tensor quantized tensor.
  template <DataType storage_type, DataType expressed_type>
  static TensorWithData Create(
      Shape shape, absl::Span<const StorageType<expressed_type>> data,
      StorageType<expressed_type> scale, StorageType<storage_type> zero_point) {
    static_assert(IsInteger(storage_type));
    static_assert(IsFloat(expressed_type));
    using StorageT = typename Storage<storage_type>::Type;
    using ExpressedT = typename Storage<expressed_type>::Type;

    Tensor tensor{QuantizedPerTensorTensorType{
        .shape = std::move(shape),
        .element_type = QuantizedElementTypePerTensor(storage_type, zero_point,
                                                      expressed_type, scale)}};

    const ExpressedT scale_inv = ExpressedT(1.0) / scale;
    std::vector<StorageT> quantized_data;
    for (const auto& expressed_value : data) {
      quantized_data.push_back(Quantize<storage_type, expressed_type>(
          expressed_value, zero_point, scale_inv));
    }

    std::vector<std::byte> buffer(tensor.SizeInBytes());
    std::memcpy(buffer.data(), quantized_data.data(), tensor.SizeInBytes());
    return TensorWithData(std::move(tensor), std::move(buffer));
  }

  TensorWithData(const TensorWithData& other)
      : tensor_(other.tensor_), buffer_(other.buffer_) {
    tensor_.data = buffer_.data();
  }

  TensorWithData& operator=(const TensorWithData& other) {
    tensor_ = other.tensor_;
    buffer_ = other.buffer_;
    tensor_.data = buffer_.data();
    return *this;
  }

  const Tensor& tensor() const { return tensor_; }

 private:
  TensorWithData(Tensor tensor, std::vector<std::byte> buffer)
      : tensor_(std::move(tensor)), buffer_(std::move(buffer)) {
    tensor_.data = buffer_.data();
  }

  Tensor tensor_;
  std::vector<std::byte> buffer_;
};

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_WITH_DATA_H_

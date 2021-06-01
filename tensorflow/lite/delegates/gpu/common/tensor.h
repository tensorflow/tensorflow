/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TENSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TENSOR_H_

#include <stdint.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {
namespace internal_tensor {

// Meta function given element type returns a type for Tensor data container.
template <DataType Type>
struct StorageType;

template <>
struct StorageType<DataType::FLOAT32> {
  using value = std::vector<float>;
};

template <>
struct StorageType<DataType::INT32> {
  using value = std::vector<int32_t>;
};

template <>
struct StorageType<DataType::INT16> {
  using value = std::vector<int16_t>;
};

template <>
struct StorageType<DataType::INT8> {
  using value = std::vector<int8_t>;
};

template <>
struct StorageType<DataType::UINT32> {
  using value = std::vector<uint32_t>;
};

template <>
struct StorageType<DataType::UINT16> {
  using value = std::vector<uint16_t>;
};

template <>
struct StorageType<DataType::UINT8> {
  using value = std::vector<uint8_t>;
};

}  // namespace internal_tensor

template <typename ShapeT, DataType Type>
struct Tensor {
  using ShapeType = ShapeT;

  constexpr static DataType kType = Type;

  using TensorStorageType = typename internal_tensor::StorageType<Type>::value;

  // Opaque id of a tensor.
  int64_t id = -1;

  ShapeType shape;

  TensorStorageType data;
};

// TensorRef is a reference to another tensor. If an object should never hold
// tensor data, then TensorRef should be used instead.
template <typename ShapeT>
struct TensorRef {
  using ShapeType = ShapeT;

  DataType type = DataType::UNKNOWN;

  ShapeT shape;

  // Opaque reference to a tensor. Upstream component is responsible for
  // resolving this reference into an actual tensor.
  int64_t ref = -1;

  // Specifies if the tensor should be a variable input tensor that must be an
  // output as well as an input to the graph.
  bool is_variable_input = false;
};

template <typename ShapeT, DataType Type>
constexpr DataType Tensor<ShapeT, Type>::kType;

template <typename ShapeT, DataType Type>
Tensor<ShapeT, Type> MakeZeroTensor(const ShapeT& shape) {
  Tensor<ShapeT, Type> tensor;
  tensor.shape = shape;
  tensor.data = typename Tensor<ShapeT, Type>::TensorStorageType(
      shape.DimensionsProduct(), 0);
  return tensor;
}

using TensorFloat32 = Tensor<BHWC, DataType::FLOAT32>;
using Tensor5DFloat32 = Tensor<BHWDC, DataType::FLOAT32>;

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TENSOR_H_

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_H_

#include <vector>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {

template <typename T>
inline T* GetTensorData(TfLiteTensor* tensor);

template <>
inline float* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.f : nullptr;
}

template <>
inline uint8_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.uint8 : nullptr;
}

template <>
inline int32_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i32 : nullptr;
}

template <>
inline int64_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i64 : nullptr;
}

inline int RemapDim(int max_dimensions, int d) {
  return max_dimensions - d - 1;
}

// TODO(ahentz): the implementations in kernels/internal/ take a Dims<4> object
// even if the original tensors were not 4D. We should consider rewriting them
// to take a more generic 'shape' object.
inline Dims<4> GetTensorDims(const int data[], const int size) {
  Dims<4> d;
  for (int i = 0; i < 4; ++i) {
    int src = size - i - 1;
    if (src >= 0) {
      d.sizes[i] = data[src];
    } else {
      d.sizes[i] = 1;
    }
  }
  d.strides[0] = 1;
  for (int i = 1; i < 4; i++) {
    d.strides[i] = d.strides[i - 1] * d.sizes[i - 1];
  }
  return d;
}

inline Dims<4> GetTensorDims(std::vector<int32_t> data) {
  return GetTensorDims(data.data(), data.size());
}

inline Dims<4> GetTensorDims(const TfLiteTensor* tensor) {
  if (tensor == nullptr) {
    return Dims<4>();
  }

  auto* dims = tensor->dims;
  return GetTensorDims(dims->data, dims->size);
}

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_H_

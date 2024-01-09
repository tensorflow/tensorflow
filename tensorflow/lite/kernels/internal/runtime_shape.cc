/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/runtime_shape.h"

#include <cstring>

namespace tflite {

RuntimeShape::~RuntimeShape() {
  if (size_ > kMaxSmallSize) {
    delete[] dims_pointer_;
  }
}

int32_t RuntimeShape::Dims(int i) const {
  TFLITE_DCHECK_GE(i, 0);
  TFLITE_DCHECK_LT(i, size_);
  return size_ > kMaxSmallSize ? dims_pointer_[i] : dims_[i];
}

void RuntimeShape::ReplaceWith(int dimensions_count, const int32_t* dims_data) {
  Resize(dimensions_count);
  int32_t* dst_dims = DimsData();
  std::memcpy(dst_dims, dims_data, dimensions_count * sizeof(int32_t));
}

int RuntimeShape::FlatSize() const {
  int buffer_size = 1;
  const int* dims_data = reinterpret_cast<const int*>(DimsData());
  for (int i = 0; i < size_; i++) {
    buffer_size *= dims_data[i];
  }
  return buffer_size;
}

}  // namespace tflite

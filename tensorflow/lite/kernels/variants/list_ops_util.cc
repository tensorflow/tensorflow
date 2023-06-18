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
#include "tensorflow/lite/kernels/variants/list_ops_util.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {

// Creates a `TfLiteIntArray*` from tensor data that represents a shape.
IntArrayUniquePtr TensorAsShape(const TfLiteTensor& shape) {
  if (shape.dims->size == 0) {
    // `shape` tensor encode an unranked shape.
    return BuildTfLiteIntArray({});
  }
  const int rank = shape.dims->data[0];
  const int* begin = reinterpret_cast<const int*>(shape.data.data);
  const int* end = begin + rank;
  return BuildTfLiteIntArray(std::vector<int>(begin, end));
}
}  // namespace variants
}  // namespace tflite

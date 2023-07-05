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

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {

// Creates a `TfLiteIntArray*` from tensor data that represents a shape.
IntArrayUniquePtr TensorAsShape(const TfLiteTensor& shape) {
  if (shape.dims->size == 0) {
    // `shape` tensor encode an unranked shape.
    return BuildTfLiteArray({});
  }
  const int rank = shape.dims->data[0];
  const int* begin = reinterpret_cast<const int*>(shape.data.data);
  const int* end = begin + rank;
  return BuildTfLiteArray(std::vector<int>(begin, end));
}

IntArrayUniquePtr MergeShapesOrNull(IntArrayUniquePtr l, IntArrayUniquePtr r) {
  if (l->size == 0) {
    return r;
  }
  if (r->size == 0) {
    return l;
  }
  if (l->size != r->size) {
    return nullptr;
  }
  for (int i = 0; i < r->size; ++i) {
    if (l->data[i] == -1 && r->data[i] != -1) {
      l->data[i] = r->data[i];
      continue;
    }
    if (r->data[i] == -1) continue;
    if (l->data[i] != r->data[i]) {
      return nullptr;
    }
  }
  return l;
}

bool IsShapeFullyDefined(const TfLiteIntArray& shape) {
  for (int i = 0; i < shape.size; ++i) {
    if (shape.data[i] < 0) {
      return false;
    }
  }
  return true;
}

}  // namespace variants
}  // namespace tflite

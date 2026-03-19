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

#include <vector>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"

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
  if (l == nullptr) {
    return r;
  }
  if (r == nullptr) {
    return l;
  }
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
    if (l->data[i] == -1) {
      l->data[i] = r->data[i];
    } else if (r->data[i] != -1 && l->data[i] != r->data[i]) {
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

TfLiteStatus GetShapeIfAllEqual(const TensorArray& arr,
                                IntArrayUniquePtr& result) {
  // All shapes are equal iff all shapes equal the first present shape.
  const TfLiteIntArray* common_shape = nullptr;

  for (int i = 0; i < arr.NumElements(); ++i) {
    const TfLiteTensor* cur_element = arr.At(i);
    if (cur_element == nullptr) {
      continue;
    }

    if (common_shape == nullptr) {
      common_shape = cur_element->dims;
      continue;
    }

    if (!TfLiteIntArrayEqual(common_shape, cur_element->dims)) {
      return kTfLiteError;
    }
  }

  result = common_shape != nullptr ? BuildTfLiteArray(*common_shape) : nullptr;
  return kTfLiteOk;
}

}  // namespace variants
}  // namespace tflite

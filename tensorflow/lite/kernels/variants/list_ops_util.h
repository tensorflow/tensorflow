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
#ifndef TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_UTIL_H_

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {

// Creates a dims array from tensor whose data represents a shape
// signature.
IntArrayUniquePtr TensorAsShape(const TfLiteTensor& shape);

// "Merges" two shape signature arrays if possible, returns nullptr otherwise.
// Merging means to compute the most specified possible signature which
// is compatible with both inputs.
IntArrayUniquePtr MergeShapesOrNull(IntArrayUniquePtr l, IntArrayUniquePtr r);

// Checks if array encodes a fully defined shape.
bool IsShapeFullyDefined(const TfLiteIntArray& shape);

}  // namespace variants
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_UTIL_H_

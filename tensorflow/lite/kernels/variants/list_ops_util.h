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

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {

// Creates a dims array from tensor whose data represents a shape
// signature.
// TODO(b/288302706) `TfLiteIntArray` is ill-equiped to encode the semantics
// of something like a `tf.TensorShape`. In particular, there is no way
// to cleanly capture the difference between a concrete scalar shape, and an
// unranked shape-signature. The latter is defined to be compatible with any
// shape (like `tf.TensorShape(None)`). This causes the need for some extra
// checks. Consider wrapping in something like a `std::union` to differentiate
// between these cases.
IntArrayUniquePtr TensorAsShape(const TfLiteTensor& shape);

// "Merges" two shape signature arrays if possible, returns nullptr otherwise.
// Merging means to compute the most specified possible signature which
// is compatible with both inputs.
IntArrayUniquePtr MergeShapesOrNull(IntArrayUniquePtr l, IntArrayUniquePtr r);

// Checks if array encodes a fully defined shape.
bool IsShapeFullyDefined(const TfLiteIntArray& shape);

// Returns a status denoting whether all of the elements in the `arr`
// have the same shape. Write that shape to `result`.
// If the `arr` has no set elements, still succeed but set `result` to nullptr.
// TODO(b/288302706) This may be a performance bottleneck. We could potentially
// amortize this work by constraining `TensorArray::element_shape_` every
// time an element is added. This may cause divergence from tensorflow behavior
// however; further investigation is needed.
TfLiteStatus GetShapeIfAllEqual(const TensorArray& arr,
                                IntArrayUniquePtr& result);

}  // namespace variants
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_UTIL_H_

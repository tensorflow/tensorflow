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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_SHAPE_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_SHAPE_UTIL_H_

#include <limits>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Converts a TensorShape to a constant Tensor.
//
// The input TensorShape input_shape is used to populate the elements of
// shape_constant, which is modified in place.
absl::Status TensorShapeToConstant(const TensorShape& input_shape,
                                   Tensor* shape_constant);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_SHAPE_UTIL_H_

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_UTIL_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace tensorflow {

// Returns a floating point scalar constant of 'type' with 'value'.
// If 'type' is complex, returns a real value with zero imaginary component.
xla::XlaOp FloatLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                        double value);

// Makes a 1D tensor [0, ..., x, y] from two tensors x and y with zeros
// prepended until the array is length n_dims.
xla::XlaOp PrependZerosInMajorDims(xla::XlaOp x,
                                   absl::Span<const xla::XlaOp> starts);

// Returns a integer scalar constant of 'type' with 'value'.
// If 'type' is complex, returns a real value with zero imaginary component.
xla::XlaOp IntegerLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                          int64 value);

// Returns the concatenation of `xs` and `ys`.
std::vector<int64> ConcatVectors(absl::Span<const int64> xs,
                                 absl::Span<const int64> ys);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_UTIL_H_

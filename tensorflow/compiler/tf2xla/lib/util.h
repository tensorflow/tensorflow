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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_UTIL_H_

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Returns a zero-filled tensor with shape `shape`.
xla::ComputationDataHandle Zeros(xla::ComputationBuilder* builder,
                                 const xla::Shape& shape);

// Returns a floating point scalar constant of 'type' with 'value'.
// If 'type' is complex, returns a real value with zero imaginary component.
xla::ComputationDataHandle FloatLiteral(xla::ComputationBuilder* builder,
                                        xla::PrimitiveType type, double value);

// Performs a slice in the minor dimensions of a Tensor.
xla::StatusOr<xla::ComputationDataHandle> SliceInMinorDims(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& x,
    gtl::ArraySlice<int64> start, gtl::ArraySlice<int64> end);

// Updates a slice of 'x', i.e.,
// x[start[0], ..., start[n]] = update
xla::StatusOr<xla::ComputationDataHandle> UpdateSlice(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& x,
    const xla::ComputationDataHandle& update, gtl::ArraySlice<int64> start);

// Updates a slice of 'x', where 'start' contains a list of minor dimensions:
// x[..., start[0], ..., start[n]] = update
xla::StatusOr<xla::ComputationDataHandle> UpdateSliceInMinorDims(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& x,
    const xla::ComputationDataHandle& update, gtl::ArraySlice<int64> start);

// Transposes a stack of matrices `x` by swapping the last two dimensions.
xla::StatusOr<xla::ComputationDataHandle> TransposeInMinorDims(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& x);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_UTIL_H_

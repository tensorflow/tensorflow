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

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Returns a zero-filled tensor with shape `shape`.
xla::XlaOp Zeros(xla::XlaBuilder* builder, const xla::Shape& shape);

// Returns a floating point scalar constant of 'type' with 'value'.
// If 'type' is complex, returns a real value with zero imaginary component.
xla::XlaOp FloatLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                        double value);

// Makes a 1D tensor [0, ..., x, y] from two tensors x and y with zeros
// prepended until the array is length n_dims.
xla::XlaOp PrependZerosInMajorDims(xla::XlaOp x,
                                   gtl::ArraySlice<xla::XlaOp> starts);

// Returns a integer scalar constant of 'type' with 'value'.
// If 'type' is complex, returns a real value with zero imaginary component.
xla::XlaOp IntegerLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                          int64 value);

// Builds a vector of zeros of length rank(x) with the last values being
// those in `starts`.
xla::XlaOp PrependZerosInMajorDims(xla::XlaOp x,
                                   gtl::ArraySlice<xla::XlaOp> starts);

// Performs a slice in the minor dimensions of a Tensor.
xla::XlaOp SliceInMinorDims(xla::XlaOp x, gtl::ArraySlice<int64> start,
                            gtl::ArraySlice<int64> end);

// Returns the concatenation of `xs` and `ys`.
std::vector<int64> ConcatVectors(gtl::ArraySlice<int64> xs,
                                 gtl::ArraySlice<int64> ys);

// Performs a dynamic slice in the minor dimensions of a Tensor.
xla::XlaOp DynamicSliceInMinorDims(xla::XlaOp x,
                                   gtl::ArraySlice<xla::XlaOp> starts,
                                   gtl::ArraySlice<int64> sizes);

// Updates a slice of 'x', i.e.,
// x[start[0], ..., start[n]] = update
xla::XlaOp UpdateSlice(xla::XlaOp x, xla::XlaOp update,
                       gtl::ArraySlice<int64> start);

// Updates a slice of 'x', where 'start' contains a list of minor dimensions:
// x[..., start[0], ..., start[n]] = update
xla::XlaOp UpdateSliceInMinorDims(xla::XlaOp x, xla::XlaOp update,
                                  gtl::ArraySlice<int64> start);

xla::XlaOp DynamicUpdateSliceInMinorDims(xla::XlaOp x, xla::XlaOp update,
                                         gtl::ArraySlice<xla::XlaOp> starts);

// Transposes a stack of matrices `x` by swapping the last two dimensions.
xla::XlaOp TransposeInMinorDims(xla::XlaOp x);

// Applies a complex conjugation operation if `a` is complex and `conjugate_a`
// is true, otherwise returns its argument.
xla::XlaOp MaybeConjugate(xla::XlaOp x, bool conjugate);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_UTIL_H_

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_MATRIX_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_MATRIX_H_

#include <array>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Returns an m x n matrix with 1s on the diagonal elements, zeros everywhere
// else.
XlaOp IdentityMatrix(XlaBuilder* builder, PrimitiveType type, int64 m, int64 n);

// Get the diagonals of the last two dimensions. If 'x' has shape
// [..., M, N], then the output has shape [..., min(M, N)], containing the
// diagonal elements (i.e., with indices [..., i, i]).
XlaOp GetMatrixDiagonal(XlaOp x);

// Returns a lower-triangular mask, i.e., true below the `diagonal`-th diagonal
// and false above that diagonal.
XlaOp TriangleMask(XlaOp x, int diagonal);

// Get the upper or lower triangle part of the last two dimensions
XlaOp Triangle(XlaOp x, bool lower);

// Get the upper triangle part of the last two dimensions
XlaOp UpperTriangle(XlaOp x);

// Get the lower triangle part of the last two dimensions
XlaOp LowerTriangle(XlaOp x);

// Multiplies slices of two tensors in batches.

// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
// viewed as an element of a batch), and arranges the individual results
// in a single output tensor of the same batch size.
//
// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
// and `[..., r_y, c_y]`.
//
// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
//
//     r_o = c_x if transpose_x else r_x
//     c_o = r_y if transpose_y else c_y
//
// It is computed as:
//
//     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
xla::XlaOp BatchDot(
    xla::XlaOp x, xla::XlaOp y,
    xla::PrecisionConfig::Precision precision = xla::PrecisionConfig::DEFAULT);

// Parse an einsum string into dimension numbers:
//   "ab,cb->ac"
// becomes:
//   {{0, 1},{2, 1},{0, 2}}
//
// NOTE: This function is meant for testing, there is no need to call it
// directly.

StatusOr<std::array<std::vector<int64>, 3>> ParseEinsumString(
    absl::string_view einsum_config);

// Determine if each dimension label is in at least two inputs.
//
// NOTE: This function is meant for testing, there is no need to call it
// directly.
Status ValidateEinsumNumericDimensions(absl::Span<const int64> x_config,
                                       absl::Span<const int64> y_config,
                                       absl::Span<const int64> output_config);

// Supports two operand einsum notation like "ab,cb->ac".
xla::XlaOp Einsum(
    xla::XlaOp x, xla::XlaOp y, absl::string_view einsum_config,
    xla::PrecisionConfig::Precision precision = xla::PrecisionConfig::DEFAULT);

// Same as above but supporting numeric labels on dimensins. So "ab,cb->ac"
// becomes:
//   x_config = {0, 1}
//   y_config = {2, 1}
//   output_config = {0, 2}
xla::XlaOp Einsum(
    xla::XlaOp x, absl::Span<const int64> x_config, xla::XlaOp y,
    absl::Span<const int64> y_config, absl::Span<const int64> output_config,
    xla::PrecisionConfig::Precision precision = xla::PrecisionConfig::DEFAULT);

// Transposes a stack of matrices `x` by swapping the last two dimensions.
xla::XlaOp TransposeInMinorDims(xla::XlaOp x);

// Transposes `x` in its minor dimensions if `transpose` is true, otherwise
// returns `x` unchanged.
xla::XlaOp MaybeTransposeInMinorDims(xla::XlaOp x, bool transpose);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_MATRIX_H_

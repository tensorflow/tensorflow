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
#include <vector>

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

// Returns a mask where the 'diagonal'-th diagonal is true and everything else
// is false.
XlaOp GetDiagonalMask(XlaOp x, int diagonal = 0);

// Get the diagonals of the last two dimensions. Use k>0 for diagonals above the
// main diagonal, and k<0 for diagonals below the main diagonal.
//
// If 'x' has shape [..., M, N]
//  If k >= 0: then the output has shape [..., min(M, N - k)], containing the
//            diagonal elements (i.e., with indices [..., i, i + k]).
//  If k < 0: then the output has shape [..., min(M + k, N)], containing the
//            diagonal elements (i.e., with indices [..., i - k, i]).
XlaOp GetMatrixDiagonal(XlaOp x, int k = 0);
XlaOp GetMatrixDiagonalViaGather(XlaOp x, int k = 0);

// Places diag along the kth diagonal of target.
XlaOp SetMatrixDiagonal(XlaOp matrix, XlaOp diag, int k = 0);

// Returns a lower-triangular mask, i.e., true below and including the
// `diagonal`-th diagonal and false above that diagonal.
XlaOp TriangleMask(XlaOp x, int diagonal);

// Get the upper or lower triangle part of the last two dimensions
XlaOp Triangle(XlaOp x, bool lower);

// Get the upper triangle part of the last two dimensions
XlaOp UpperTriangle(XlaOp x);

// Get the lower triangle part of the last two dimensions
XlaOp LowerTriangle(XlaOp x);

// If x is an array of shape [..., n, n], symmetrizes the matrix by replacing
// the upper triangle with the transpose of the lower triangle (if lower is
// True, vice-versa otherwise). If the type of `x` is complex, makes the matrix
// Hermitian by taking the conjugate of the complex part and setting the
// complex diagonal to zero.
XlaOp Symmetrize(XlaOp x, bool lower);

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
xla::XlaOp BatchDot(
    xla::XlaOp x, bool transpose_x, xla::XlaOp y, bool transpose_y,
    xla::PrecisionConfig::Precision precision = xla::PrecisionConfig::DEFAULT);

// Parse an einsum string into dimension numbers:
//   "ab,cb->ac"
// becomes:
//   {{0, 1},{2, 1},{0, 2}}
//
// Each occurrence of ellipsis ("...") occurring in the input is replaced with
// the same numeric dimensions. The number of such dimensions is inferred from
// x_rank and y_rank. For example:
//   einsum_config: "...ab,...bcd->...acd"
//   x_rank: 4
//   y_rank: 5
// becomes:
//   {{0, 1, 2, 3},{0, 1, 3, 4, 5},{0, 1, 2, 4, 5}}
//
// NOTE: This function is meant for testing, there is no need to call it
// directly.

StatusOr<std::array<std::vector<int64>, 3>> ParseEinsumString(
    absl::string_view einsum_config, int64 x_rank, int64 y_rank);

// If an einsum config does not contain an -> one will be added and the output
// config will be the sorted characters with any ellipsis at the beginning.
// Returns an empty string if the einsum string already has an ->.
std::string NormalizeEinsumString(absl::string_view einsum_config);

// Supports two operand einsum notation like "ab,cb->ac".
xla::XlaOp Einsum(
    xla::XlaOp x, xla::XlaOp y, absl::string_view einsum_config,
    xla::PrecisionConfig::Precision precision = xla::PrecisionConfig::DEFAULT);
xla::XlaOp Einsum(
    xla::XlaOp x, absl::string_view einsum_config,
    xla::PrecisionConfig::Precision precision = xla::PrecisionConfig::DEFAULT);


// Same as above but supporting numeric labels on dimensions. So "ab,cb->ac"
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

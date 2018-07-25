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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_TRIANGULAR_SOLVE_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_TRIANGULAR_SOLVE_H_

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace tensorflow {

// Solves systems of linear equations with lower or upper triangular coefficient
// matrices by forward- or back-substitution. Broadcasting along leading
// dimensions, this routine solves one of the matrix systems
//   `op(a) * x = b`,  or `x * op(a) = b`,
// for the variable `x` given `a` and `b`, where `op(a)` is either
//   `op(a) = a`,  or `op(a) = transpose(a)`,  or `op(a) = conj(transpose(a))`.
// That is, the innermost matrices in the output satisfy a scalar system
// depending on the value of the value of (left_side, transpose_a, conjugate_a)
// according to:
//   (F, F, F) => `output[..., i, k]  a[..., k, j] = b[..., i, j]`,
//   (F, F, T) => `output[..., i, k] a*[..., k, j] = b[..., i, j]`,
//   (F, T, F) => `output[..., i, k]  a[..., j, k] = b[..., i, j]`,
//   (F, T, T) => `output[..., i, k] a*[..., j, k] = b[..., i, j]`,
//   (T, F, F) => ` a[..., i, k] output[..., k, j] = b[..., i, j]`,
//   (T, F, T) => `a*[..., i, k] output[..., k, j] = b[..., i, j]`,
//   (T, T, F) => ` a[..., i, k] output[..., j, k] = b[..., i, j]`,
//   (T, T, T) => `a*[..., i, k] output[..., j, k] = b[..., i, j]`,
// where * denotes complex conjugation and where the index `k` is summed over.
//
// `a` is a tensor of shape `[..., M, M]` whose innermost 2 dimensions form
// square matrices. If lower is true (false), then the strictly upper (lower)
// triangular part of each innermost matrix in `a` is assumed to be zero and is
// not accessed.
// `b` is a tensor of shape `[..., M, K]` if left_side is true, otherwise a
// tensor of shape `[..., K, M]`.
// `left_side` is a boolean, indicating whether to solve a system of the form
// op(a) * x = b (true) or x * op(a) = b (false).
// `lower` is a boolean, indicating whether the argument `a` is lower-triangular
// (true) or upper-triangular (false).
// `transpose_a` is a boolean indicating whether the matrix `a` is transposed.
// `conjugate_a` is a boolean indicating whether the entries of `a` are complex
// conjugated (independently of whether they are transposed), so that when both
// transpose_a and conjugate_a are true the effect is a Hermitian adjoint.
//
// Uses a blocked algorithm if `block_size` is > 1; if block_size == 1 then no
// blocking is used.
xla::XlaOp TriangularSolve(xla::XlaOp a, xla::XlaOp b, bool left_side,
                           bool lower, bool transpose_a, bool conjugate_a,
                           int64 block_size = 128);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_TRIANGULAR_SOLVE_H_

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATVEC_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATVEC_H_

#include "third_party/eigen3/Eigen/Core"

#include "tensorflow/core/platform/types.h"

namespace xla {

namespace detail {

using tensorflow::int32;
using tensorflow::int64;

// Does mat * x or mat^T * x.
template <typename T>
void MatVec(T* out_buf, T* mat_buf, T* x_buf, int64 rows, int64 cols,
            int32 transpose) {
  // Use an Eigen Matrix instead of a Tensor, as the GEMV from Matrix seems to
  // be faster (b/30223679).  See also: the matmul op kernel in TensorFlow,
  // which implements the same optimization.
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using MatrixMap = Eigen::Map<Matrix>;

  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using VectorMap = Eigen::Map<Vector>;

  auto x = VectorMap(x_buf, cols);
  auto out = VectorMap(out_buf, rows);

  int64 mat_rows = rows;
  int64 mat_cols = cols;

  if (transpose) {
    std::swap(mat_rows, mat_cols);
  }

  auto mat = MatrixMap(mat_buf, mat_rows, mat_cols);

  if (transpose) {
    out = mat.transpose() * x;
  } else {
    out = mat * x;
  }
}

// Converts matmul-style args to matvec.
template <typename T>
void DispatchMatVec(T* out, T* lhs, T* rhs, int64 m, int64 n, int64 k,
                    int32 transpose_lhs, int32 transpose_rhs) {
  // If the input is in the form x * A, where x is the vector, then bring A back
  // over to the left hand side.  We make use of the identity
  //
  //   (x * A)^T = A^T * x^T
  //
  // We do not need to take the transpose of x or of the result since taking
  // the transpose of a vector does not change the memory layout.
  const int64 cols = k;

  T* mat;
  T* vec;
  int64 rows;
  bool transpose_mat;

  bool is_mat_vec = (n == 1);

  if (is_mat_vec) {
    mat = lhs;
    vec = rhs;
    rows = m;
    transpose_mat = transpose_lhs;
  } else {
    mat = rhs;
    vec = lhs;
    rows = n;
    transpose_mat = !transpose_rhs;
  }

  MatVec<T>(out, mat, vec, rows, cols, transpose_mat);
}

}  // namespace detail

// Performs a matrix-vector multiplication using Eigen. 'lhs' and 'rhs' are
// pointers to buffers containing input matrices in column-major order. 'out' is
// a pointer to a buffer sufficiently large to hold the result of the
// operation. Following standard nomenclature: lhs is m x k, rhs is k x n, and
// out is m x n.
//
// This requires that m = 1 or n = 1.
//
// TODO(b/64684907): Compare runtime performance of these functions with dot
// simplification.
template <typename T>
void EigenMatVec(T* out, T* lhs, T* rhs, tensorflow::int64 m,
                 tensorflow::int64 n, tensorflow::int64 k,
                 tensorflow::int32 transpose_lhs,
                 tensorflow::int32 transpose_rhs) {
  assert((m == 1 || n == 1) && "not a matrix-vector multiply");
  detail::DispatchMatVec<T>(out, lhs, rhs, m, n, k, transpose_lhs,
                            transpose_rhs);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATVEC_H_

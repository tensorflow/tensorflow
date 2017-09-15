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

#include <algorithm>
#include <cassert>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/compiler/xla/service/cpu/runtime_matvec.h"

using tensorflow::int32;
using tensorflow::int64;

namespace {

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

}  // namespace

namespace xla {

void EigenMatVecF32(float* out, float* lhs, float* rhs, int64 m, int64 n,
                    int64 k, int32 transpose_lhs, int32 transpose_rhs) {
  assert((m == 1 || n == 1) && "not a matrix-vector multiply");
  DispatchMatVec<float>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
}

void EigenMatVecF64(double* out, double* lhs, double* rhs, int64 m, int64 n,
                    int64 k, int32 transpose_lhs, int32 transpose_rhs) {
  assert((m == 1 || n == 1) && "not a matrix-vector multiply");
  DispatchMatVec<double>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
}

}  // namespace xla

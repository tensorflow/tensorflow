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

#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matvec.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::int32;
using tensorflow::int64;

namespace {

template <typename T>
void MatMul(const void* run_options_ptr, T* out, T* lhs, T* rhs, int64 m,
            int64 n, int64 k, int32 transpose_lhs, int32 transpose_rhs) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);

  int64 lhs_rows = m;
  int64 lhs_cols = k;
  if (transpose_lhs) {
    std::swap(lhs_rows, lhs_cols);
  }

  int64 rhs_rows = k;
  int64 rhs_cols = n;
  if (transpose_rhs) {
    std::swap(rhs_rows, rhs_cols);
  }

  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, Eigen::Aligned> A(
      lhs, lhs_rows, lhs_cols);
  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, Eigen::Aligned> B(
      rhs, rhs_rows, rhs_cols);
  Eigen::TensorMap<Eigen::Tensor<T, 2>, Eigen::Aligned> C(out, m, n);

  typedef typename Eigen::Tensor<T, 2>::DimensionPair DimPair;
  int lhs_contract_dim = transpose_lhs ? 0 : 1;
  int rhs_contract_dim = transpose_rhs ? 1 : 0;
  const Eigen::array<DimPair, 1> dims(
      {DimPair(lhs_contract_dim, rhs_contract_dim)});

  // Matrix multiply is a special case of the "contract" operation where
  // the contraction is performed along dimension 1 of the lhs and dimension
  // 0 of the rhs.
  C.device(*run_options->intra_op_thread_pool()) = A.contract(B, dims);
}

template <typename T>
void MatMulImpl(const void* run_options_ptr, T* out, T* lhs, T* rhs, int64 m,
                int64 n, int64 k, int32 transpose_lhs, int32 transpose_rhs) {
  if (m == 1 || n == 1) {
    // Despite being single threaded, this version of matrix * vector is faster.
    xla::EigenMatVec<T>(out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
  } else {
    MatMul<T>(run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs,
              transpose_rhs);
  }
}

}  // namespace

void __xla_cpu_runtime_EigenMatMulF16(const void* run_options_ptr,
                                      Eigen::half* out, Eigen::half* lhs,
                                      Eigen::half* rhs, int64 m, int64 n,
                                      int64 k, int32 transpose_lhs,
                                      int32 transpose_rhs) {
  MatMulImpl<Eigen::half>(run_options_ptr, out, lhs, rhs, m, n, k,
                          transpose_lhs, transpose_rhs);
}

void __xla_cpu_runtime_EigenMatMulF32(const void* run_options_ptr, float* out,
                                      float* lhs, float* rhs, int64 m, int64 n,
                                      int64 k, int32 transpose_lhs,
                                      int32 transpose_rhs) {
  MatMulImpl<float>(run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs,
                    transpose_rhs);
}

void __xla_cpu_runtime_EigenMatMulF64(const void* run_options_ptr, double* out,
                                      double* lhs, double* rhs, int64 m,
                                      int64 n, int64 k, int32 transpose_lhs,
                                      int32 transpose_rhs) {
  MatMulImpl<double>(run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs,
                     transpose_rhs);
}

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

#if defined(ENABLE_MKL) && !defined(INTEL_MKL_DNN_ONLY)
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.h"

#include "dnnl.hpp"
#include "tensorflow/compiler/xla/executable_run_options.h"

#define EIGEN_USE_THREADS
#include "absl/base/dynamic_annotations.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"

namespace {
// oneDNN API for 32-bit Matrix Multiplication.

// MatMul function is defined as: c = alpha * op(a) * op(b) + beta * c.
// Since XLA MatMul does not used alpha, beta, we set them to 1.0 and 0.0.
// Matrix lhs, rhs and out are all column-major.
void MatMulF32(const void* run_options_ptr, float* out, float* lhs, float* rhs,
               int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
               int32_t transpose_rhs) {
  const float alpha = 1.0f, beta = 0.0f;
  // lda, ldb, and ldc are the leading dimensions of matrices a, b, and c,
  // respectively. For column-major matrices, the leading dimension is the
  // stride between consecutive columns (which equals the number of rows). If
  // the matrix is transposed, the leading dimension is the stride between
  // consecutive rows (which equals the number of columns).
  int lda = transpose_lhs ? k : m;
  int ldb = transpose_rhs ? n : k;
  int ldc = m;
  char c_trans_a = transpose_lhs ? 'T' : 'N';
  char c_trans_b = transpose_rhs ? 'T' : 'N';

  dnnl_sgemm(c_trans_a, c_trans_b, m, n, k, alpha, lhs, lda, rhs, ldb, beta,
             out, ldc);
}

}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_MKLMatMulF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs, int64_t m,
    int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  // BLAS GEMM MatMul uses OpenMP for parallelization, so we pass the thread
  // number specified in intra_op_thread_pool to MKL.
  int prev_num_threads = mkl_set_num_threads_local(
      run_options->intra_op_thread_pool()->numThreads());
  MatMulF32(nullptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
  // Set thread number back to the previous number.
  mkl_set_num_threads_local(prev_num_threads);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_MKLSingleThreadedMatMulF32(const void* run_options_ptr,
                                             float* out, float* lhs, float* rhs,
                                             int64_t m, int64_t n, int64_t k,
                                             int32_t transpose_lhs,
                                             int32_t transpose_rhs) {
  // Set the thread number to 1 for single threaded execution.
  int prev_num_threads = mkl_set_num_threads_local(1);
  MatMulF32(nullptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
  // Set thread number back to the previous number.
  mkl_set_num_threads_local(prev_num_threads);
}
#endif  // ENABLE_MKL

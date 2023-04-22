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
#include "tensorflow/compiler/xla/service/cpu/runtime_lightweight_check.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/types.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace {

bool Is16BytesAligned(void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

template <typename T, Eigen::AlignmentType Alignment>
void MatMul(const void* run_options_ptr, T* out, T* lhs, T* rhs, int64_t m,
            int64_t n, int64_t k, tensorflow::int32 transpose_lhs,
            tensorflow::int32 transpose_rhs) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);

  int64_t lhs_rows = m;
  int64_t lhs_cols = k;
  if (transpose_lhs) {
    std::swap(lhs_rows, lhs_cols);
  }

  int64_t rhs_rows = k;
  int64_t rhs_cols = n;
  if (transpose_rhs) {
    std::swap(rhs_rows, rhs_cols);
  }

  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, Alignment> A(lhs, lhs_rows,
                                                                 lhs_cols);
  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, Alignment> B(rhs, rhs_rows,
                                                                 rhs_cols);
  Eigen::TensorMap<Eigen::Tensor<T, 2>, Alignment> C(out, m, n);

  typedef typename Eigen::Tensor<T, 2>::DimensionPair DimPair;
  int lhs_contract_dim = transpose_lhs ? 0 : 1;
  int rhs_contract_dim = transpose_rhs ? 1 : 0;
  const Eigen::array<DimPair, 1> dims(
      {DimPair(lhs_contract_dim, rhs_contract_dim)});

  // Matrix multiply is a special case of the "contract" operation where
  // the contraction is performed along dimension 1 of the lhs and dimension
  // 0 of the rhs.
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  C.device(*run_options->intra_op_thread_pool()) = A.contract(B, dims);
}

template <typename T>
void MatMulDispatch(const void* run_options_ptr, T* out, T* lhs, T* rhs,
                    int64_t m, int64_t n, int64_t k,
                    tensorflow::int32 transpose_lhs,
                    tensorflow::int32 transpose_rhs) {
  bool all_buffers_16b_aligned =
      Is16BytesAligned(out) && Is16BytesAligned(lhs) && Is16BytesAligned(rhs);

  if (!all_buffers_16b_aligned) {
    MatMul<T, Eigen::Unaligned>(run_options_ptr, out, lhs, rhs, m, n, k,
                                transpose_lhs, transpose_rhs);
    return;
  }

  MatMul<T, Eigen::Aligned16>(run_options_ptr, out, lhs, rhs, m, n, k,
                              transpose_lhs, transpose_rhs);
}

}  // namespace

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulF16(
    const void* run_options_ptr, Eigen::half* out, Eigen::half* lhs,
    Eigen::half* rhs, int64_t m, int64_t n, int64_t k,
    tensorflow::int32 transpose_lhs, tensorflow::int32 transpose_rhs) {
  MatMulDispatch<Eigen::half>(run_options_ptr, out, lhs, rhs, m, n, k,
                              transpose_lhs, transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs, int64_t m,
    int64_t n, int64_t k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs) {
  MatMulDispatch<float>(run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs,
                        transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulF64(
    const void* run_options_ptr, double* out, double* lhs, double* rhs,
    int64_t m, int64_t n, int64_t k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs) {
  MatMulDispatch<double>(run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs,
                         transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulC64(
    const void* run_options_ptr, std::complex<float>* out,
    std::complex<float>* lhs, std::complex<float>* rhs, int64_t m, int64_t n,
    int64_t k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs) {
  MatMulDispatch<std::complex<float>>(run_options_ptr, out, lhs, rhs, m, n, k,
                                      transpose_lhs, transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulC128(
    const void* run_options_ptr, std::complex<double>* out,
    std::complex<double>* lhs, std::complex<double>* rhs, int64_t m, int64_t n,
    int64_t k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs) {
  MatMulDispatch<std::complex<double>>(run_options_ptr, out, lhs, rhs, m, n, k,
                                       transpose_lhs, transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulS32(
    const void* run_options_ptr, tensorflow::int32* out, tensorflow::int32* lhs,
    tensorflow::int32* rhs, int64_t m, int64_t n, int64_t k,
    tensorflow::int32 transpose_lhs, tensorflow::int32 transpose_rhs) {
  MatMulDispatch<tensorflow::int32>(run_options_ptr, out, lhs, rhs, m, n, k,
                                    transpose_lhs, transpose_rhs);
}

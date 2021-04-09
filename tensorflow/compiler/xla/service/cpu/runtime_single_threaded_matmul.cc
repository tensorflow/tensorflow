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

#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
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
void MatMul(const void* run_options_ptr, T* out, T* lhs, T* rhs,
            tensorflow::int64 m, tensorflow::int64 n, tensorflow::int64 k,
            tensorflow::int32 transpose_lhs, tensorflow::int32 transpose_rhs) {
  tensorflow::int64 lhs_rows = m;
  tensorflow::int64 lhs_cols = k;
  if (transpose_lhs) {
    std::swap(lhs_rows, lhs_cols);
  }

  tensorflow::int64 rhs_rows = k;
  tensorflow::int64 rhs_cols = n;
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
  C = A.contract(B, dims);
}

template <typename T>
void SingleThreadedMatMulDispatch(const void* run_options_ptr, T* out, T* lhs,
                                  T* rhs, tensorflow::int64 m,
                                  tensorflow::int64 n, tensorflow::int64 k,
                                  tensorflow::int32 transpose_lhs,
                                  tensorflow::int32 transpose_rhs) {
  bool all_buffers_16b_aligned =
      Is16BytesAligned(out) && Is16BytesAligned(lhs) && Is16BytesAligned(rhs);

  if (!all_buffers_16b_aligned) {
    MatMul<T, Eigen::Unaligned>(run_options_ptr, out, lhs, rhs, m, n, k,
                                transpose_lhs, transpose_rhs);
  }

  MatMul<T, Eigen::Aligned16>(run_options_ptr, out, lhs, rhs, m, n, k,
                              transpose_lhs, transpose_rhs);
}

}  // namespace

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulF16(
    const void* run_options_ptr, Eigen::half* out, Eigen::half* lhs,
    Eigen::half* rhs, tensorflow::int64 m, tensorflow::int64 n,
    tensorflow::int64 k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs) {
  SingleThreadedMatMulDispatch<Eigen::half>(run_options_ptr, out, lhs, rhs, m,
                                            n, k, transpose_lhs, transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs,
    tensorflow::int64 m, tensorflow::int64 n, tensorflow::int64 k,
    tensorflow::int32 transpose_lhs, tensorflow::int32 transpose_rhs) {
  SingleThreadedMatMulDispatch<float>(run_options_ptr, out, lhs, rhs, m, n, k,
                                      transpose_lhs, transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulF64(
    const void* run_options_ptr, double* out, double* lhs, double* rhs,
    tensorflow::int64 m, tensorflow::int64 n, tensorflow::int64 k,
    tensorflow::int32 transpose_lhs, tensorflow::int32 transpose_rhs) {
  SingleThreadedMatMulDispatch<double>(run_options_ptr, out, lhs, rhs, m, n, k,
                                       transpose_lhs, transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulC64(
    const void* run_options_ptr, std::complex<float>* out,
    std::complex<float>* lhs, std::complex<float>* rhs, tensorflow::int64 m,
    tensorflow::int64 n, tensorflow::int64 k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs) {
  SingleThreadedMatMulDispatch<std::complex<float>>(
      run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulC128(
    const void* run_options_ptr, std::complex<double>* out,
    std::complex<double>* lhs, std::complex<double>* rhs, tensorflow::int64 m,
    tensorflow::int64 n, tensorflow::int64 k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs) {
  SingleThreadedMatMulDispatch<std::complex<double>>(
      run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_EigenSingleThreadedMatMulS32(
    const void* run_options_ptr, tensorflow::int32* out, tensorflow::int32* lhs,
    tensorflow::int32* rhs, tensorflow::int64 m, tensorflow::int64 n,
    tensorflow::int64 k, tensorflow::int32 transpose_lhs,
    tensorflow::int32 transpose_rhs) {
  SingleThreadedMatMulDispatch<tensorflow::int32>(
      run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
}

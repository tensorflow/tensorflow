/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_MATMUL_COMMON_H_
#define XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_MATMUL_COMMON_H_

#include <cstdint>

#include "absl/base/attributes.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "xla/tsl/framework/contraction/eigen_contraction_kernel.h"
#endif

namespace xla {

static inline bool Is16BytesAligned(void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

template <typename T, Eigen::AlignmentType Alignment>
void SingleThreadedMatMul(const void* run_options_ptr, T* out, T* lhs, T* rhs,
                          int64_t m, int64_t n, int64_t k,
                          int32_t transpose_lhs, int32_t transpose_rhs) {
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
  C = A.contract(B, dims);
}

template <typename T>
void SingleThreadedMatMulDispatch(const void* run_options_ptr, T* out, T* lhs,
                                  T* rhs, int64_t m, int64_t n, int64_t k,
                                  int32_t transpose_lhs,
                                  int32_t transpose_rhs) {
  bool all_buffers_16b_aligned =
      Is16BytesAligned(out) && Is16BytesAligned(lhs) && Is16BytesAligned(rhs);

  if (!all_buffers_16b_aligned) {
    SingleThreadedMatMul<T, Eigen::Unaligned>(
        run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
  }

  SingleThreadedMatMul<T, Eigen::Aligned16>(run_options_ptr, out, lhs, rhs, m,
                                            n, k, transpose_lhs, transpose_rhs);
}

}  // namespace xla

#endif  // XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_MATMUL_COMMON_H_

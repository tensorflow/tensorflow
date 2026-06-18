/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_DOT_LIB_H_
#define XLA_BACKENDS_CPU_RUNTIME_DOT_LIB_H_

#include <array>
#include <cstdint>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/functional/any_invocable.h"

#define EIGEN_USE_THREADS
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu::internal {

// Done callback is called when the MatMul computation is complete.
using DoneCallback = absl::AnyInvocable<void()>;

// Col-major x Col-major MatMul implementation as Eigen contraction.
template <typename LhsType, typename RhsType, typename OutType,
          Eigen::AlignmentType alignment>
void MatMul(const Eigen::ThreadPoolDevice* device, OutType* out, LhsType* lhs,
            RhsType* rhs, int64_t m, int64_t n, int64_t k,
            int32_t transpose_lhs, int32_t transpose_rhs, DoneCallback done);

// Col-major x Col-major MatMul implementation as Eigen contraction.
template <typename LhsType, typename RhsType, typename OutType>
void TypedMatMul(const Eigen::ThreadPoolDevice* device, void* out, void* lhs,
                 void* rhs, int64_t m, int64_t n, int64_t k, bool transpose_lhs,
                 bool transpose_rhs, DoneCallback done);

//===----------------------------------------------------------------------===//
// TypedMatMul/MatMul implementation details.
//===----------------------------------------------------------------------===//

template <typename LhsType, typename RhsType, typename OutType,
          Eigen::AlignmentType alignment>
void MatMul(const Eigen::ThreadPoolDevice* device, OutType* out, LhsType* lhs,
            RhsType* rhs, int64_t m, int64_t n, int64_t k,
            int32_t transpose_lhs, int32_t transpose_rhs, DoneCallback done) {
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

  const Eigen::TensorMap<Eigen::Tensor<const LhsType, 2>, alignment> a(
      lhs, lhs_rows, lhs_cols);
  const Eigen::TensorMap<Eigen::Tensor<const RhsType, 2>, alignment> b(
      rhs, rhs_rows, rhs_cols);
  Eigen::TensorMap<Eigen::Tensor<OutType, 2>, alignment> c(out, m, n);

  typedef typename Eigen::Tensor<LhsType, 2>::DimensionPair DimPair;
  int lhs_contract_dim = transpose_lhs ? 0 : 1;
  int rhs_contract_dim = transpose_rhs ? 1 : 0;

  std::array<DimPair, 1> dims({DimPair(lhs_contract_dim, rhs_contract_dim)});

  if (device != nullptr) {
    c.device(*device, std::move(done)) =
        a.contract(b, dims).template cast<OutType>();
  } else {
    c = a.contract(b, dims).template cast<OutType>();
    done();
  }
}

template <typename LhsType, typename RhsType, typename OutType>
void TypedMatMul(const Eigen::ThreadPoolDevice* device, void* out, void* lhs,
                 void* rhs, int64_t m, int64_t n, int64_t k, bool transpose_lhs,
                 bool transpose_rhs, DoneCallback done) {
  auto is_16_byte_aligned = [](void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };

  bool is_aligned = is_16_byte_aligned(lhs) && is_16_byte_aligned(rhs) &&
                    is_16_byte_aligned(out);

  if (ABSL_PREDICT_TRUE(is_aligned)) {
    MatMul<LhsType, RhsType, OutType, Eigen::Aligned16>(
        device, static_cast<OutType*>(out), static_cast<LhsType*>(lhs),
        static_cast<RhsType*>(rhs), m, n, k, transpose_lhs, transpose_rhs,
        std::move(done));
  } else {
    MatMul<LhsType, RhsType, OutType, Eigen::Unaligned>(
        device, static_cast<OutType*>(out), static_cast<LhsType*>(lhs),
        static_cast<RhsType*>(rhs), m, n, k, transpose_lhs, transpose_rhs,
        std::move(done));
  }
}

// Declare TypedMatMul template for all supported data types to enable
// parallel compilation.
#define DECLARE_TYPED_MATMUL(T)                                                \
  extern template void TypedMatMul<T, T, T>(                                   \
      const Eigen::ThreadPoolDevice* device, void* out, void* lhs, void* rhs,  \
      int64_t m, int64_t n, int64_t k, bool transpose_lhs, bool transpose_rhs, \
      DoneCallback done)

DECLARE_TYPED_MATMUL(Eigen::half);
DECLARE_TYPED_MATMUL(float);
DECLARE_TYPED_MATMUL(double);
DECLARE_TYPED_MATMUL(int32_t);
DECLARE_TYPED_MATMUL(std::complex<float>);
DECLARE_TYPED_MATMUL(std::complex<double>);

#define DECLARE_MIXED_MATMUL(LhsType, RhsType, OutType)                        \
  extern template void TypedMatMul<LhsType, RhsType, OutType>(                 \
      const Eigen::ThreadPoolDevice* device, void* out, void* lhs, void* rhs,  \
      int64_t m, int64_t n, int64_t k, bool transpose_lhs, bool transpose_rhs, \
      DoneCallback done)

DECLARE_MIXED_MATMUL(int8_t, int8_t, int32_t);

#undef DECLARE_TYPED_MATMUL

}  // namespace xla::cpu::internal

#endif  // XLA_BACKENDS_CPU_RUNTIME_DOT_LIB_H_

/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_DOT_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_DOT_THUNK_H_

#include "xla/backends/cpu/runtime/dot_lib.h"
#define EIGEN_USE_THREADS

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class DotThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<DotThunk>> Create(
      Info info, DotDimensionNumbers dot_dimensions,
      BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
      BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
      BufferAllocation::Slice out_buffer, Shape out_shape);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final { return DotBufferUses(dot_slices_); }

 private:
  DotThunk(Info info, DotDimensionNumbers dot_dimensions, DotSlices dot_slices,
           DotShape dot_shape, DotCanonicalDims dot_canonical_dims);

  using DoneCallback = absl::AnyInvocable<void()>;

  // Col-major x Col-major MatMul implementation as Eigen contraction.
  template <typename T, Eigen::AlignmentType alignment>
  static void MatMul(const Eigen::ThreadPoolDevice* device, T* out, T* lhs,
                     T* rhs, int64_t m, int64_t n, int64_t k,
                     int32_t transpose_lhs, int32_t transpose_rhs,
                     DoneCallback done);

  template <typename T>
  static void TypedMatMul(const Eigen::ThreadPoolDevice* device, void* out,
                          void* lhs, void* rhs, int64_t m, int64_t n, int64_t k,
                          bool transpose_lhs, bool transpose_rhs,
                          DoneCallback done);

  DotDimensionNumbers dot_dimensions_;
  DotSlices dot_slices_;
  DotShape dot_shape_;
  DotCanonicalDims dot_canonical_dims_;

  // Contracting dimensions of the LHS and RHS matmul shapes.
  absl::InlinedVector<int64_t, 2> lhs_matmul_contracting_dims_;
  absl::InlinedVector<int64_t, 2> rhs_matmul_contracting_dims_;
};

//===----------------------------------------------------------------------===//
// DotThunk implementation details.
//===----------------------------------------------------------------------===//

template <typename T, Eigen::AlignmentType alignment>
void DotThunk::MatMul(const Eigen::ThreadPoolDevice* device, T* out, T* lhs,
                      T* rhs, int64_t m, int64_t n, int64_t k,
                      int32_t transpose_lhs, int32_t transpose_rhs,
                      DoneCallback done) {
  int64_t lhs_rows = m;
  int64_t lhs_cols = k;
  if (transpose_lhs) std::swap(lhs_rows, lhs_cols);

  int64_t rhs_rows = k;
  int64_t rhs_cols = n;
  if (transpose_rhs) std::swap(rhs_rows, rhs_cols);

  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, alignment> a(lhs, lhs_rows,
                                                                 lhs_cols);
  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, alignment> b(rhs, rhs_rows,
                                                                 rhs_cols);
  Eigen::TensorMap<Eigen::Tensor<T, 2>, alignment> c(out, m, n);

  typedef typename Eigen::Tensor<T, 2>::DimensionPair DimPair;
  int lhs_contract_dim = transpose_lhs ? 0 : 1;
  int rhs_contract_dim = transpose_rhs ? 1 : 0;
  std::array<DimPair, 1> dims({DimPair(lhs_contract_dim, rhs_contract_dim)});

  c.device(*device, std::move(done)) = a.contract(b, dims);
}

template <typename T>
void DotThunk::TypedMatMul(const Eigen::ThreadPoolDevice* device, void* out,
                           void* lhs, void* rhs, int64_t m, int64_t n,
                           int64_t k, bool transpose_lhs, bool transpose_rhs,
                           DoneCallback done) {
  auto is_16_byte_aligned = [](void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };

  bool is_aligned = is_16_byte_aligned(lhs) && is_16_byte_aligned(rhs) &&
                    is_16_byte_aligned(out);

  if (ABSL_PREDICT_TRUE(is_aligned)) {
    MatMul<T, Eigen::Aligned16>(device, static_cast<T*>(out),
                                static_cast<T*>(lhs), static_cast<T*>(rhs), m,
                                n, k, transpose_lhs, transpose_rhs,
                                std::move(done));
  } else {
    MatMul<T, Eigen::Unaligned>(device, static_cast<T*>(out),
                                static_cast<T*>(lhs), static_cast<T*>(rhs), m,
                                n, k, transpose_lhs, transpose_rhs,
                                std::move(done));
  }
}

// Extern DotThunk::TypedMatMul template for all supported data types to enable
// parallel compilation.
#define DOT_THUNK_EXTERN_MATMUL_TEMPLATE(T)                                    \
  extern template void DotThunk::TypedMatMul<T>(                               \
      const Eigen::ThreadPoolDevice* device, void* out, void* lhs, void* rhs,  \
      int64_t m, int64_t n, int64_t k, bool transpose_lhs, bool transpose_rhs, \
      DoneCallback done)

DOT_THUNK_EXTERN_MATMUL_TEMPLATE(Eigen::half);
DOT_THUNK_EXTERN_MATMUL_TEMPLATE(float);
DOT_THUNK_EXTERN_MATMUL_TEMPLATE(double);
DOT_THUNK_EXTERN_MATMUL_TEMPLATE(int32_t);
DOT_THUNK_EXTERN_MATMUL_TEMPLATE(std::complex<float>);
DOT_THUNK_EXTERN_MATMUL_TEMPLATE(std::complex<double>);

#undef DOT_THUNK_EXTERN_MATMUL_TEMPLATE

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_DOT_THUNK_H_

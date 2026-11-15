/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_ONEMKL_BLAS_H_
#define XLA_STREAM_EXECUTOR_SYCL_ONEMKL_BLAS_H_

#include "oneapi/mkl.hpp"
#include "absl/status/status_macros.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/onemkl_util.h"

namespace stream_executor::sycl {

// Helper functions transforming blas arguments into oneMKL arguments.
inline oneapi::mkl::transpose ToMklTranspose(blas::Transpose trans) {
  if (trans == blas::Transpose::kTranspose)
    return oneapi::mkl::transpose::trans;
  else if (trans == blas::Transpose::kNoTranspose)
    return oneapi::mkl::transpose::nontrans;
  else
    return oneapi::mkl::transpose::conjtrans;
}

template <typename CType>
absl::Status ComplexMatmulImpl(const gpu::GemmConfig& config, Stream* stream,
                               const gpu::BlasLt::MemoryArgs& args) {
  gpu::MatrixLayout lhs_layout = config.lhs_layout;
  gpu::MatrixLayout rhs_layout = config.rhs_layout;
  gpu::MatrixLayout out_layout = config.output_layout;
  gpu::MatrixLayout c_layout = config.c_layout;
  // cublasLt matmul requires batch sizes to be equal. If only one operand has a
  // batch, the other will be broadcast (as its batch_stride == 0).
  size_t batch_size = std::max(lhs_layout.batch_size, rhs_layout.batch_size);
  lhs_layout.batch_size = batch_size;
  rhs_layout.batch_size = batch_size;

  DeviceAddressBase a_buf = args.a;
  DeviceAddressBase b_buf = args.b;

  // oneMKL default is column major
  bool must_swap_operands =
      gpu::MakeOutputColumnMajor(lhs_layout, rhs_layout, out_layout, &c_layout);
  if (must_swap_operands) {
    std::swap(a_buf, b_buf);
  }

  auto trans_a = (lhs_layout.order == gpu::MatrixLayout::Order::kColumnMajor
                      ? blas::Transpose::kNoTranspose
                      : blas::Transpose::kTranspose);
  auto trans_b = (rhs_layout.order == gpu::MatrixLayout::Order::kColumnMajor
                      ? blas::Transpose::kNoTranspose
                      : blas::Transpose::kTranspose);
  const uint64_t m = out_layout.num_rows;
  const uint64_t n = out_layout.num_cols;
  const uint64_t k = lhs_layout.num_cols;
  const int lda = static_cast<int>(lhs_layout.leading_dim_stride);
  const int ldb = static_cast<int>(rhs_layout.leading_dim_stride);
  const int ldc = static_cast<int>(out_layout.leading_dim_stride);

  // oneMKL overwrites the buffer c with the results. If c and d are not the
  // same, initialize d with c. Initialization is not needed if beta == 0.0
  DeviceAddressBase out_buf = args.d;
  if (config.beta != 0.0 && args.c.opaque() != args.d.opaque()) {
    ABSL_RETURN_IF_ERROR(stream->Memcpy(&out_buf, args.c, out_buf.size()));
  }
  ::sycl::queue* stream_queue =
      absl::bit_cast<::sycl::queue*>(stream->platform_specific_handle().stream);

  auto a = reinterpret_cast<const CType*>(a_buf.opaque());
  auto b = reinterpret_cast<const CType*>(b_buf.opaque());
  auto c = reinterpret_cast<CType*>(out_buf.opaque());
  CType alpha(config.alpha.real(), config.alpha.imag());
  CType beta(config.beta);
  absl::StatusOr<::sycl::event> status =
      (batch_size != 1) ? ExecMklFunc([&] {
        return oneapi::mkl::blas::gemm_batch(
            *stream_queue, ToMklTranspose(trans_a), ToMklTranspose(trans_b), m,
            n, k, alpha, a, lda, lhs_layout.batch_stride, b, ldb,
            rhs_layout.batch_stride, beta, c, ldc, out_layout.batch_stride,
            batch_size);
      })
                        : ExecMklFunc([&] {
                            return oneapi::mkl::blas::gemm(
                                *stream_queue, ToMklTranspose(trans_a),
                                ToMklTranspose(trans_b), m, n, k, alpha, a, lda,
                                b, ldb, beta, c, ldc);
                          });
  return status.status();
}
}  // namespace stream_executor::sycl
#endif  // XLA_STREAM_EXECUTOR_SYCL_ONEMKL_BLAS_H_

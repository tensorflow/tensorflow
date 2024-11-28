/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_ADAPTOR_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_ADAPTOR_H_

#include <cstdint>
#include <type_traits>

#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_blas_lt_gemm_runner.h"

namespace stream_executor::gpu {

template <typename TBlasSupport>
struct GpuBlasLtAdaptor final : TBlasSupport {
  static_assert(std::is_base_of_v<blas::BlasSupport, TBlasSupport>,
                "TBlasSupport shall implement the BlasSupport interface");

  template <typename... TArgs>
  explicit GpuBlasLtAdaptor(TArgs... args) : TBlasSupport{args...} {}

  absl::Status DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64_t m, uint64 n,
                          uint64_t k, blas::DataType dtype, const void *alpha,
                          const DeviceMemoryBase &a, int lda,
                          const DeviceMemoryBase &b, int ldb, const void *beta,
                          DeviceMemoryBase *c, int ldc,
                          const NumericOptions &numeric_options,
                          blas::CallContext context,
                          ScratchAllocator *scratch_allocator) {
    if (IsGpuBlasLtEnabled()) {
      auto &runner = gpu::BlasLtGemmRunner::i(stream);
      switch (dtype) {
        case blas::DataType::kFloat:
          return DoBlasGemmImpl<float>(
              stream, transa, transb, m, n, k, dtype, alpha, a, lda, b, ldb,
              beta, c, ldc, numeric_options, context, scratch_allocator);
        case blas::DataType::kBF16:
          return DoBlasGemmImpl<Eigen::bfloat16>(
              stream, transa, transb, m, n, k, dtype, alpha, a, lda, b, ldb,
              beta, c, ldc, numeric_options, context, scratch_allocator);
        default:
          return absl::FailedPreconditionError("Unknown type");
      };
    } else {
      return TBlasSupport::DoBlasGemm(
          stream, transa, transb, m, n, k, dtype, alpha, a, lda, b, ldb, beta,
          c, ldc, numeric_options, context, scratch_allocator);
    }
  }
  // bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
  // blas::Transpose transb, uint64_t m, uint64_t n,
  // uint64 k, float alpha, DeviceMemorySlice<float> a,
  // int lda, DeviceMemorySlice<float> b, int ldb,
  // float beta, DeviceMemorySlice<float> c, int ldc,
  // int batch_count, const NumericOptions &numeric_options,
  // ScratchAllocator *scratch_allocator,
  // blas::CallContext context) override {
  // if (IsGpuBlasLtEnabled()) {
  // auto &runner = gpu::BlasLtGemmRunner::i(stream);
  // return CheckStatus(runner.Run(*stream, transa, transb, m, n, k, alpha, a,
  // lda, b, ldb, beta, c, ldc,
  // scratch_allocator));
  //} else {
  // return TBlasSupport::DoBlasGemmBatched(
  // stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
  // batch_count, numeric_options, scratch_allocator, context);
  //}
  //}

 private:
  template <typename T>
  absl::Status DoBlasGemmImpl(Stream *stream, blas::Transpose transa,
                              blas::Transpose transb, uint64_t m, uint64 n,
                              uint64_t k, blas::DataType dtype,
                              const void *alpha, const DeviceMemoryBase &a,
                              int lda, const DeviceMemoryBase &b, int ldb,
                              const void *beta, DeviceMemoryBase *c, int ldc,
                              const NumericOptions &numeric_options,
                              blas::CallContext context,
                              ScratchAllocator *scratch_allocator) {
    auto &runner = gpu::BlasLtGemmRunner::i(stream);
    auto alpha_v = *static_cast<const T *>(alpha);
    auto beta_v = *static_cast<const T *>(beta);
    return runner.Run(
        *stream, transa, transb, m, n, k, alpha_v,
        static_cast<DeviceMemory<T>>(a), lda, static_cast<DeviceMemory<T>>(b),
        ldb, beta_v, static_cast<DeviceMemory<T> *>(c), ldc, scratch_allocator);
  }

  bool CheckStatus(absl::Status status) {
    if (status.code() == absl::StatusCode::kOk) return true;
    LOG(ERROR) << status;
    return false;
  }

  bool IsGpuBlasLtEnabled() { return false; }
};

}  // namespace stream_executor::gpu
   //
#endif

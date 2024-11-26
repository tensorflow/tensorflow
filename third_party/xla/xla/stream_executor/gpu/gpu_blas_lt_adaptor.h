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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_H_

#include <cstdint>
#include <utility>

#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_blas_lt_gemm_runner.h"

namespace stream_executor::gpu {

template <typename TBlasSupport>
struct GpuBlasLtAdaptor final : TBlasSupport {
  template <typename... TArgs>
  explicit GpuBlasLtAdaptor(TArgs... args)
      : TBlasSupport{std::forward(args)...} {}

  bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64 k, float alpha, DeviceMemorySlice<float> a,
                         int lda, DeviceMemorySlice<float> b, int ldb,
                         float beta, DeviceMemorySlice<float> c, int ldc,
                         int batch_count, const NumericOptions &numeric_options,
                         ScratchAllocator *scratch_allocator,
                         blas::CallContext context) override {
    if (IsGpuBlasLtEnabled()) {
      const auto &runner = gpu::BlasLtGemmRunner::i(stream);
      return CheckStatus(runner.Run(*this, transa, transb, m, n, k, alpha, a,
                                    lda, b, ldb, beta, c, ldc, batch_count,
                                    scratch_allocator));
    } else {
      return TBlasSupport::DoBlasGemmBatched(
          stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
          batch_count, numeric_options, scratch_allocator, context);
    }
  }

 private:
  bool CheckStatus(absl::Status status) {
    return status.code() == absl::StatusCode::kOk;
  }

  bool IsGpuBlasLtEnabled() { return false; }
};
}  // namespace stream_executor::gpu
   //
#endif

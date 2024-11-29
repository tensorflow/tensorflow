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
#include "xla/debug_options_flags.h"

namespace stream_executor::gpu {
class WorkspaceScratchAllocator : public tensorflow::se::ScratchAllocator {
 public:
  using Stream = tensorflow::se::Stream;
  using DeviceMemoryBytes = tensorflow::se::DeviceMemory<uint8>;

  explicit WorkspaceScratchAllocator(DeviceMemoryBase workspace)
      : workspace_{workspace}, bytes_allocated_{} {}

  int64_t GetMemoryLimitInBytes() override { return workspace_.size(); }
  tsl::StatusOr<DeviceMemoryBytes> AllocateBytes(int64_t byte_size) override {
    if ((workspace_.size() - bytes_allocated_) < byte_size) {
      return tsl::Status{absl::StatusCode::kUnavailable,
                         absl::StrCat("Requested memory size (", byte_size,
                                      ") exceeds the memory limit (",
                                      workspace_.size(), ").")};
    }

    auto result = workspace_.GetSlice(bytes_allocated_, byte_size);
    bytes_allocated_ += byte_size;
    return result;
  }

 private:
  DeviceMemoryBytes workspace_;
  int64_t bytes_allocated_;
};

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
                          blas::CallContext context) override {
    if (IsGpuBlasLtEnabled()) {
      auto &runner = gpu::BlasLtGemmRunner::i(stream);
      auto workspace = TBlasSupport::GetWorkspace();
      WorkspaceScratchAllocator allocator{*workspace};
      switch (dtype) {
        case blas::DataType::kFloat:
          return DoBlasGemmImpl<float>(stream, transa, transb, m, n, k, dtype,
                                       alpha, a, lda, b, ldb, beta, c, ldc,
                                       numeric_options, context, &allocator);
        case blas::DataType::kBF16:
          return DoBlasGemmImpl<Eigen::bfloat16>(
              stream, transa, transb, m, n, k, dtype, alpha, a, lda, b, ldb,
              beta, c, ldc, numeric_options, context, &allocator);
        default:
          return absl::FailedPreconditionError("Unknown type");
      };
    } else {
      return TBlasSupport::DoBlasGemm(stream, transa, transb, m, n, k, dtype,
                                      alpha, a, lda, b, ldb, beta, c, ldc,
                                      numeric_options, context);
    }
  }

  bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64 k, float alpha, DeviceMemorySlice<float> a,
                         int lda, DeviceMemorySlice<float> b, int ldb,
                         float beta, DeviceMemorySlice<float> c, int ldc,
                         int batch_count, const NumericOptions &numeric_options,
                         ScratchAllocator *scratch_allocator,
                         blas::CallContext context) override {
    return DoBlasGemmBatchedImpl<float>(
        stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_count, numeric_options, scratch_allocator, context);
  }

  bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64 k, double alpha, DeviceMemorySlice<double> a,
                         int lda, DeviceMemorySlice<double> b, int ldb,
                         double beta, DeviceMemorySlice<double> c, int ldc,
                         int batch_count, const NumericOptions &numeric_options,
                         ScratchAllocator *scratch_allocator,
                         blas::CallContext context) override {
    return DoBlasGemmBatchedImpl<double>(
        stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_count, numeric_options, scratch_allocator, context);
  }

  bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64 k, float alpha,
                         DeviceMemorySlice<Eigen::bfloat16> a, int lda,
                         DeviceMemorySlice<Eigen::bfloat16> b, int ldb,
                         float beta, DeviceMemorySlice<Eigen::bfloat16> c,
                         int ldc, int batch_count,
                         const NumericOptions &numeric_options,
                         ScratchAllocator *scratch_allocator,
                         blas::CallContext context) override {
    return DoBlasGemmBatchedImpl<Eigen::bfloat16>(
        stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_count, numeric_options, scratch_allocator, context);
  }

  bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64 k, float alpha,
                         DeviceMemorySlice<Eigen::half> a, int lda,
                         DeviceMemorySlice<Eigen::half> b, int ldb, float beta,
                         DeviceMemorySlice<Eigen::half> c, int ldc,
                         int batch_count, const NumericOptions &numeric_options,
                         ScratchAllocator *scratch_allocator,
                         blas::CallContext context) override {
    return DoBlasGemmBatchedImpl<Eigen::half>(
        stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_count, numeric_options, scratch_allocator, context);
  }

  bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64 k, std::complex<float> alpha,
                         DeviceMemorySlice<std::complex<float>> a, int lda,
                         DeviceMemorySlice<std::complex<float>> b, int ldb,
                         std::complex<float> beta,
                         DeviceMemorySlice<std::complex<float>> c, int ldc,
                         int batch_count, const NumericOptions &numeric_options,
                         ScratchAllocator *scratch_allocator,
                         blas::CallContext context) override {
    return DoBlasGemmBatchedImpl<std::complex<float>>(
        stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_count, numeric_options, scratch_allocator, context);
  }

  bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64 k, std::complex<double> alpha,
                         DeviceMemorySlice<std::complex<double>> a, int lda,
                         DeviceMemorySlice<std::complex<double>> b, int ldb,
                         std::complex<double> beta,
                         DeviceMemorySlice<std::complex<double>> c, int ldc,
                         int batch_count, const NumericOptions &numeric_options,
                         ScratchAllocator *scratch_allocator,
                         blas::CallContext context) override {
    return DoBlasGemmBatchedImpl<std::complex<double>>(
        stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_count, numeric_options, scratch_allocator, context);
  }

  absl::Status DoBlasGemmStridedBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, blas::DataType dtype, const void *alpha,
      const DeviceMemoryBase &a, int lda, int64_t stride_a,
      const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
      DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
      const NumericOptions &numeric_options,
      blas::CallContext context) override {
    if (IsGpuBlasLtEnabled()) {
      auto &runner = gpu::BlasLtGemmRunner::i(stream);
      switch (dtype) {
        case blas::DataType::kFloat:
          return DoBlasGemmStridedBatchedImpl<float>(
              stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
              stride_b, beta, c, ldc, stride_c, batch_count, numeric_options,
              context);
        case blas::DataType::kBF16:
          return DoBlasGemmStridedBatchedImpl<Eigen::bfloat16>(
              stream, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
              stride_b, beta, c, ldc, stride_c, batch_count, numeric_options,
              context);
        default:
          return absl::FailedPreconditionError("Unknown type");
      };
    } else {
      return TBlasSupport::DoBlasGemmStridedBatched(
          stream, transa, transb, m, n, k, dtype, alpha, a, lda, stride_a, b,
          ldb, stride_b, beta, c, ldc, stride_c, batch_count, numeric_options,
          context);
    }
  }

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
    auto memory_c = DeviceMemory<T>(*c);
    return runner.Run(*stream, transa, transb, m, n, k, alpha_v,
                      DeviceMemory<T>(a), lda, DeviceMemory<T>(b), ldb, beta_v,
                      &memory_c, ldc, scratch_allocator);
  }

  template <typename T>
  absl::Status DoBlasGemmStridedBatchedImpl(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64_t k, const void *alpha,
      const DeviceMemoryBase &a, int lda, int64_t stride_a,
      const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
      DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
      const NumericOptions &numeric_options, blas::CallContext context) {
    auto workspace = TBlasSupport::GetWorkspace();
    WorkspaceScratchAllocator allocator{*workspace};
    auto &runner = gpu::BlasLtGemmRunner::i(stream);
    auto memory_c = DeviceMemory<T>(*c);
    auto alpha_v = *static_cast<const T *>(alpha);
    auto beta_v = *static_cast<const T *>(beta);
    return runner.RunStridedBatched(
        *stream, transa, transb, m, n, k, alpha_v, DeviceMemory<T>(a), lda,
        stride_a, DeviceMemory<T>(b), ldb, stride_b, beta_v, &memory_c, ldc,
        stride_c, batch_count, &allocator);
  }

  bool CheckStatus(absl::Status status) {
    if (status.code() == absl::StatusCode::kOk) return true;
    LOG(ERROR) << status;
    return false;
  }

  template <typename T, typename TScaler>
  bool DoBlasGemmBatchedImpl(Stream *stream, blas::Transpose transa,
                             blas::Transpose transb, uint64_t m, uint64_t n,
                             uint64 k, TScaler alpha, DeviceMemorySlice<T> a,
                             int lda, DeviceMemorySlice<T> b, int ldb,
                             TScaler beta, DeviceMemorySlice<T> c, int ldc,
                             int batch_count,
                             const NumericOptions &numeric_options,
                             ScratchAllocator *scratch_allocator,
                             blas::CallContext context) {
    if (IsGpuBlasLtEnabled()) {
      auto &runner = gpu::BlasLtGemmRunner::i(stream);
      return CheckStatus(runner.RunBatched(*stream, transa, transb, m, n, k,
                                           alpha, a, lda, b, ldb, beta, c, ldc,
                                           batch_count, scratch_allocator));
    } else {
      return TBlasSupport::DoBlasGemmBatched(
          stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
          batch_count, numeric_options, scratch_allocator, context);
    }
  }

  bool IsGpuBlasLtEnabled() {
    static const std::atomic_bool result =
        xla::GetDebugOptionsFromFlags().xla_gpu_enable_cublaslt();
    return result;
  }
};

}  // namespace stream_executor::gpu
   //
#endif

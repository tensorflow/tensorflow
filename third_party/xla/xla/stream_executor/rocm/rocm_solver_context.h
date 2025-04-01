/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_SOLVER_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_SOLVER_CONTEXT_H_

#include <complex>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#define TENSORFLOW_USE_HIPSOLVER (TF_ROCM_VERSION >= 40500)
#define TENSORFLOW_USE_ROCSOLVER (TF_ROCM_VERSION < 40500)

#include "rocm/rocm_config.h"
// Macros to ease the transition from rocsolver to hipsolver.
#if TENSORFLOW_USE_HIPSOLVER
#include "xla/stream_executor/rocm/hipsolver_wrapper.h"
using gpusolverHandle_t = hipsolverHandle_t;
#else  // TENSORFLOW_USE_ROCSOLVER
#include "xla/stream_executor/rocm/rocblas_wrapper.h"
#include "xla/stream_executor/rocm/rocsolver_wrapper.h"
using gpusolverHandle_t = rocblas_handle;
#endif  // TF_ROCM_VERSION >= 40500

#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu_solver_context.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {

class RocmSolverContext : public GpuSolverContext {
 public:
  ~RocmSolverContext() override;
  static absl::StatusOr<std::unique_ptr<GpuSolverContext>> Create();

  absl::Status SetStream(Stream* stream) override;
  absl::Status PotrfBatched(blas::UpperLower uplo, int n,
                            DeviceMemory<float*> as, int lda,
                            DeviceMemory<int> lapack_info,
                            int batch_size) override;
  absl::Status PotrfBatched(blas::UpperLower uplo, int n,
                            DeviceMemory<double*> as, int lda,
                            DeviceMemory<int> lapack_info,
                            int batch_size) override;
  absl::Status PotrfBatched(blas::UpperLower uplo, int n,
                            DeviceMemory<std::complex<float>*> as, int lda,
                            DeviceMemory<int> lapack_info,
                            int batch_size) override;
  absl::Status PotrfBatched(blas::UpperLower uplo, int n,
                            DeviceMemory<std::complex<double>*> as, int lda,
                            DeviceMemory<int> lapack_info,
                            int batch_size) override;
  absl::Status Potrf(blas::UpperLower uplo, int n, DeviceMemory<float> a,
                     int lda, DeviceMemory<int> lapack_info,
                     DeviceMemory<float> workspace) override;
  absl::Status Potrf(blas::UpperLower uplo, int n, DeviceMemory<double> a,
                     int lda, DeviceMemory<int> lapack_info,
                     DeviceMemory<double> workspace) override;
  absl::Status Potrf(blas::UpperLower uplo, int n,
                     DeviceMemory<std::complex<float>> a, int lda,
                     DeviceMemory<int> lapack_info,
                     DeviceMemory<std::complex<float>> workspace) override;
  absl::Status Potrf(blas::UpperLower uplo, int n,
                     DeviceMemory<std::complex<double>> a, int lda,
                     DeviceMemory<int> lapack_info,
                     DeviceMemory<std::complex<double>> workspace) override;
  absl::StatusOr<int64_t> PotrfBufferSize(xla::PrimitiveType type,
                                          blas::UpperLower uplo, int n, int lda,
                                          int batch_size) override;

 private:
  explicit RocmSolverContext(gpusolverHandle_t handle);

  gpusolverHandle_t handle_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_SOLVER_CONTEXT_H_

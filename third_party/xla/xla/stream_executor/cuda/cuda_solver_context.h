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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_SOLVER_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_SOLVER_CONTEXT_H_

#include <complex>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu_solver_context.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {

class CudaSolverContext : public GpuSolverContext {
 public:
  static absl::StatusOr<std::unique_ptr<GpuSolverContext>> Create();

  ~CudaSolverContext() override;

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
  explicit CudaSolverContext(cusolverDnHandle_t handle);

  cusolverDnHandle_t handle_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_SOLVER_CONTEXT_H_

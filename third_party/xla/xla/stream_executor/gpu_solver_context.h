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

#ifndef XLA_STREAM_EXECUTOR_GPU_SOLVER_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_GPU_SOLVER_CONTEXT_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {

// A virtual interface for interacting with a gpu solver context.
//
// This is used to abstract away the implementation of the solver context,
// allowing us to switch between different solver implementations (e.g.
// cuSolver, rocmsolver, etc.).
class GpuSolverContext {
 public:
  virtual absl::Status SetStream(Stream* stream) = 0;
  virtual ~GpuSolverContext() = default;

  // Computes the Cholesky factorization of multiple matrices.  See
  // https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-batchpotrf
  //
  // `as` is a list of pointers to the batch_size individual n x n matrices
  // that make up the input array.
  virtual absl::Status PotrfBatched(blas::UpperLower uplo, int n,
                                    DeviceMemory<float*> as, int lda,
                                    DeviceMemory<int> lapack_info,
                                    int batch_size) = 0;
  virtual absl::Status PotrfBatched(blas::UpperLower uplo, int n,
                                    DeviceMemory<double*> as, int lda,
                                    DeviceMemory<int> lapack_info,
                                    int batch_size) = 0;
  virtual absl::Status PotrfBatched(blas::UpperLower uplo, int n,
                                    DeviceMemory<std::complex<float>*> as,
                                    int lda, DeviceMemory<int> lapack_info,
                                    int batch_size) = 0;
  virtual absl::Status PotrfBatched(blas::UpperLower uplo, int n,
                                    DeviceMemory<std::complex<double>*> as,
                                    int lda, DeviceMemory<int> lapack_info,
                                    int batch_size) = 0;

  virtual absl::Status Potrf(blas::UpperLower uplo, int n,
                             DeviceMemory<float> a, int lda,
                             DeviceMemory<int> lapack_info,
                             DeviceMemory<float> workspace) = 0;
  virtual absl::Status Potrf(blas::UpperLower uplo, int n,
                             DeviceMemory<double> a, int lda,
                             DeviceMemory<int> lapack_info,
                             DeviceMemory<double> workspace) = 0;
  virtual absl::Status Potrf(blas::UpperLower uplo, int n,
                             DeviceMemory<std::complex<float>> a, int lda,
                             DeviceMemory<int> lapack_info,
                             DeviceMemory<std::complex<float>> workspace) = 0;
  virtual absl::Status Potrf(blas::UpperLower uplo, int n,
                             DeviceMemory<std::complex<double>> a, int lda,
                             DeviceMemory<int> lapack_info,
                             DeviceMemory<std::complex<double>> workspace) = 0;

  // Returns the max size of the `workspace` required by Potrf and PotrfBatched,
  // in number of elements of `type`.
  //
  // (PotrfBatched doesn't require a workspace per se -- it uses the
  // input array as scratch.  But we do need to materialize the `as` input, and
  // we do this in the workspace.)
  //
  // This is a bit of a hack; we could instead split it up into two functions.
  // But at the moment, it's an implementation detail of CholeskyThunk whether
  // it calls Potrf or PotrfBatched, so we need to allocate enough scratch space
  // for either.
  //
  // In practice, this does not result in a notable increase in scratch space
  // needed, because both cases require a relatively small amount of scratch.
  virtual absl::StatusOr<int64_t> PotrfBufferSize(xla::PrimitiveType type,
                                                  blas::UpperLower uplo, int n,
                                                  int lda, int batch_size) = 0;
};

struct GpuSolverContextFactory {
  using Type =
      std::function<absl::StatusOr<std::unique_ptr<GpuSolverContext>>()>;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_SOLVER_CONTEXT_H_

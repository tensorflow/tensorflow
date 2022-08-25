/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUSOLVER_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUSOLVER_CONTEXT_H_

#include <complex>

#define TENSORFLOW_USE_HIPSOLVER \
  (TENSORFLOW_USE_ROCM && (TF_ROCM_VERSION >= 40500))
#define TENSORFLOW_USE_ROCSOLVER \
  (TENSORFLOW_USE_ROCM && (TF_ROCM_VERSION < 40500))
#define TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER \
  (!TENSORFLOW_USE_ROCM || TENSORFLOW_USE_HIPSOLVER)

#if !TENSORFLOW_USE_ROCM
#include "third_party/gpus/cuda/include/cusolverDn.h"
using gpusolverHandle_t = cusolverDnHandle_t;
#else
#include "rocm/rocm_config.h"
// Macros to ease the transition from rocsolver to hipsolver.
#if TENSORFLOW_USE_HIPSOLVER
#include "tensorflow/compiler/xla/stream_executor/rocm/hipsolver_wrapper.h"
using gpusolverHandle_t = hipsolverHandle_t;
#else  // TENSORFLOW_USE_ROCSOLVER
#include "tensorflow/compiler/xla/stream_executor/rocm/rocblas_wrapper.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocsolver_wrapper.h"
using gpusolverHandle_t = rocblas_handle;
#endif  // TF_ROCM_VERSION >= 40500
#endif  // TENSORFLOW_USE_ROCM

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

class GpuSolverContext {
 public:
  // stream may be nullptr, in which case the context can only be used for
  // buffer size queries.
  static StatusOr<GpuSolverContext> Create(se::Stream* stream);
  GpuSolverContext() = default;
  ~GpuSolverContext();

  GpuSolverContext(const GpuSolverContext&) = delete;
  GpuSolverContext(GpuSolverContext&&);
  GpuSolverContext& operator=(const GpuSolverContext&) = delete;
  GpuSolverContext& operator=(GpuSolverContext&&);

  bool SupportsPotrfBatched() const {
    return true;
  }

  // Computes the Cholesky factorization A = L * L^T for a single matrix.
  // Returns Status::OK() if the kernel was launched successfully. See:
  // http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-potrf
  Status Potrf(se::blas::UpperLower uplo, int n, se::DeviceMemory<float> a,
               int lda, se::DeviceMemory<int> lapack_info,
               se::DeviceMemoryBase workspace);
  Status Potrf(se::blas::UpperLower uplo, int n, se::DeviceMemory<double> a,
               int lda, se::DeviceMemory<int> lapack_info,
               se::DeviceMemoryBase workspace);
  Status Potrf(se::blas::UpperLower uplo, int n,
               se::DeviceMemory<std::complex<float>> a, int lda,
               se::DeviceMemory<int> lapack_info,
               se::DeviceMemoryBase workspace);
  Status Potrf(se::blas::UpperLower uplo, int n,
               se::DeviceMemory<std::complex<double>> a, int lda,
               se::DeviceMemory<int> lapack_info,
               se::DeviceMemoryBase workspace);

  // Computes the Cholesky factorization of multiple matrices.  See
  // https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-batchpotrf
  //
  // `as` is a list of pointers to the batch_size individual n x n matricies
  // that make up the input array.
  Status PotrfBatched(se::blas::UpperLower uplo, int n,
                      se::DeviceMemory<float*> as, int lda,
                      se::DeviceMemory<int> lapack_info, int batch_size);
  Status PotrfBatched(se::blas::UpperLower uplo, int n,
                      se::DeviceMemory<double*> as, int lda,
                      se::DeviceMemory<int> lapack_info, int batch_size);
  Status PotrfBatched(se::blas::UpperLower uplo, int n,
                      se::DeviceMemory<std::complex<float>*> as, int lda,
                      se::DeviceMemory<int> lapack_info, int batch_size);
  Status PotrfBatched(se::blas::UpperLower uplo, int n,
                      se::DeviceMemory<std::complex<double>*> as, int lda,
                      se::DeviceMemory<int> lapack_info, int batch_size);

  // Returns the max size of the `workspace` required by Potrf and PotrfBatched,
  // in number of elements of `type`.
  //
  // (cusolver's PotrfBatched doesn't require a workspace per se -- it uses the
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
  StatusOr<int64_t> PotrfBufferSize(PrimitiveType type,
                                    se::blas::UpperLower uplo, int n, int lda,
                                    int batch_size);

 private:
  GpuSolverContext(se::Stream* stream, gpusolverHandle_t handle);

  gpusolverHandle_t handle() const { return handle_; }

  se::Stream* stream_ = nullptr;
  gpusolverHandle_t handle_ = nullptr;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUSOLVER_CONTEXT_H_

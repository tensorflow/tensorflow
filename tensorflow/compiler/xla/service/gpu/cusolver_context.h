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

#if !TENSORFLOW_USE_ROCM
#include "third_party/gpus/cuda/include/cusolverDn.h"
using gpusolverHandle_t = cusolverDnHandle_t;
#else
#include "tensorflow/stream_executor/rocm/rocsolver_wrapper.h"
using gpusolverHandle_t = rocblas_handle;
#endif

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"

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

  // Computes the Cholesky factorization A = L * L^T for a single matrix.
  // Returns Status::OK() if the kernel was launched successfully. See:
  // http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-potrf
  template <typename T, typename = std::enable_if_t<
                            std::is_same<T, float>::value ||
                            std::is_same<T, double>::value ||
                            std::is_same<T, std::complex<float>>::value ||
                            std::is_same<T, std::complex<double>>::value>>
  Status Potrf(se::blas::UpperLower uplo, int n, se::DeviceMemory<T> dev_A,
               int lda, se::DeviceMemory<int> dev_lapack_info,
               se::DeviceMemory<T> workspace) = delete;

  // Returns the size of the `workspace` required by Potrf, in number of
  // elements of `type`.
  StatusOr<int64> PotrfBufferSize(PrimitiveType type, se::blas::UpperLower uplo,
                                  int n, int lda);

 private:
  GpuSolverContext(se::Stream* stream, gpusolverHandle_t handle);

  gpusolverHandle_t handle() const { return handle_; }

  se::Stream* stream_ = nullptr;
  gpusolverHandle_t handle_ = nullptr;
};

#define CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)
#define POTRF_INSTANCE(T, type_prefix)                                  \
  template <>                                                           \
  Status GpuSolverContext::Potrf<T>(                                    \
      se::blas::UpperLower uplo, int n, se::DeviceMemory<T> A, int lda, \
      se::DeviceMemory<int> lapack_info, se::DeviceMemory<T> workspace);
CALL_LAPACK_TYPES(POTRF_INSTANCE);
#undef POTRF_INSTANCE
#undef CALL_LAPACK_TYPES

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUSOLVER_CONTEXT_H_

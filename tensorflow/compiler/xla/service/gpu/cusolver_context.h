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

#include "cuda/include/cublas_v2.h"
#include "cuda/include/cusolverDn.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"

namespace xla {
namespace gpu {

class CusolverContext {
 public:
  static StatusOr<CusolverContext> Create(se::Stream* stream);
  CusolverContext() = default;
  ~CusolverContext();

  CusolverContext(const CusolverContext&) = delete;
  CusolverContext(CusolverContext&&);
  CusolverContext& operator=(const CusolverContext&) = delete;
  CusolverContext& operator=(CusolverContext&&);

  se::Stream* stream() const { return stream_; }
  cusolverDnHandle_t handle() const { return handle_; }

  // Computes the Cholesky factorization A = L * L^T for a single matrix.
  // Returns Status::OK() if the kernel was launched successfully. See:
  // http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-potrf
  Status Potrf(se::blas::UpperLower uplo, int n, se::DeviceMemory<float> dev_A,
               int lda, se::DeviceMemory<int> dev_lapack_info,
               se::DeviceMemory<float> workspace);
  Status Potrf(se::blas::UpperLower uplo, int n, se::DeviceMemory<double> dev_A,
               int lda, se::DeviceMemory<int> dev_lapack_info,
               se::DeviceMemory<double> workspace);
  Status Potrf(se::blas::UpperLower uplo, int n,
               se::DeviceMemory<std::complex<float>> dev_A, int lda,
               se::DeviceMemory<int> dev_lapack_info,
               se::DeviceMemory<std::complex<float>> workspace);
  Status Potrf(se::blas::UpperLower uplo, int n,
               se::DeviceMemory<std::complex<double>> dev_A, int lda,
               se::DeviceMemory<int> dev_lapack_info,
               se::DeviceMemory<std::complex<double>> workspace);

  // Returns the size of the `workspace` required by Potrf, in number of
  // elements of size T.
  StatusOr<int64> PotrfBufferSize(se::blas::UpperLower uplo, int n,
                                  se::DeviceMemory<float> dev_A, int lda);
  StatusOr<int64> PotrfBufferSize(se::blas::UpperLower uplo, int n,
                                  se::DeviceMemory<double> dev_A, int lda);
  StatusOr<int64> PotrfBufferSize(se::blas::UpperLower uplo, int n,
                                  se::DeviceMemory<std::complex<float>> dev_A,
                                  int lda);
  StatusOr<int64> PotrfBufferSize(se::blas::UpperLower uplo, int n,
                                  se::DeviceMemory<std::complex<double>> dev_A,
                                  int lda);

 private:
  CusolverContext(se::Stream* stream, cusolverDnHandle_t handle);

  se::Stream* stream_ = nullptr;
  cusolverDnHandle_t handle_ = nullptr;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUSOLVER_CONTEXT_H_

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// CUDA-specific support for BLAS functionality -- this wraps the cuBLAS library
// capabilities, and is only included into CUDA implementation code -- it will
// not introduce cuda headers into other code.

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_H_

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_lt.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"

namespace stream_executor {

class Stream;

namespace gpu {
class GpuExecutor;
}  // namespace gpu

namespace cuda {

// Opaque and unique identifier for the cuBLAS plugin.
extern const PluginId kCuBlasPlugin;

// BLAS plugin for CUDA platform via cuBLAS library.
//
// This satisfies the platform-agnostic BlasSupport interface.
//
// Note that the cuBLAS handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent GpuExecutor is tied
// to. This simply happens as an artifact of creating the cuBLAS handle when a
// CUDA context is active.
//
// Thread-safe post-initialization.
class CUDABlas : public blas::BlasSupport {
 public:
  explicit CUDABlas(gpu::GpuExecutor *parent);

  // Allocates a cuBLAS handle.
  bool Init();

  // Releases the cuBLAS handle, if present.
  ~CUDABlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

  BlasLt &blas_lt() { return blas_lt_; }

 private:
  // Tells cuBLAS to enqueue the BLAS operation onto a particular Stream.
  //
  // cuBLAS is stateful, and only be associated with one stream (in order to
  // enqueue dispatch) at a given time. As a result, this generally must be
  // invoked before calling into cuBLAS.
  bool SetStream(Stream *stream) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns the underlying CUDA stream.
  cudaStream_t CUDAStream(Stream *stream);

  // A helper function that calls the real cuBLAS function together with error
  // handling.
  //
  // cublas_func:        cuBLAS function pointer.
  // cublas_name:        cuBLAS function name.
  // stream:             Stream to enqueue the BLAS operation onto.
  // pointer_mode_host:  Indicate if the pointer to a scalar value is from host
  //                     (true) or device (false).
  // args:               Arguments of cuBLAS function.
  template <typename FuncT, typename... Args>
  tsl::Status DoBlasInternalImpl(FuncT cublas_func, Stream *stream,
                                 bool pointer_mode_host, cublasMath_t math_type,
                                 Args... args);

  // Convenience functions that call DoBlasInternalImpl with err_on_failure=true
  // and math_type=CUBLAS_DEFAULT_MATH.
  template <typename FuncT, typename... Args>
  bool DoBlasInternal(FuncT cublas_func, Stream *stream, bool pointer_mode_host,
                      Args... args) {
    return DoBlasInternalImpl(cublas_func, stream, pointer_mode_host,
                              CUBLAS_DEFAULT_MATH, args...)
        .ok();
  }

  // A helper function to implement DoBlasGemmBatched interfaces for generic
  // types.
  template <typename T, typename Scalar, typename FuncT>
  tsl::Status DoBlasGemmBatchedInternal(
      FuncT cublas_func, Stream *stream, blas::Transpose transa,
      blas::Transpose transb, uint64_t m, uint64 n, uint64 k, Scalar alpha,
      const DeviceMemorySlice<T> &a_array, int lda,
      const DeviceMemorySlice<T> &b_array, int ldb, Scalar beta,
      const DeviceMemorySlice<T> &c_array, int ldc, int batch_count,
      ScratchAllocator *scratch_allocator);

  // Helper function for implementing DoBlasGemmWithProfiling.
  template <typename T, typename ParamType>
  bool DoBlasGemmWithProfilingImpl(Stream *stream, blas::Transpose transa,
                                   blas::Transpose transb, uint64_t m,
                                   uint64_t n, uint64 k, const ParamType &alpha,
                                   const DeviceMemory<T> &a, int lda,
                                   const DeviceMemory<T> &b, int ldb,
                                   const ParamType &beta, DeviceMemory<T> *c,
                                   int ldc,
                                   blas::ProfileResult *output_profile_result);

  // Helper function for implementing DoBlasGemvWithProfiling.
  template <typename T>
  bool DoBlasGemvWithProfilingImpl(Stream *stream, blas::Transpose trans,
                                   uint64_t m, uint64 n, const T &alpha,
                                   const DeviceMemory<T> &a, int lda,
                                   const DeviceMemory<T> &x, int incx,
                                   const T &beta, DeviceMemory<T> *y, int incy,
                                   blas::ProfileResult *output_profile_result);

  // Guards the cuBLAS handle for this device.
  absl::Mutex mu_;

  // GpuExecutor which instantiated this CUDABlas.
  // Immutable post-initialization.
  gpu::GpuExecutor *parent_;

  // cuBLAS library handle on the device.
  cublasHandle_t blas_ ABSL_GUARDED_BY(mu_);

  BlasLt blas_lt_;

  SE_DISALLOW_COPY_AND_ASSIGN(CUDABlas);
};

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_H_

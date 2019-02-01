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

// ROCM-specific support for BLAS functionality -- this wraps the rocBLAS library
// capabilities, and is only included into ROCM implementation code -- it will
// not introduce rocm headers into other code.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_

#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/plugin_registry.h"

namespace stream_executor {

class Stream;

namespace gpu {

// Opaque and unique identifier for the rocBLAS plugin.
extern const PluginId kRocBlasPlugin;

class GpuExecutor;

// BLAS plugin for ROCM platform via rocBLAS library.
//
// This satisfies the platform-agnostic BlasSupport interface.
//
// Note that the rocBLAS handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent GpuExecutor is tied
// to. This simply happens as an artifact of creating the rocBLAS handle when a
// ROCM context is active.
//
// Thread-safe post-initialization.
class ROCMBlas : public blas::BlasSupport {
 public:
  explicit ROCMBlas(GpuExecutor* parent);

  // Allocates a rocBLAS handle.
  bool Init();

  // Releases the rocBLAS handle, if present.
  ~ROCMBlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

 private:
  // Tells rocBLAS to enqueue the BLAS operation onto a particular Stream.
  //
  // rocBLAS is stateful, and only be associated with one stream (in order to
  // enqueue dispatch) at a given time. As a result, this generally must be
  // invoked before calling into rocBLAS.
  bool SetStream(Stream *stream) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // A helper function that calls the real rocBLAS function together with error
  // handling.
  //
  // rocblas_func:       rocBLAS function pointer.
  // rocblas_name:       rocBLAS function name.
  // stream:             Stream to enqueue the BLAS operation onto.
  // pointer_mode_host:  Indicate if the pointer to a scalar value is from host
  //                     (true) or device (false).
  // err_on_failure:     Whether to print an error if the rocBLAS function fails.
  // args:               Arguments of rocBLAS function.
  template <typename FuncT, typename... Args>
  bool DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                          bool pointer_mode_host, bool err_on_failure,
                          Args... args);

  // Convenience functions that call DoBlasInternalImpl with different values
  // for err_on_failure.
  template <typename FuncT, typename... Args>
  bool DoBlasInternal(FuncT rocblas_func, Stream *stream, bool pointer_mode_host,
                      Args... args) {
    return DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                              /*err_on_failure=*/true, args...);
  }
  template <typename FuncT, typename... Args>
  bool DoBlasInternalFailureOK(FuncT rocblas_func, Stream *stream,
                               bool pointer_mode_host, Args... args) {
    return DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                              /*err_on_failure=*/false, args...);
  }

  // A helper function to implement DoBlasGemmBatched interfaces for generic
  // types.
  template <typename T, typename FuncT>
  port::Status DoBlasGemmBatchedInternal(
      FuncT rocblas_func, Stream *stream, blas::Transpose transa,
      blas::Transpose transb, uint64 m, uint64 n, uint64 k, T alpha,
      const port::ArraySlice<DeviceMemory<T> *> &a_array, int lda,
      const port::ArraySlice<DeviceMemory<T> *> &b_array, int ldb, T beta,
      const port::ArraySlice<DeviceMemory<T> *> &c_array, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator);

  // Helper function for implementing DoBlasGemmWithAlgorithm.
  //
  // We take alpha and beta by const reference because T might be Eigen::half,
  // and we want to avoid pulling in a dependency on Eigen.  When we pass the
  // references to rocBLAS, we essentially reinterpret_cast to __half, which is
  // safe because Eigen::half inherits from __half.
  template <typename InT, typename OutT, typename CompT>
  bool DoBlasGemmWithAlgorithmImpl(
      Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
      uint64 n, uint64 k, const CompT &alpha, const DeviceMemory<InT> &a,
      int lda, const DeviceMemory<InT> &b, int ldb, const CompT &beta,
      DeviceMemory<OutT> *c, int ldc, blas::ComputationType computation_type,
      blas::AlgorithmType algorithm,
      blas::ProfileResult *output_profile_result);

  // Helper function for implementing DoBlasGemmWithProfiling.
  template <typename T, typename ParamType>
  bool DoBlasGemmWithProfilingImpl(
      Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
      uint64 n, uint64 k, const ParamType &alpha, const DeviceMemory<T> &a,
      int lda, const DeviceMemory<T> &b, int ldb, const ParamType &beta,
      DeviceMemory<T> *c, int ldc, blas::ProfileResult *output_profile_result);

  // Helper function for implementing DoBlasGemvWithProfiling.
  template <typename T>
  bool DoBlasGemvWithProfilingImpl(Stream *stream, blas::Transpose trans,
                                   uint64 m, uint64 n, const T &alpha,
                                   const DeviceMemory<T> &a, int lda,
                                   const DeviceMemory<T> &x, int incx,
                                   const T &beta, DeviceMemory<T> *y, int incy,
                                   blas::ProfileResult *output_profile_result);

  // mutex that guards the rocBLAS handle for this device.
  mutex mu_;

  // GpuExecutor which instantiated this ROCMBlas.
  // Immutable post-initialization.
  GpuExecutor* parent_;

  // rocBLAS library handle on the device.
  rocblas_handle blas_ GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(ROCMBlas);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_

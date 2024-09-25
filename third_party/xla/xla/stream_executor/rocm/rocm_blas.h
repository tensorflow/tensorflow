/* Copyright 2015 The OpenXLA Authors.

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

// ROCM-specific support for BLAS functionality -- this wraps the rocBLAS
// library capabilities, and is only included into ROCM implementation code --
// it will not introduce rocm headers into other code.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "rocm/rocm_config.h"

#define ROCBLAS_BETA_FEATURES_API
#if TF_ROCM_VERSION >= 50600
#include "rocm/include/rocblas/rocblas.h"
#else
#include "rocm/include/rocblas.h"
#endif
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/plugin_registry.h"
#if TF_HIPBLASLT
#include "xla/stream_executor/rocm/hip_blas_lt.h"
#endif
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

class Stream;

namespace gpu {

template <bool ErrorIfMissing, class Target, class A, class B, class... T>
struct ChooseType {
  using type = std::conditional_t<
      std::is_same_v<Target, A>, B,
      typename ChooseType<ErrorIfMissing, Target, T...>::type>;
};

template <class Target, class A, class B>
struct ChooseType<false, Target, A, B> {
  // default case: return the same type Target if there is no recursive match
  using type = std::conditional_t<std::is_same_v<Target, A>, B, Target>;
};

template <class Target, class A, class B>
struct ChooseType<true, Target, A, B> {
  // default case: return compile error if type is not found
  static_assert(std::is_same_v<Target, A>,
                "ChooseType: the target type is not found!");
  using type = B;
};

// Type conversion helper that helps to map non-rocblas types to rocblas types
template <typename T>
using RocBlasType_t =
    typename ChooseType<false, T, Eigen::half, rocblas_half, Eigen::bfloat16,
                        rocblas_bfloat16, std::complex<float>,
                        rocblas_float_complex, std::complex<double>,
                        rocblas_double_complex>::type;

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
  explicit ROCMBlas(GpuExecutor *parent);

  // Allocates a rocBLAS handle.
  bool Init();

  // Releases the rocBLAS handle, if present.
  ~ROCMBlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

  gpu::BlasLt *GetBlasLt() override {
#if TF_HIPBLASLT
    return &blas_lt_;
#else
    return nullptr;
#endif
  }

 private:
  // Tells rocBLAS to enqueue the BLAS operation onto a particular Stream.
  //
  // rocBLAS is stateful, and only be associated with one stream (in order to
  // enqueue dispatch) at a given time. As a result, this generally must be
  // invoked before calling into rocBLAS.
  bool SetStream(Stream *stream) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns the underlying ROCm stream
  hipStream_t ROCMStream(Stream *stream);

  // A helper function that calls the real rocBLAS function together with error
  // handling.
  //
  // rocblas_func:       rocBLAS function pointer.
  // rocblas_name:       rocBLAS function name.
  // stream:             Stream to enqueue the BLAS operation onto.
  // pointer_mode_host:  Indicate if the pointer to a scalar value is from host
  //                     (true) or device (false).
  // err_on_failure:     Whether to print an error if the rocBLAS function
  // fails. args:               Arguments of rocBLAS function.
  template <typename FuncT, typename... Args>
  absl::Status DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args &&...args);

  // Convenience functions that call DoBlasInternalImpl with different values
  // for err_on_failure.
  template <typename FuncT, typename... Args>
  bool DoBlasInternal(FuncT rocblas_func, Stream *stream,
                      bool pointer_mode_host, Args &&...args) {
    auto ret = DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                                  /*err_on_failure=*/true,
                                  std::forward<Args>(args)...);
    return ret.ok();
  }

  // Same as above, but returns absl::Status.
  template <typename FuncT, typename... Args>
  absl::Status DoBlasInternalStatus(FuncT rocblas_func, Stream *stream,
                                    bool pointer_mode_host, Args &&...args) {
    return DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                              /*err_on_failure=*/true,
                              std::forward<Args>(args)...);
  }

  template <typename FuncT, typename... Args>
  bool DoBlasInternalFailureOK(FuncT rocblas_func, Stream *stream,
                               bool pointer_mode_host, Args &&...args) {
    auto ret = DoBlasInternalImpl(rocblas_func, stream, pointer_mode_host,
                                  /*err_on_failure=*/false,
                                  std::forward<Args>(args)...);
    return ret.ok();
  }

  // A helper function to implement DoBlasGemmBatched interfaces for generic
  // types.
  //
  // Note: This function is implemented using gemm_strided_batched interface,
  // NOT gemm_batched interface, because rocblas do not support it. As a
  // result, if the passed in batch matrix are not allocated in strided batched
  // format, it might end up in non-trivial amount of memory allocation and
  // copy. To avoid this, always prioritize to use DoBlasGemmStridedBatched
  // interface.
  //
  // In most use cases, batch matrix do get allocated in strided manner, making
  // calling this interface equivalent with DoBlasGemmStridedBatched. The only
  // use case we see so far that violates this observation is when batch
  // matrix is created by broadcasting from a smaller matrix. When it happens,
  // It will take advantage of the AllocateStridedBuffer subroutine to
  // reallocate the memory layout to be strided batched.
  template <typename T, typename FuncT>
  absl::Status DoBlasGemmBatchedInternal(
      FuncT rocblas_func, Stream *stream, blas::Transpose transa,
      blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, T alpha,
      DeviceMemorySlice<T> a_ptrs_to_wrappers, int lda,
      DeviceMemorySlice<T> b_ptrs_to_wrappers, int ldb, T beta,
      DeviceMemorySlice<T> c_ptrs_to_wrappers, int ldc, int batch_count,
      ScratchAllocator *scratch_allocator);

  // mutex that guards the rocBLAS handle for this device.
  absl::Mutex mu_;

  // GpuExecutor which instantiated this ROCMBlas.
  // Immutable post-initialization.
  GpuExecutor *parent_;

  // rocBLAS library handle on the device.
  rocblas_handle blas_ ABSL_GUARDED_BY(mu_);

  // container holding solutions vector (to avoid reallocating it each time)
  std::vector<rocblas_int> solutions_;

  void MaybeLogGemmOp(StreamExecutor::GemmCallTrace::GemmType op,
                      blas::CallContext context, uint64_t size1,
                      uint64_t size2);

#if TF_HIPBLASLT
  rocm::BlasLt blas_lt_;
#endif

  ROCMBlas(const ROCMBlas &) = delete;
  void operator=(const ROCMBlas &) = delete;

  bool has_mfma_ = false;
  bool use_hgemm_alt_impl_ = false;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_BLAS_H_

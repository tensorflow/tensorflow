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

// The Stream is used in conjunction with the StreamExecutor "parent" to
// perform actions with a linear stream of dependencies. Dependencies can also
// be created between Streams to do task management (i.e. limit which tasks
// can be performed concurrently and specify what task dependencies exist).

#ifndef XLA_STREAM_EXECUTOR_STREAM_H_
#define XLA_STREAM_EXECUTOR_STREAM_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/thread_annotations.h"

namespace stream_executor {

namespace internal {
class StreamInterface;
}  // namespace internal

class DeviceMemoryBase;
template <typename ElemT>
class DeviceMemory;

namespace dnn {
class BatchDescriptor;
class FilterDescriptor;
class ConvolutionDescriptor;
class ProfileResult;
class AlgorithmDesc;
}  // namespace dnn

class StreamExecutor;
class ScratchAllocator;

namespace detail {

// Helper to return if `T` is the same type as `First` or any or `Rest`.
template <typename T>
constexpr bool is_any_of() {
  return false;
}

template <typename T, typename First, typename... Rest>
constexpr bool is_any_of() {
  return std::is_same_v<T, First> || is_any_of<T, Rest...>();
}

}  // namespace detail

// Convert a type to the corresponding QuantizedActivationMode.
template <typename ElementType>
struct Quantization;

// Represents a stream of dependent computations on a GPU device.
//
// The operations within a stream execute linearly and asynchronously until
// BlockHostUntilDone() is invoked, which synchronously joins host code with
// the execution of the stream.
//
// If any given operation fails when entraining work for the stream, ok() will
// indicate that an error has occurred. After initialization, once a stream is
// !ok(), it will never be ok().
//
// Thread-safe post-initialization.
class Stream {
 public:
  // Platform specific handle to the underlying resources behind a stream
  // implementation (e.g. it gives access to CUstream for CUDA platform).
  struct PlatformSpecificHandle {
    void *stream = nullptr;  // will be nullptr if not supported
  };

  // Instantiate a stream tied to parent as a platform executor. Work
  // entrained onto this stream will be launched/managed on that
  // StreamExecutor's platform.
  explicit Stream(StreamExecutor *parent);

  // Deallocates any stream resources that the parent StreamExecutor has
  // bestowed
  // upon this object.
  ~Stream();

  // TODO(ezhulenev): Consider removing this platform-specific accessor and
  // forward all users to platform-specific headers, however it requires careful
  // build rules set up to avoid leaking even more implementation details.
  PlatformSpecificHandle platform_specific_handle() const;

  // Returns whether any errors have occurred while entraining work for this
  // stream.
  bool ok() const { return !InErrorState(); }

  // Retrieves execution status back into the stream from the underlying
  // implementation without blocking the stream.
  //
  // Normally, Stream::BlockHostUntilDone is used to get execution status.
  // However, some devices use out-of-band mechnanisms to ensure their streams
  // have finished on-device work, without needing to block the streams. (These
  // devices should also override AllowsSyncOnCompletion to return false.) For
  // these devices, this method can be used after work is finished to retrieve
  // execution status.
  absl::Status RefreshStatus() TF_LOCKS_EXCLUDED(mu_);

  // Initialize the stream. This must be performed before entraining any other
  // operations.
  Stream &Init() TF_LOCKS_EXCLUDED(mu_);

  // Get or create a sub-stream from this stream. If there is any sub-stream in
  // the pool that can be reused then just return this sub-stream.  Otherwise
  // create a new sub-stream.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  Stream *GetOrCreateSubStream() TF_LOCKS_EXCLUDED(mu_);

  // Return the sub-stream back to the host stream so that it can be reused
  // later. Sub-streams that are !ok() will not be reused.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  void ReturnSubStream(Stream *sub_stream) TF_LOCKS_EXCLUDED(mu_);

  // Entrains onto the stream of operations: a kernel launch with the given
  // (variadic) parameters for the invocation. These arguments can be things
  // like DeviceMemory or primitive types such as int. What arguments you may
  // pass to a given kernel are noted as the template parameters to the
  // TypedKernel type that the machocc compiler generates.
  //
  // Template parameters:
  //  Params...   The type list of formal parameters that the typed kernel
  //              expects, which is matched against Args...
  //  Args...     The deduced type list for passed actual arguments
  //
  // Implementation: A compile-time compatibility check is performed that has
  // some leniency versus an exact parameter pack match -- for example,
  // `const DeviceMemory<T>` is considered "pack compatible" with a
  // `const DeviceMemory<T>&` formal parameter; in part, because we don't have
  // perfect forwarding support without rvalue references. It also attempts to
  // spit out helpful static_assert error traces with information as to the
  // argument number and types that were mismatched.
  template <typename... Params, typename... Args>
  absl::Status ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                          const TypedKernel<Params...> &kernel, Args... args);

  template <typename... Params, typename... Args>
  absl::Status ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                          ClusterDim cluster_dims,
                          const TypedKernel<Params...> &kernel, Args... args);

  // Same as above, with an explicit argument for shared memory size in bytes.
  template <typename... Params, typename... Args>
  absl::Status ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                          int32_t shmem_bytes,
                          const TypedKernel<Params...> &kernel, Args... args);

  template <typename... Params, typename... Args>
  absl::Status ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                          ClusterDim cluster_dims, int32_t shmem_bytes,
                          const TypedKernel<Params...> &kernel, Args... args);

  // Create a dependency for this stream's next work on the other stream
  // completing. Does not take ownership of other, and other must not be
  // null.
  //
  // Checks that a stream does not wait for itself, and it is up to the
  // user to guarantee that a stream does not come to wait on itself in a
  // cyclic manner; in that case, behavior is undefined.
  //
  // N.B. Base recursion case for the variadic ThenWaitFor.
  Stream &ThenWaitFor(Stream *other);

  // Waits for an event object to be set.
  // Note that ThenRecordEvent must have been called on the event before
  // you call this function; otherwise the event will be considered complete
  // and this wait will do nothing.
  Stream &ThenWaitFor(Event *event);

  // Inserts the specified event into the end of this stream. Once the stream
  // has processed all events prior to the insertion point, the event will be
  // marked as completed.
  // The stream does not take ownership of event - meaning that event's lifetime
  // must extend past the point at which it is marked complete!
  Stream &ThenRecordEvent(Event *event);

  /////////////////
  // BLAS support

  template <typename InputType, typename OutputType>
  absl::Status ThenBlasGemm(blas::Transpose transa, blas::Transpose transb,
                            uint64_t m, uint64 n, uint64 k,
                            const DeviceMemory<InputType> &a, int lda,
                            const DeviceMemory<InputType> &b, int ldb,
                            DeviceMemory<OutputType> *c, int ldc,
                            const NumericOptions &numeric_options,
                            blas::CallContext context) {
    InputType alpha{1.0};
    InputType beta{0.0};
    return ThenBlasGemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, numeric_options, context);
  }

  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status ThenBlasGemm(blas::Transpose transa, blas::Transpose transb,
                            uint64_t m, uint64 n, uint64 k, ConstantType alpha,
                            const DeviceMemory<InputType> &a, int lda,
                            const DeviceMemory<InputType> &b, int ldb,
                            ConstantType beta, DeviceMemory<OutputType> *c,
                            int ldc, const NumericOptions &numeric_options,
                            blas::CallContext context) {
    static_assert(
        detail::is_any_of<InputType, int8_t, Eigen::half, Eigen::bfloat16,
                          float, double, std::complex<float>,
                          std::complex<double>>(),
        "Input can be int8_t, half, bf16, float, double, std::complex<float> "
        "or "
        "std::complex<double>");
    static_assert(!std::is_same_v<InputType, Eigen::half> ||
                      detail::is_any_of<ConstantType, float, Eigen::half>(),
                  "If input is Eigen::half, constant has to be either "
                  "Eigen::half or float");
    static_assert(detail::is_any_of<InputType, int8_t, Eigen::half,
                                    Eigen::bfloat16, ConstantType>(),
                  "If input is not int8_t, Eigen::half, constant and input "
                  "types have to match");
    blas::BlasSupport *blas = parent()->AsBlas();
    if (!blas) {
      return absl::InternalError(
          "Attempting to perform BLAS operation using "
          "StreamExecutor without BLAS support");
    }

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    return blas->DoBlasGemm(
        this, transa, transb, m, n, k, blas::ToDataType<InputType>::value,
        alpha_ptr, a, lda, b, ldb, beta_ptr, c, ldc, numeric_options, context);
  }

  // TODO(reedwm): Update all callers to pass correct NumericOptions.
  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status ThenBlasGemm(blas::Transpose transa, blas::Transpose transb,
                            uint64_t m, uint64 n, uint64 k, ConstantType alpha,
                            const DeviceMemory<InputType> &a, int lda,
                            const DeviceMemory<InputType> &b, int ldb,
                            ConstantType beta, DeviceMemory<OutputType> *c,
                            int ldc, blas::CallContext context) {
    return ThenBlasGemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, NumericOptions{}, context);
  }

  template <typename InputType, typename OutputType>
  absl::Status ThenBlasGemmWithAlgorithm(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, const DeviceMemory<InputType> &a, int lda,
      const DeviceMemory<InputType> &b, int ldb, DeviceMemory<OutputType> *c,
      int ldc, blas::ComputationType computation_type,
      blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result,
      blas::CallContext context) {
    OutputType alpha{1};
    OutputType beta{0};
    return ThenBlasGemmWithAlgorithm(transa, transb, m, n, k, alpha, a, lda, b,
                                     ldb, beta, c, ldc, computation_type,
                                     algorithm, NumericOptions{},
                                     output_profile_result, context);
  }

  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status ThenBlasGemmWithAlgorithm(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, ConstantType alpha, const DeviceMemory<InputType> &a, int lda,
      const DeviceMemory<InputType> &b, int ldb, ConstantType beta,
      DeviceMemory<OutputType> *c, int ldc,
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,
      const NumericOptions &numeric_options,
      blas::ProfileResult *output_profile_result, blas::CallContext context) {
    TF_RETURN_IF_ERROR(
        CheckTypesForExtendedBlas<InputType, OutputType, ConstantType>(
            computation_type));

    blas::BlasSupport *blas = parent()->AsBlas();
    if (!blas) {
      return absl::InternalError(
          "Attempting to perform BLAS operation using "
          "StreamExecutor without BLAS support");
    }

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    absl::Status st = blas->DoBlasGemmWithAlgorithm(
        this, transa, transb, m, n, k, alpha_ptr, a,
        blas::ToDataType<InputType>::value, lda, b,
        blas::ToDataType<InputType>::value, ldb, beta_ptr, c,
        blas::ToDataType<OutputType>::value, ldc, computation_type, algorithm,
        numeric_options, output_profile_result, context);

    if (output_profile_result) {
      // The error is recorded in the profile.
      return absl::OkStatus();
    }
    return st;
  }

  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status ThenBlasGemmStridedBatchedWithAlgorithm(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, ConstantType alpha, const DeviceMemory<InputType> &a, int lda,
      int64_t stride_a, const DeviceMemory<InputType> &b, int ldb,
      int64_t stride_b, ConstantType beta, DeviceMemory<OutputType> *c, int ldc,
      int64_t stride_c, int batch_count, blas::ComputationType computation_type,
      blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
      blas::ProfileResult *output_profile_result, blas::CallContext context) {
    TF_RETURN_IF_ERROR(
        CheckTypesForExtendedBlas<InputType, OutputType, ConstantType>(
            computation_type));

    blas::BlasSupport *blas = parent()->AsBlas();
    if (!blas) {
      return absl::InternalError(
          "Attempting to perform BLAS operation using "
          "StreamExecutor without BLAS support");
    }
    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);
    absl::Status st = blas->DoBlasGemmStridedBatchedWithAlgorithm(
        this, transa, transb, m, n, k, alpha_ptr, a,
        blas::ToDataType<InputType>::value, lda, stride_a, b,
        blas::ToDataType<InputType>::value, ldb, stride_b, beta_ptr, c,
        blas::ToDataType<OutputType>::value, ldc, stride_c, batch_count,
        computation_type, algorithm, numeric_options, output_profile_result,
        context);
    if (output_profile_result) {
      // The error is recorded in the profile.
      return absl::OkStatus();
    }
    return st;
  }

  template <typename T>
  using DeviceMemorySlice = absl::Span<DeviceMemory<T> *const>;

  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status ThenBlasGemmStridedBatched(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, ConstantType alpha, const DeviceMemory<InputType> &a, int lda,
      int64_t stride_a, const DeviceMemory<InputType> &b, int ldb,
      int64_t stride_b, ConstantType beta, DeviceMemory<OutputType> *c, int ldc,
      int64_t stride_c, int batch_count, const NumericOptions &numeric_options,
      blas::CallContext context) {
    static_assert(
        detail::is_any_of<InputType, int8_t, float, Eigen::half,
                          Eigen::bfloat16, double, std::complex<float>,
                          std::complex<double>>(),
        "Unsupported input type");
    static_assert(std::is_same_v<ConstantType, InputType> ||
                      (detail::is_any_of<InputType, int8_t, Eigen::half,
                                         Eigen::bfloat16>() &&
                       std::is_same_v<ConstantType, float>),
                  "Mismatched input and alpha/beta types");
    blas::BlasSupport *blas = parent()->AsBlas();
    if (!blas) {
      return absl::InternalError(
          "Attempting to perform BLAS operation using "
          "StreamExecutor without BLAS support");
    }

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    return blas->DoBlasGemmStridedBatched(
        this, transa, transb, m, n, k, blas::ToDataType<InputType>::value,
        alpha_ptr, a, lda, stride_a, b, ldb, stride_b, beta_ptr, c, ldc,
        stride_c, batch_count, numeric_options, context);
  }

  // Entrain onto the stream: a memcpy to a host destination from a GPU source
  // of the given target size. host_dst must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate or otherwise allocated and
  // then registered with StreamExecutor::HostMemoryRegister.
  Stream &ThenMemcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                     uint64_t size);

  // Entrain onto the stream: a memcpy to a GPU destination from a host source
  // of the given target size. host_src must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate or otherwise allocated and
  // then registered with StreamExecutor::HostMemoryRegister.
  Stream &ThenMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                     uint64_t size);

  // Alternative interface for memcpying from device to host that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  Stream &ThenMemcpyD2H(const DeviceMemory<T> &gpu_src,
                        absl::Span<T> host_dst) {
    auto host_size = host_dst.size() * sizeof(T);
    CHECK(gpu_src.size() == 0 || host_size >= gpu_src.size());
    return ThenMemcpy(host_dst.begin(), gpu_src, host_size);
  }

  // Alternative interface for memcpying from host to device that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  Stream &ThenMemcpyH2D(absl::Span<const T> host_src,
                        DeviceMemory<T> *gpu_dst) {
    auto host_size = host_src.size() * sizeof(T);
    CHECK(gpu_dst->size() == 0 || gpu_dst->size() >= host_size);
    return ThenMemcpy(gpu_dst, host_src.begin(), host_size);
  }

  // Entrain onto the stream: a memcpy to a GPU destination from a GPU source
  // of the given target size. gpu_src/dst must be pointers to GPU memory and
  // peer access must be enabled between their owning StreamExecutors.
  Stream &ThenMemcpy(DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src,
                     uint64_t size);

  // Calls to the device-to-device copy overload of ThenMemcpy -- useful for
  // ensuring that the host pointer isn't getting confused accidentally with a
  // device pointer if you're not doing metaprogramming against the API.
  Stream &ThenMemcpyD2D(DeviceMemoryBase *gpu_dst,
                        const DeviceMemoryBase &gpu_src, uint64_t size) {
    return ThenMemcpy(gpu_dst, gpu_src, size);
  }

  // Entrain onto the stream: a memset of zero at a GPU location of size bytes.
  // The location must not be null.
  Stream &ThenMemZero(DeviceMemoryBase *location, uint64_t size);

  // Entrain onto the stream: a memset of a 32-bit pattern at a GPU location of
  // size bytes, where bytes must be evenly 32-bit sized (i.e. evenly divisible
  // by 4). The location must not be null.
  Stream &ThenMemset32(DeviceMemoryBase *location, uint32_t pattern,
                       uint64_t size);

  // (Synchronously) block the host code waiting for the operations
  // entrained on the stream (enqueued to this point in program
  // execution) to complete.
  //
  // Returns an OK status if the blocking was successful and the stream is ok().
  // Otherwise returns an error describing why the blocking failed.
  absl::Status BlockHostUntilDone() TF_LOCKS_EXCLUDED(mu_);

  // Returns the (opaque) platform-specific backing object. Ownership is not
  // transferred to the caller.
  internal::StreamInterface *implementation() { return implementation_.get(); }

  // Entrains onto the stream a callback to the host (from the device).
  // Behaves as ThenDoHostCallbackWithStatus below, but the callback should
  // never fail or its failure is inconsequential.
  //
  // This is kept for backward compatibility. Future code should use
  // ThenDoHostCallbackWithStatus and explicitly return a success status.
  // TODO(b/112125301): Eventually remove this method.
  Stream &ThenDoHostCallback(absl::AnyInvocable<void() &&> callback);

  // Entrains onto the stream a callback to the host (from the device).
  // Host callbacks block/occupy the stream just as device functions
  // (execute one at a time, block later stream operations).
  // Whether the callback return status affects the result of BlockHostUntilDone
  // is platform-dependent.
  //
  // Behavior is undefined when synchronizing using OpenCL user events.
  // Behavior is undefined if host callbacks call device routines or insert
  // them into any stream.
  //
  // On certain platforms, ThenDoHostCallback is expected to have significant
  // negative effects on performance.
  Stream &ThenDoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback);

  // Returns the StreamExecutor (parent object) associated with this stream.
  StreamExecutor *parent() const {
    CHECK(parent_ != nullptr);
    return parent_;
  }

  //
  CudaComputeCapability GetCudaComputeCapability() const {
    return parent()->GetDeviceDescription().cuda_compute_capability();
  }

  RocmComputeCapability GetRocmComputeCapability() const {
    return parent()->GetDeviceDescription().rocm_compute_capability();
  }

  // Returns a debugging string "[stream=0x...,impl=0x...]".
  std::string DebugStreamPointers() const;

  void SetPriority(StreamPriority priority);
  void SetPriority(int priority);

  std::variant<StreamPriority, int> priority() const;

 private:
  // Checks whether types match before a call to extended BLAS version.
  template <typename ABType, typename CType, typename ScaleType>
  absl::Status CheckTypesForExtendedBlas(
      blas::ComputationType computation_type) {
    static_assert(
        detail::is_any_of<ABType, Eigen::half, Eigen::bfloat16, float, double,
                          int8_t, std::complex<float>, std::complex<double>>(),
        "The only buffer types supported are: Eigen::half, float, "
        "double, int8, std::complex<float> and std::complex<double>");
    static_assert(
        std::is_same_v<ScaleType, CType> ||
            (std::is_same_v<ScaleType, float> &&
             detail::is_any_of<CType, Eigen::half, Eigen::bfloat16>()),
        "Mismatched alpha/beta and output types");

    bool valid_computation_type = [computation_type] {
      switch (computation_type) {
        case blas::ComputationType::kF16:
          return std::is_same_v<CType, Eigen::half>;
        case blas::ComputationType::kF32:
          return detail::is_any_of<CType, Eigen::half, Eigen::bfloat16, float,
                                   std::complex<float>>();
        case blas::ComputationType::kF64:
          return detail::is_any_of<CType, double, std::complex<double>>();
        case blas::ComputationType::kI32:
          return std::is_same_v<CType, int32_t>;
        case blas::ComputationType::kF16AsF32:   // fall-through
        case blas::ComputationType::kBF16AsF32:  // fall-through
        case blas::ComputationType::kTF32AsF32:
          return detail::is_any_of<CType, float, std::complex<float>>();
      }
    }();

    if (!valid_computation_type) {
      return absl::InternalError(absl::StrCat(
          "Invalid computation type ",
          blas::ComputationTypeString(computation_type), " for output type: ",
          blas::DataTypeString(blas::ToDataType<CType>::value)));
    }
    return absl::OkStatus();
  }

  bool InErrorState() const TF_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    return !status_.ok();
  }

  // Sets the error state if operation_retcode is false.
  // This is a useful shorthand for many stream routines.
  void CheckError(bool operation_retcode) TF_LOCKS_EXCLUDED(mu_);

  // Checks the status and logs the error message, if any.
  void CheckStatus(absl::Status status) TF_LOCKS_EXCLUDED(mu_);

  void SetError() { CheckError(false /* = operation_retcode */); }

  void SetErrorAndLogNoDnnSupport() {
    SetError();
    LOG(WARNING) << "attempting to perform DNN operation using StreamExecutor "
                    "without DNN support";
  }

  // The StreamExecutor that supports the operation of this stream.
  StreamExecutor *parent_;

  // The platform-dependent implementation that the StreamExecutor interface
  // delegates to.
  std::unique_ptr<internal::StreamInterface> implementation_;

  // mutex that guards the allocation / error state flags.
  // Mutable so that it can be obtained via const reader lock.
  mutable absl::Mutex mu_;

  // Whether Init() was successfully called to allocate this stream on the
  // underlying platform. It simply flips from 0 to 1 with a sanity check.
  // See StreamExecutor::AllocateStream.
  bool allocated_ ABSL_GUARDED_BY(mu_);

  // The last error (if any) of all method calls.
  absl::Status status_ ABSL_GUARDED_BY(mu_);

  // Sub-streams that are generated from this stream. Each element has a pointer
  // to sub-stream and a boolean value indicating if this substream is ready to
  // be reused.
  std::vector<std::pair<std::unique_ptr<Stream>, bool>> sub_streams_
      ABSL_GUARDED_BY(mu_);

  // Non-extended BLAS interface requires alpha/beta to be floats when input
  // type is Eigen::half. However, for consistency purposes it is convenient
  // for the interface to accept Eigen::half.
  template <typename T>
  void UpcastHalfToFloat(void **alpha_ptr, void **beta_ptr,
                         float *alpha_storage, float *beta_storage) {
    if (std::is_same<T, Eigen::half>::value) {
      *alpha_storage =
          static_cast<float>(*reinterpret_cast<Eigen::half *>(*alpha_ptr));
      *beta_storage =
          static_cast<float>(*reinterpret_cast<Eigen::half *>(*beta_ptr));
      *alpha_ptr = alpha_storage;
      *beta_ptr = beta_storage;
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      *alpha_storage =
          static_cast<float>(*reinterpret_cast<Eigen::bfloat16 *>(*alpha_ptr));
      *beta_storage =
          static_cast<float>(*reinterpret_cast<Eigen::bfloat16 *>(*beta_ptr));
      *alpha_ptr = alpha_storage;
      *beta_ptr = beta_storage;
    }
  }

  Stream(const Stream &) = delete;
  void operator=(const Stream &) = delete;
};

////////////
// Inlines

template <typename... Params, typename... Args>
inline absl::Status Stream::ThenLaunch(ThreadDim thread_dims,
                                       BlockDim block_dims,
                                       const TypedKernel<Params...> &kernel,
                                       Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(
      parent_->Launch(this, thread_dims, block_dims, *kernel, *kernel_args));
  return absl::OkStatus();
}

template <typename... Params, typename... Args>
inline absl::Status Stream::ThenLaunch(ThreadDim thread_dims,
                                       BlockDim block_dims, int32_t shmem_bytes,
                                       const TypedKernel<Params...> &kernel,
                                       Args... args) {
  auto kernel_args = PackKernelArgs(shmem_bytes, args...);
  TF_RETURN_IF_ERROR(
      parent_->Launch(this, thread_dims, block_dims, *kernel, *kernel_args));
  return absl::OkStatus();
}

template <typename... Params, typename... Args>
inline absl::Status Stream::ThenLaunch(ThreadDim thread_dims,
                                       BlockDim block_dims,
                                       ClusterDim cluster_dims,
                                       const TypedKernel<Params...> &kernel,
                                       Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(parent_->Launch(this, thread_dims, block_dims,
                                     cluster_dims, *kernel, *kernel_args));
  return absl::OkStatus();
}

template <typename... Params, typename... Args>
inline absl::Status Stream::ThenLaunch(
    ThreadDim thread_dims, BlockDim block_dims, ClusterDim cluster_dims,
    int32_t shmem_bytes, const TypedKernel<Params...> &kernel, Args... args) {
  auto kernel_args = PackKernelArgs(shmem_bytes, args...);
  TF_RETURN_IF_ERROR(parent_->Launch(this, thread_dims, block_dims,
                                     cluster_dims, *kernel, *kernel_args));
  return absl::OkStatus();
}

template <>
struct Quantization<uint8_t> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k8Bit;
};

template <>
struct Quantization<uint16_t> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k16Bit;
};

template <>
struct Quantization<int32_t> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k32Bit;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_H_

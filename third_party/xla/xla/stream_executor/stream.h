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

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/thread_annotations.h"

namespace stream_executor {

namespace internal {
class StreamInterface;
}  // namespace internal

class DeviceMemoryBase;
template <typename ElemT>
class DeviceMemory;

class StreamExecutor;

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
  explicit Stream(StreamExecutor *parent,
                  std::unique_ptr<StreamInterface> implementation);

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

  // Get or create a sub-stream from this stream. If there is any sub-stream in
  // the pool that can be reused then just return this sub-stream.  Otherwise
  // create a new sub-stream.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  absl::StatusOr<Stream *> GetOrCreateSubStream() TF_LOCKS_EXCLUDED(mu_);

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

  // Same as above, with an explicit argument for shared memory size in bytes.
  template <typename... Params, typename... Args>
  absl::Status ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                          int32_t shmem_bytes,
                          const TypedKernel<Params...> &kernel, Args... args);

  // Create a dependency for this stream's next work on the other stream
  // completing. Does not take ownership of other, and other must not be
  // null.
  //
  // Checks that a stream does not wait for itself, and it is up to the
  // user to guarantee that a stream does not come to wait on itself in a
  // cyclic manner; in that case, behavior is undefined.
  absl::Status WaitFor(Stream *other);

  // Waits for an event object to be set.
  // Note that RecordEvent must have been called on the event before
  // you call this function; otherwise the event will be considered complete
  // and this wait will do nothing.
  absl::Status WaitFor(Event *event);

  // Inserts the specified event into the end of this stream. Once the stream
  // has processed all events prior to the insertion point, the event will be
  // marked as completed.
  // The stream does not take ownership of event - meaning that event's lifetime
  // must extend past the point at which it is marked complete!
  absl::Status RecordEvent(Event *event);

  // Entrain onto the stream: a memcpy to a host destination from a GPU source
  // of the given target size. host_dst must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate.
  absl::Status Memcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                      uint64_t size);

  // Entrain onto the stream: a memcpy to a GPU destination from a host source
  // of the given target size. host_src must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate.
  absl::Status Memcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                      uint64_t size);

  // Alternative interface for memcpying from device to host that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  absl::Status MemcpyD2H(const DeviceMemory<T> &gpu_src,
                         absl::Span<T> host_dst) {
    auto host_size = host_dst.size() * sizeof(T);
    if (gpu_src.size() == 0 || host_size >= gpu_src.size()) {
      return Memcpy(host_dst.begin(), gpu_src, host_size);
    }
    return absl::InternalError("Bad source size.");
  }

  // Alternative interface for memcpying from host to device that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  absl::Status MemcpyH2D(absl::Span<const T> host_src,
                         DeviceMemory<T> *gpu_dst) {
    auto host_size = host_src.size() * sizeof(T);
    if (gpu_dst->size() == 0 || gpu_dst->size() >= host_size) {
      return Memcpy(gpu_dst, host_src.begin(), host_size);
    }
    return absl::InternalError("Bad destination size.");
  }

  // Entrain onto the stream: a memcpy to a GPU destination from a GPU source
  // of the given target size. gpu_src/dst must be pointers to GPU memory and
  // peer access must be enabled between their owning StreamExecutors.
  absl::Status Memcpy(DeviceMemoryBase *gpu_dst,
                      const DeviceMemoryBase &gpu_src, uint64_t size);
  absl::Status MemcpyD2D(DeviceMemoryBase *gpu_dst,
                         const DeviceMemoryBase &gpu_src, uint64_t size) {
    return Memcpy(gpu_dst, gpu_src, size);
  }

  // Entrain onto the stream: a memset of zero at a GPU location of size bytes.
  // The location must not be null.
  absl::Status MemZero(DeviceMemoryBase *location, uint64_t size);

  // Entrain onto the stream: a memset of a 32-bit pattern at a GPU location of
  // size bytes, where bytes must be evenly 32-bit sized (i.e. evenly divisible
  // by 4). The location must not be null.
  absl::Status Memset32(DeviceMemoryBase *location, uint32_t pattern,
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
  StreamInterface *implementation() { return implementation_.get(); }

  // Entrains onto the stream a callback to the host (from the device).
  // Behaves as DoHostCallbackWithStatus below, but the callback should
  // never fail or its failure is inconsequential.
  //
  // This is kept for backward compatibility. Future code should use
  // DoHostCallbackWithStatus and explicitly return a success status.
  // TODO(b/112125301): Eventually remove this method.
  absl::Status DoHostCallback(absl::AnyInvocable<void() &&> callback);

  // Entrains onto the stream a callback to the host (from the device).
  // Host callbacks block/occupy the stream just as device functions
  // (execute one at a time, block later stream operations).
  // Whether the callback return status affects the result of BlockHostUntilDone
  // is platform-dependent.
  //
  // On certain platforms, DoHostCallback is expected to have significant
  // negative effects on performance.
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback);

  // Returns the StreamExecutor (parent object) associated with this stream.
  StreamExecutor *parent() const {
    CHECK(parent_ != nullptr);
    return parent_;
  }

  CudaComputeCapability GetCudaComputeCapability() const {
    return parent()->GetDeviceDescription().cuda_compute_capability();
  }

  RocmComputeCapability GetRocmComputeCapability() const {
    return parent()->GetDeviceDescription().rocm_compute_capability();
  }

  std::variant<StreamPriority, int> priority() const;

 private:
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

  // The StreamExecutor that supports the operation of this stream.
  StreamExecutor *parent_;

  // The platform-dependent implementation that the StreamExecutor interface
  // delegates to.
  std::unique_ptr<StreamInterface> implementation_;

  // mutex that guards the allocation / error state flags.
  // Mutable so that it can be obtained via const reader lock.
  mutable absl::Mutex mu_;

  // The last error (if any) of all method calls.
  absl::Status status_ ABSL_GUARDED_BY(mu_);

  // Sub-streams that are generated from this stream. Each element has a pointer
  // to sub-stream and a boolean value indicating if this substream is ready to
  // be reused.
  std::vector<std::pair<std::unique_ptr<Stream>, bool>> sub_streams_
      ABSL_GUARDED_BY(mu_);

  Stream(const Stream &) = delete;
  void operator=(const Stream &) = delete;
};

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

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_H_

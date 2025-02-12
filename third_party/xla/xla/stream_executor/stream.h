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
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor {

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

  // Deallocates any stream resources that the parent StreamExecutor has
  // bestowed
  // upon this object.
  virtual ~Stream() = default;

  // TODO(ezhulenev): Consider removing this platform-specific accessor and
  // forward all users to platform-specific headers, however it requires careful
  // build rules set up to avoid leaking even more implementation details.
  virtual PlatformSpecificHandle platform_specific_handle() const = 0;

  // Returns whether any errors have occurred while entraining work for this
  // stream.
  virtual bool ok() const = 0;

  // Retrieves execution status back into the stream from the underlying
  // implementation without blocking the stream.
  //
  // Normally, Stream::BlockHostUntilDone is used to get execution status.
  // However, some devices use out-of-band mechanisms to ensure their streams
  // have finished on-device work, without needing to block the streams. (These
  // devices should also override AllowsSyncOnCompletion to return false.) For
  // these devices, this method can be used after work is finished to retrieve
  // execution status.
  virtual absl::Status RefreshStatus() {
    return absl::UnimplementedError(
        "RefreshStatus is not supported on this stream.");
  }

  // Get or create a sub-stream from this stream. If there is any sub-stream in
  // the pool that can be reused then just return this sub-stream.  Otherwise
  // create a new sub-stream.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  virtual absl::StatusOr<Stream *> GetOrCreateSubStream() = 0;

  // Return the sub-stream back to the host stream so that it can be reused
  // later. Sub-streams that are !ok() will not be reused.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  virtual void ReturnSubStream(Stream *sub_stream) = 0;

  // Create a dependency for this stream's next work on the other stream
  // completing. Does not take ownership of other, and other must not be
  // null.
  //
  // Checks that a stream does not wait for itself, and it is up to the
  // user to guarantee that a stream does not come to wait on itself in a
  // cyclic manner; in that case, behavior is undefined.
  virtual absl::Status WaitFor(Stream *other) = 0;

  // Waits for an event object to be set.
  // Note that RecordEvent must have been called on the event before
  // you call this function; otherwise the event will be considered complete
  // and this wait will do nothing.
  virtual absl::Status WaitFor(Event *event) = 0;

  // Inserts the specified event into the end of this stream. Once the stream
  // has processed all events prior to the insertion point, the event will be
  // marked as completed.
  // The stream does not take ownership of event - meaning that event's lifetime
  // must extend past the point at which it is marked complete!
  virtual absl::Status RecordEvent(Event *event) = 0;

  // Entrain onto the stream: a memcpy to a host destination from a GPU source
  // of the given target size. host_dst must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate.
  virtual absl::Status Memcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                              uint64_t size) = 0;

  // Entrain onto the stream: a memcpy to a GPU destination from a host source
  // of the given target size. host_src must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate.
  virtual absl::Status Memcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                              uint64_t size) = 0;

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
  virtual absl::Status Memcpy(DeviceMemoryBase *gpu_dst,
                              const DeviceMemoryBase &gpu_src, uint64_t size) {
    return absl::UnimplementedError(
        "Memcpy from device to device is not implemented for this "
        "stream.");
  }

  absl::Status MemcpyD2D(DeviceMemoryBase *gpu_dst,
                         const DeviceMemoryBase &gpu_src, uint64_t size) {
    return Memcpy(gpu_dst, gpu_src, size);
  }

  // Entrain onto the stream: a memset of zero at a device location of size
  // bytes. The location must not be null.
  virtual absl::Status MemZero(DeviceMemoryBase *location, uint64_t size) {
    return absl::UnimplementedError("MemZero is not supported on this stream.");
  }

  // Entrain onto the stream: a memset of a 32-bit pattern at device location of
  // size bytes, where bytes must be evenly 32-bit sized (i.e. evenly divisible
  // by 4). The location must not be null.
  virtual absl::Status Memset32(DeviceMemoryBase *location, uint32_t pattern,
                                uint64_t size) {
    return absl::UnimplementedError(
        "Memset32 is not supported on this stream.");
  }

  // (Synchronously) block the host code waiting for the operations
  // entrained on the stream (enqueued to this point in program
  // execution) to complete.
  //
  // Returns an OK status if the blocking was successful and the stream is ok().
  // Otherwise returns an error describing why the blocking failed.
  virtual absl::Status BlockHostUntilDone() = 0;

  // Entrains onto the stream a callback to the host (from the device).
  // Behaves as DoHostCallbackWithStatus below, but the callback should
  // never fail or its failure is inconsequential.
  //
  // This is kept for backward compatibility. Future code should use
  // DoHostCallbackWithStatus and explicitly return a success status.
  // TODO(b/112125301): Eventually remove this method.
  absl::Status DoHostCallback(absl::AnyInvocable<void() &&> callback) {
    return DoHostCallbackWithStatus([cb = std::move(callback)]() mutable {
      std::move(cb)();
      return absl::OkStatus();
    });
  }

  // Entrains onto the stream a callback to the host (from the device).
  // Host callbacks block/occupy the stream just as device functions
  // (execute one at a time, block later stream operations).
  // Whether the callback return status affects the result of BlockHostUntilDone
  // is platform-dependent.
  //
  // On certain platforms, DoHostCallback is expected to have significant
  // negative effects on performance.
  virtual absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) = 0;

  // Returns the StreamExecutor (parent object) associated with this stream.
  virtual StreamExecutor *parent() const = 0;

  // Returns the CudaComputeCapability for this stream.
  virtual CudaComputeCapability GetCudaComputeCapability() const = 0;

  // Returns the RocmComputeCapability for this stream.
  virtual RocmComputeCapability GetRocmComputeCapability() const = 0;

  // Gets priority for a stream.
  virtual std::variant<StreamPriority, int> priority() const = 0;

  // Get/set a name for a stream, which can be shown in profiling tools
  virtual const std::string &GetName() const = 0;
  virtual void SetName(std::string name) = 0;

  // Create an EventBasedTimer that can be used to time operations on this
  // stream using Events.
  //
  // If use_delay_kernel is true, the timer will launch a delay kernel into the
  // stream and queue a start event immediately afterwards. This delay kernel
  // blocks execution on the stream until EventBasedTimer::GetElapsedDuration()
  // is called, at which point an end event is queued and the delay kernel
  // exits. This allows the device execution time of the tasks queued to the
  // stream while the timer is active to be measured more accurately.
  virtual absl::StatusOr<std::unique_ptr<EventBasedTimer>>
  CreateEventBasedTimer(bool use_delay_kernel) {
    return absl::UnimplementedError(
        "This stream does not support EventBasedTimers.");
  }

  // Helper method to launch a kernel with optional cluster dimensions.
  virtual absl::Status LaunchKernel(
      const ThreadDim &thread_dims, const BlockDim &block_dims,
      const std::optional<ClusterDim> &cluster_dims, void *function,
      absl::string_view name, void **args, int64_t shmem_bytes) {
    return absl::UnimplementedError("Not implemented");
  }
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_H_

/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_STREAM_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_STREAM_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/sycl/sycl_context.h"
#include "xla/stream_executor/sycl/sycl_event.h"

namespace stream_executor::sycl {

class SyclStream : public StreamCommon {
 public:
  // Makes the current stream wait until all operations enqueued in the other
  // stream up to its most recently recorded event have completed.
  absl::Status WaitFor(Stream* other) override;

  // Records the most recent event on the current stream into the given Event
  // object. This typically marks the point up to which work has been enqueued
  // on the stream. This allows other streams or operations to synchronize with
  // this point in the stream's execution.
  absl::Status RecordEvent(Event* event) override;

  // Blocks execution on the current stream until the given event is completed.
  absl::Status WaitFor(Event* event) override;

  // Enqueues an asynchronous operation to set the specified device memory
  // region to the given value.
  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern,
                        uint64_t size) override;

  // Enqueues an asynchronous operation to zero out the specified device memory
  // region.
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override;

  // Enqueues an asynchronous copy from host memory to device memory.
  absl::Status Memcpy(DeviceAddressBase* gpu_dst, const void* host_src,
                      uint64_t size) override;

  // Enqueues an asynchronous copy from device memory to host memory.
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& gpu_src,
                      uint64_t size) override;

  // Enqueues an asynchronous copy from one device memory region to another.
  absl::Status Memcpy(DeviceAddressBase* gpu_dst,
                      const DeviceAddressBase& gpu_src, uint64_t size) override;

  // Enqueues a host callback to be executed after all previously enqueued
  // operations on the current stream have completed.
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;

  // Blocks the host until all previously enqueued operations on the current
  // stream have completed.
  absl::Status BlockHostUntilDone() override;

  // Returns a platform-specific handle for the underlying SYCL stream.
  Stream::PlatformSpecificHandle platform_specific_handle() const override {
    return {stream_handle_.get()};
  }

  // Creates an event-based timer for measuring elapsed time on the current
  // stream.
  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      bool use_delay_kernel) override {
    return executor_->CreateEventBasedTimer(this, use_delay_kernel);
  }

  // Creates a new SyclStream instance for the specified executor.
  // Supports optional stream priority and multiple streams per device.
  static absl::StatusOr<std::unique_ptr<SyclStream>> Create(
      StreamExecutor* executor, bool enable_multiple_streams,
      std::optional<std::variant<StreamPriority, int>> priority);

  // Destructor: waits for all pending operations on the stream to complete,
  // deallocates the stream from the executor, and destroys the underlying SYCL
  // stream.
  ~SyclStream() override;

  ::sycl::queue* stream_handle() const { return stream_handle_.get(); }

 private:
  SyclStream(StreamExecutor* executor, SyclEvent completed_event,
             std::optional<std::variant<StreamPriority, int>> priority,
             StreamPtr stream_handle)
      : StreamCommon(executor, priority),
        executor_(executor),
        completed_event_(std::move(completed_event)),
        stream_handle_(std::move(stream_handle)) {}

  // Updates 'completed_event_' to the most recent event available on the
  // current stream. This allows other streams or events to synchronize with
  // this point.
  // NOTE: This does *not* record a new event; it only copies the most recent
  // event. Actual event recording will be implemented in the future.
  absl::Status RecordCompletedEvent();

  // Launches a SYCL kernel on the current stream with the specified thread,
  // block, and optional cluster dimensions, kernel function, name, arguments,
  // and shared memory size.
  absl::Status LaunchKernel(const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const std::optional<ClusterDim>& cluster_dims,
                            void* function, absl::string_view name, void** args,
                            int64_t shmem_bytes) override;

  // The Executor to which this stream is bound.
  StreamExecutor* executor_;

  // The most recent event recorded on this stream, representing the completion
  // of all operations enqueued up to that point.
  SyclEvent completed_event_;

  // The underlying SYCL stream (queue).
  StreamPtr stream_handle_;
};

}  // namespace stream_executor::sycl
#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_STREAM_H_

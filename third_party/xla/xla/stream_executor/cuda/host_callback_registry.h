/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_HOST_CALLBACK_REGISTRY_H_
#define XLA_STREAM_EXECUTOR_CUDA_HOST_CALLBACK_REGISTRY_H_

#include <atomic>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace stream_executor::gpu {
class StreamStatusMonitor;
class HostCallbackRegistry;

// MPSC queue for host callbacks for a single stream.
//
// The single consumer part is somewhat loose because while the stream status
// monitor is the only thread that removes nodes from the queue during
// a normal run, in certain scenarios, like BlockHostUntilDone or different
// threads handling errors inside BlockhostUntilDone, we might have multiple
// threads competing for the removal of nodes. Consumer methods are hence mutex
// protected. Since these have very low contention, there is no expectation for
// this to be a bottleneck.
//
// All public methods are thread safe.
// The registry deliberately avoids using CUDA apis directly and instead accepts
// callbacks for scheduling and status checking, which keeps it testable without
// needing a GPU.
class StreamCallbackRegistry {
 public:
  struct HostCallbackNode;

  using StatusCb = absl::AnyInvocable<absl::Status()>;
  using DeviceCb = void (*)(void*);
  using EnqueueCb = absl::AnyInvocable<absl::Status(DeviceCb, /*data=*/void*)>;

  // Constructor.
  // synchronization_callback: Callback for a blocking call to synchronize the
  //                           stream such that all operations including host
  //                           callbacks are completed on the stream.
  // status_callback: Callback to be called asynchronously to poll the stream
  //                  status.
  // poll_interval: Interval between stream status polls.
  StreamCallbackRegistry(StatusCb synchronization_callback,
                         StatusCb status_callback);
  // Not allowed to move or copy.
  StreamCallbackRegistry(const StreamCallbackRegistry&) = delete;
  StreamCallbackRegistry& operator=(const StreamCallbackRegistry&) = delete;
  StreamCallbackRegistry(StreamCallbackRegistry&&) = delete;
  StreamCallbackRegistry& operator=(StreamCallbackRegistry&&) = delete;

  ~StreamCallbackRegistry();

  // Adds a callback to the registry and schedules it on the stream.
  // Producer method, Can be called by any thread.
  // callback: Callback to be scheduled on the stream.
  // error_cb: Callback to be called asynchronously if the above callback fails
  //           or cannot be scheduled.
  // enqueue_cb: Callback to use to schedule the callback on the stream after
  //             internal bookkeeping.
  absl::Status AddCallback(absl::AnyInvocable<absl::Status() &&> callback,
                           absl::AnyInvocable<void(absl::Status) &&> error_cb,
                           EnqueueCb enqueue_cb);

  // Marks callbacks as done with the given status but does not remove them
  // from the registry.
  // The assumption is that if FailAll is called, the stream
  // is not ok, so the host callback will not be scheduled. Consumer method,
  // mutex protected.
  void FailAll(absl::Status status);
  // Removes both cancelled and completed nodes from the registry.
  // Except the sentinel node.
  // Consumer method, mutex protected.
  void Prune();

 private:
  // Add a node to the registry.
  // Can be called by any thread.
  void AppendNode(std::unique_ptr<HostCallbackNode> node);
  void RefreshCallback(bool is_last_refresh);

 private:
  friend class StreamStatusMonitor;
  StatusCb synchronization_callback_;
  StatusCb status_callback_;
  std::atomic<HostCallbackNode*> head_{nullptr};
  std::atomic<HostCallbackNode*> tail_{nullptr};
  std::atomic<bool> is_shutting_down_{false};
  absl::Mutex mutex_;  // Used to guard the consumer side of the queue.
};

// Registry for callbacks across all streams.
// This class has two responsibilities:
// 1. Create RAII handles for streams to register with the registry.
// 2. Own a monitor thread that periodically polls the stream status for all
//    registered handles.
//
// All public methods are thread safe.
// Typically a stream would use CreateHandle to get a handle and then use
// AddCallback on the handle to add callbacks to the per stream registry.
// When the handle is destroyed, the stream is deregistered from the registry.
// Handles must be destroyed before the registry.
class HostCallbackRegistry {
 public:
  explicit HostCallbackRegistry(int device_ordinal,
                                absl::Duration poll_interval);
  ~HostCallbackRegistry();

  // RAII handle for StreamCallbackRegistry.
  // This is the primary interface for a stream to interact with the registry.
  class RegistryHandle {
   public:
    // Reexpose the callback types from StreamCallbackRegistry.
    using StatusCb = StreamCallbackRegistry::StatusCb;
    using EnqueueCb = StreamCallbackRegistry::EnqueueCb;
    using DeviceCb = StreamCallbackRegistry::DeviceCb;
    explicit RegistryHandle(
        HostCallbackRegistry* absl_nonnull host_callback_registry,
        RegistryHandle::StatusCb synchronization_callback,
        RegistryHandle::StatusCb status_callback);
    // Destructor.
    // Deregisters the handle from the registry before destroying the handle.
    ~RegistryHandle();

    // See StreamCallbackRegistry::AddCallback for argument details.
    absl::Status AddCallback(absl::AnyInvocable<absl::Status() &&> callback,
                             absl::AnyInvocable<void(absl::Status) &&> error_cb,
                             EnqueueCb enqueue_cb);

    // See StreamCallbackRegistry::FailAll for argument details.
    void FailAll(absl::Status status);

   private:
    HostCallbackRegistry* absl_nonnull host_callback_registry_;
    StreamCallbackRegistry handle_;
  };

  // Create an RAII handle that is deregistered from the registry upon
  // destruction.
  std::unique_ptr<RegistryHandle> CreateHandle(
      RegistryHandle::StatusCb synchronization_callback,
      RegistryHandle::StatusCb status_callback);

 private:
  // Registers a handle with the registry.
  // After this call returns, the handle is monitored by the stream status
  // monitor.
  void RegisterHandle(StreamCallbackRegistry* handle);

  // Deregisters a handle from the registry.
  // After this call returns, the handle is not monitored by the stream status
  // monitor anymore.
  void DeregisterHandle(StreamCallbackRegistry* handle);

  using RegistryHandles = std::vector<StreamCallbackRegistry*>;

  // Register/deregister a handle with the registry.
  void CopyOnWriteImpl(StreamCallbackRegistry* handle, bool is_deregister);
  // Returns a copy of the current registry handles.
  std::shared_ptr<RegistryHandles> GetCurrentStreamCallbackRegistrys();

  friend class StreamStatusMonitor;

  // Shared registry handles with the monitor thread.
  std::shared_ptr<RegistryHandles> registry_handles_ ABSL_GUARDED_BY(mutex_) =
      std::make_shared<RegistryHandles>();
  absl::Mutex mutex_;
  std::unique_ptr<StreamStatusMonitor> stream_status_monitor_;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_HOST_CALLBACK_REGISTRY_H_

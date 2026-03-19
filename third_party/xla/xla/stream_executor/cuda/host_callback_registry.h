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
#include <cstdint>
#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace stream_executor::gpu {

// MPSC queue for host callbacks.
//
// The single consumer part is somewhat loose because while the stream status
// monitor is the only thread that removes nodes from the queue during
// a normal run, in certain scenarios, like BlockHostUntilDone different threads
// handling errors inside BlockhostUntilDone, we might have multiple threads
// can be competing for the removal of nodes. Consumer methods are hence mutex
// protected. Since these have very low contention, there is no expectation for
// this to be a bottleneck.
//
// All public methods are thread safe.
// The registry deliberately avoids using CUDA apis directly and instead accepts
// callbacks for scheduling and status checking, which keeps it testable without
// needing a GPU.
class HostCallbackRegistry {
 public:
  struct HostCallbackNode;
  class StreamStatusMonitor;

  using StatusCb = absl::AnyInvocable<absl::Status()>;
  using DeviceCb = void (*)(void*);
  using EnqueueCb = absl::AnyInvocable<absl::Status(DeviceCb, /*data=*/void*)>;

  // Constructor.
  // device_ordinal: The device ordinal of the stream.
  // synchronization_callback: Callback for a blocking call to synchronize the
  //                           stream such that all operations including host
  //                           callbacks are completed on the stream.
  // status_callback: Callback to be called asynchronously to poll the stream
  //                  status.
  // poll_interval: Interval between stream status polls.
  HostCallbackRegistry(int32_t device_ordinal,
                       StatusCb synchronization_callback,
                       StatusCb status_callback, absl::Duration poll_interval);
  // Not allowed to move or copy.
  HostCallbackRegistry(const HostCallbackRegistry&) = delete;
  HostCallbackRegistry& operator=(const HostCallbackRegistry&) = delete;
  HostCallbackRegistry(HostCallbackRegistry&&) = delete;
  HostCallbackRegistry& operator=(HostCallbackRegistry&&) = delete;

  ~HostCallbackRegistry();

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

  // Removes both cancelled and completed nodes from the registry.
  // Except the sentinel node.
  // Consumer method, mutex protected.
  void Prune();

  // Marks callbacks as done with the given status but does not remove them
  // from the registry.
  // The assumption is that if FailAll is called, the stream
  // is not ok, so the host callback will not be scheduled. Consumer method,
  // mutex protected.
  void FailAll(absl::Status status);

 private:
  // Add a node to the registry.
  // Can be called by any thread.
  void AppendNode(std::unique_ptr<HostCallbackNode> node);

 private:
  std::atomic<HostCallbackNode*> head_{nullptr};
  std::atomic<HostCallbackNode*> tail_{nullptr};
  std::atomic<bool> is_shutting_down_{false};
  absl::Mutex mutex_;  // Used to guard the consumer side of the queue.
  std::unique_ptr<StreamStatusMonitor> stream_status_monitor_;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_HOST_CALLBACK_REGISTRY_H_

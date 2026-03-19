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

#include "xla/stream_executor/cuda/host_callback_registry.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor::gpu {

// Singly linked list of host callbacks.
struct HostCallbackRegistry::HostCallbackNode {
  HostCallbackNode() = default;
  HostCallbackNode(absl::AnyInvocable<absl::Status() &&> callback,
                   absl::AnyInvocable<void(absl::Status) &&> error_cb)
      : callback(std::move(callback)), error_cb(std::move(error_cb)) {}

  void ResetCallbacks() {
    callback = nullptr;
    error_cb = nullptr;
    is_deletable.store(true, std::memory_order_release);
  }

  bool MarkDone() { return is_done.exchange(true, std::memory_order_release); }

  void Cancel(absl::Status status) {
    if (MarkDone()) {
      return;
    }
    if (error_cb) {
      std::move(error_cb)(status);
    }
    ResetCallbacks();
  }

  void operator()() {
    if (MarkDone()) {
      return;
    }
    if (callback) {
      absl::Status status = std::move(callback)();
      if (!status.ok()) {
        if (error_cb) {
          std::move(error_cb)(status);
        } else {
          LOG(ERROR) << "Host callback failed: " << status;
        }
      }
    }
    ResetCallbacks();
  }

  absl::AnyInvocable<absl::Status() &&> callback{nullptr};
  absl::AnyInvocable<void(absl::Status) &&> error_cb{nullptr};
  std::atomic<bool> is_done{false};
  std::atomic<bool> is_deletable{false};
  // Singly linked list.
  std::atomic<HostCallbackNode*> next{nullptr};
};

// Monitor thread for the stream.
class HostCallbackRegistry::StreamStatusMonitor {
 public:
  using StatusCb = absl::AnyInvocable<absl::Status()>;
  StreamStatusMonitor(int32_t device_ordinal, StatusCb synchronization_callback,
                      StatusCb status_callback,
                      HostCallbackRegistry& host_callback_registry,
                      absl::Duration poll_interval)
      : synchronization_callback_(std::move(synchronization_callback)),
        status_callback_(std::move(status_callback)),
        host_callback_registry_(host_callback_registry),
        poll_interval_(poll_interval) {
    thread_ = std::unique_ptr<tsl::Thread>(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(),
        absl::StrFormat("cuda_stream_monitor_%d", device_ordinal),
        [&] { MonitorLoop(); }));
  }

  // Loop that periodically queries the stream status.
  void MonitorLoop() {
    auto const refresh_cb = [&] {
      if (auto status = status_callback_(); !status.ok()) {
        // At this point either all host callbacks have been scheduled or the
        // driver has discarded the remaining callbacks because of stream
        // poisoning.
        synchronization_callback_().IgnoreError();
        host_callback_registry_.FailAll(status);
      }
      host_callback_registry_.Prune();
    };
    while (!stop_.WaitForNotificationWithTimeout(poll_interval_)) {
      refresh_cb();
    }
    // Refresh one last time before exiting.
    // Synchronize to make sure all callbacks are scheduled on the stream before
    // exiting.
    synchronization_callback_().IgnoreError();
    refresh_cb();
  }

  // Blocks until the monitor thread is stopped.
  void stop() {
    if (thread_) {
      stop_.Notify();
      thread_.reset();
    }
  }

  ~StreamStatusMonitor() { stop(); }

 private:
  StatusCb synchronization_callback_;
  StatusCb status_callback_;
  HostCallbackRegistry& host_callback_registry_;
  std::unique_ptr<tsl::Thread> thread_;
  absl::Duration poll_interval_;
  absl::Notification stop_;
};

namespace {
static constexpr auto kCudaCallback = [](void* data) {
  auto* callback =
      // Casting here is a CUDA API requirement.
      // NOLINTNEXTLINE(custom-reinterpret-cast)
      reinterpret_cast<HostCallbackRegistry::HostCallbackNode*>(data);
  (*callback)();
};
}  // namespace

HostCallbackRegistry::HostCallbackRegistry(int32_t device_ordinal,
                                           StatusCb synchronization_callback,
                                           StatusCb status_callback,
                                           absl::Duration poll_interval) {
  auto sentinel = new HostCallbackNode();
  // Invoke it to mark it done and deletable for consistency.
  sentinel->operator()();
  head_.store(sentinel, std::memory_order_relaxed);
  tail_.store(sentinel, std::memory_order_relaxed);
  stream_status_monitor_ = std::make_unique<StreamStatusMonitor>(
      device_ordinal, std::move(synchronization_callback),
      std::move(status_callback), *this, poll_interval);
}

HostCallbackRegistry::~HostCallbackRegistry() {
  stream_status_monitor_->stop();
  FailAll(absl::CancelledError("Registry shutting down"));
  Prune();
  delete head_.load(std::memory_order_acquire);  // Remove the sentinel.
}

absl::Status HostCallbackRegistry::AddCallback(
    absl::AnyInvocable<absl::Status() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb, EnqueueCb enqueue_cb) {
  const auto cancellation_error = absl::CancelledError(
      "Callback registry is closed. This usually means that the stream is in "
      "an error state or being destroyed.");
  // Early exit if the registry is already shut down.
  if (is_shutting_down_.load(std::memory_order_acquire)) {
    if (error_cb) {
      std::move(error_cb)(cancellation_error);
    }
    return cancellation_error;
  }
  auto node =
      std::make_unique<HostCallbackNode>(/*.callback =*/std::move(callback),
                                         /*.error_cb =*/std::move(error_cb));
  if (const absl::Status status = enqueue_cb(kCudaCallback, node.get());
      !status.ok()) {
    node->Cancel(status);
    return status;
  }
  AppendNode(std::move(node));
  return absl::OkStatus();
}

void HostCallbackRegistry::Prune() {
  absl::MutexLock lock(mutex_);
  HostCallbackNode* curr = head_.load(std::memory_order_acquire);
  while (curr->is_deletable.load(std::memory_order_acquire)) {
    HostCallbackNode* next = curr->next.load(std::memory_order_acquire);
    if (!next) {  // Reached the tail.
      break;
    }
    head_.store(next, std::memory_order_release);
    delete curr;
    curr = next;
  }
}

void HostCallbackRegistry::FailAll(absl::Status status) {
  absl::MutexLock lock(mutex_);
  is_shutting_down_.store(true, std::memory_order_release);
  HostCallbackNode* curr = head_.load(std::memory_order_acquire);
  while (curr) {
    curr->Cancel(status);
    curr = curr->next.load(std::memory_order_acquire);
  }
}

void HostCallbackRegistry::AppendNode(std::unique_ptr<HostCallbackNode> node) {
  // On the off chance that the registry is shutting down now we have 2
  // possibilities:
  // 1. We manage to add the node before FailAll reaches the tail. FailAll takes
  // care of marking the node done.
  // 2. FailAll has already reached the tail and we add a new node to the end
  // which was not marked done. In this case, the destructor will take care of
  // the cleanup.
  HostCallbackNode* node_ptr = node.release();
  // Prev is guaranteed to be non-null because of the sentinel.
  HostCallbackNode* prev = tail_.exchange(node_ptr, std::memory_order_acq_rel);
  prev->next.store(node_ptr, std::memory_order_release);
}

}  // namespace stream_executor::gpu

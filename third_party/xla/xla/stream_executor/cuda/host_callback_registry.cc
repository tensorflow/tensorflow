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

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
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
struct StreamCallbackRegistry::HostCallbackNode {
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

void StreamCallbackRegistry::RefreshCallback(bool is_last_refresh) {
  const auto status =
      is_last_refresh ? absl::CancelledError("Shutting down registry handle.")
                      : status_callback_();
  if (!status.ok()) {
    // At this point either all host callbacks have been scheduled or the
    // driver has discarded the remaining callbacks because of stream
    // poisoning.
    synchronization_callback_().IgnoreError();
    FailAll(status);
  }
  Prune();
}

// Monitor thread for the stream.
class StreamStatusMonitor {
  // RAII handle for start/stop the monitor loop.
  class MonitoringContext {
   public:
    explicit MonitoringContext(StreamStatusMonitor& monitor)
        : monitor_(&monitor) {
      {
        absl::MutexLock lock(monitor_->mutex_);
        monitor_->is_waiting_ = false;
      }
      // Copies the pointer to the registry handles.
      local_copy_ =
          monitor_->host_callback_registry_.GetCurrentStreamCallbackRegistrys();
    }

    const HostCallbackRegistry::RegistryHandles& LocalCopy() const {
      return *local_copy_;
    }

    ~MonitoringContext() {
      absl::MutexLock lock(monitor_->mutex_);
      monitor_->is_waiting_ = true;
    }

   private:
    std::shared_ptr<const HostCallbackRegistry::RegistryHandles> local_copy_;
    StreamStatusMonitor* monitor_;
  };

 public:
  using StatusCb = absl::AnyInvocable<absl::Status()>;
  StreamStatusMonitor(HostCallbackRegistry& host_callback_registry,
                      int device_ordinal, absl::Duration poll_interval)
      : host_callback_registry_(host_callback_registry),
        poll_interval_(poll_interval) {
    thread_ = std::unique_ptr<tsl::Thread>(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(),
        absl::StrFormat("cuda_stream_monitor_%d", device_ordinal),
        [&] { MonitorLoop(); }));
  }

  // Loop that periodically queries the stream status for all registered
  // streams.
  void MonitorLoop() {
    std::string thread_name;
    tsl::Env::Default()->GetCurrentThreadName(&thread_name);
    VLOG(5) << "MonitorLoop started on thread " << thread_name;
    while (!stop_.WaitForNotificationWithTimeout(poll_interval_)) {
      MonitoringContext context(*this);
      for (StreamCallbackRegistry* handle : context.LocalCopy()) {
        handle->RefreshCallback(/*is_last_refresh=*/false);
      }
    }
    // Final cleanup.
    MonitoringContext context(*this);
    for (StreamCallbackRegistry* handle : context.LocalCopy()) {
      handle->RefreshCallback(/*is_last_refresh=*/true);
    }
    VLOG(5) << "MonitorLoop stopped on thread " << thread_name;
  }

  void WaitUntilLoopStopped() {
    absl::MutexLock lock(mutex_);
    mutex_.Await(absl::Condition(&is_waiting_));
  }

  // Blocks until the monitor thread is stopped.
  void Stop() {
    if (thread_) {
      stop_.Notify();
      thread_.reset();
    }
  }

  ~StreamStatusMonitor() { Stop(); }

 private:
  HostCallbackRegistry& host_callback_registry_;
  std::unique_ptr<tsl::Thread> thread_;
  absl::Duration poll_interval_;
  absl::Notification stop_;
  absl::Mutex mutex_;
  bool is_waiting_ ABSL_GUARDED_BY(mutex_) = true;
};

namespace {
static constexpr auto kCudaCallback = [](void* data) {
  auto* callback =
      // Casting here is a CUDA API requirement.
      // NOLINTNEXTLINE(custom-reinterpret-cast)
      reinterpret_cast<StreamCallbackRegistry::HostCallbackNode*>(data);
  (*callback)();
};
}  // namespace

StreamCallbackRegistry::StreamCallbackRegistry(
    StatusCb synchronization_callback, StatusCb status_callback)
    : synchronization_callback_(std::move(synchronization_callback)),
      status_callback_(std::move(status_callback)) {
  auto sentinel = new HostCallbackNode();
  // Invoke it to mark it done and deletable for consistency.
  sentinel->operator()();
  head_.store(sentinel, std::memory_order_relaxed);
  tail_.store(sentinel, std::memory_order_relaxed);
}

StreamCallbackRegistry::~StreamCallbackRegistry() {
  RefreshCallback(/*is_last_refresh=*/true);
  delete head_.load(std::memory_order_acquire);  // Remove the sentinel.
}

absl::Status StreamCallbackRegistry::AddCallback(
    absl::AnyInvocable<absl::Status() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb, EnqueueCb enqueue_cb) {
  const auto cancellation_error = absl::CancelledError(
      "Callback registry is closed. This usually means that the stream is in "
      "an error state or being destroyed.");
  // Early exit if this handle is already shut down.
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

void StreamCallbackRegistry::Prune() {
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

void StreamCallbackRegistry::FailAll(absl::Status status) {
  absl::MutexLock lock(mutex_);
  is_shutting_down_.store(true, std::memory_order_release);
  HostCallbackNode* curr = head_.load(std::memory_order_acquire);
  while (curr) {
    curr->Cancel(status);
    curr = curr->next.load(std::memory_order_acquire);
  }
}

void StreamCallbackRegistry::AppendNode(
    std::unique_ptr<HostCallbackNode> node) {
  // On the off chance that the registry is shutting down now we have 2
  // possibilities:
  // 1. We manage to add the node before FailAll reaches the tail. FailAll
  // takes care of marking the node done.
  // 2. FailAll has already reached the tail and we add a new node to the end
  // which was not marked done. In this case, the destructor will take care of
  // the cleanup.
  HostCallbackNode* node_ptr = node.release();
  // Prev is guaranteed to be non-null because of the sentinel.
  HostCallbackNode* prev = tail_.exchange(node_ptr, std::memory_order_acq_rel);
  prev->next.store(node_ptr, std::memory_order_release);
}

HostCallbackRegistry::HostCallbackRegistry(int device_ordinal,
                                           absl::Duration poll_interval)
    : stream_status_monitor_(std::make_unique<StreamStatusMonitor>(
          *this, device_ordinal, poll_interval)) {};

HostCallbackRegistry::~HostCallbackRegistry() {
  stream_status_monitor_->Stop();
}

std::unique_ptr<HostCallbackRegistry::RegistryHandle>
HostCallbackRegistry::CreateHandle(
    RegistryHandle::StatusCb synchronization_callback,
    RegistryHandle::StatusCb status_callback) {
  return std::make_unique<RegistryHandle>(
      this, std::move(synchronization_callback), std::move(status_callback));
}

void HostCallbackRegistry::RegisterHandle(StreamCallbackRegistry* handle) {
  CopyOnWriteImpl(handle, /*is_deregister=*/false);
}

void HostCallbackRegistry::DeregisterHandle(StreamCallbackRegistry* handle) {
  CopyOnWriteImpl(handle,
                  /*is_deregister=*/true);
  // After this the old registry handles are not used anymore.
  stream_status_monitor_->WaitUntilLoopStopped();
}

// Since we expect the number of streams to be relatively small, and for
// registration and deregistration to only happen during stream creation and
// destruction, this should be relatively inexpensive.
void HostCallbackRegistry::CopyOnWriteImpl(StreamCallbackRegistry* handle,
                                           bool is_deregister) {
  absl::MutexLock lock(mutex_);
  auto next = std::make_shared<RegistryHandles>(*registry_handles_);
  if (is_deregister) {
    next->erase(std::remove(next->begin(), next->end(), handle), next->end());
  } else {
    next->push_back(handle);
  }
  registry_handles_ = std::move(next);
}

std::shared_ptr<HostCallbackRegistry::RegistryHandles>
HostCallbackRegistry::GetCurrentStreamCallbackRegistrys() {
  absl::MutexLock lock(mutex_);
  return registry_handles_;
}

HostCallbackRegistry::RegistryHandle::RegistryHandle(
    HostCallbackRegistry* absl_nonnull host_callback_registry,
    StatusCb synchronization_callback, StatusCb status_callback)
    : host_callback_registry_(host_callback_registry),
      handle_(std::move(synchronization_callback), std::move(status_callback)) {
  host_callback_registry_->RegisterHandle(&handle_);
}

HostCallbackRegistry::RegistryHandle::~RegistryHandle() {
  host_callback_registry_->DeregisterHandle(&handle_);
}

absl::Status HostCallbackRegistry::RegistryHandle::AddCallback(
    absl::AnyInvocable<absl::Status() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb, EnqueueCb enqueue_cb) {
  return handle_.AddCallback(std::move(callback), std::move(error_cb),
                             std::move(enqueue_cb));
}

void HostCallbackRegistry::RegistryHandle::FailAll(absl::Status status) {
  handle_.FailAll(status);
}

}  // namespace stream_executor::gpu

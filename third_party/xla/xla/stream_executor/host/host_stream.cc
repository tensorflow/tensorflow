/* Copyright 2016 The OpenXLA Authors.

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

// Class method definitions for HostStream, the Stream implementation for
// the HostExecutor implementation.
#include "xla/stream_executor/host/host_stream.h"

#include <string.h>

#include <cfenv>  // NOLINT
#include <cstdint>
#include <memory>
#include <queue>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/host/host_event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/context.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/setround.h"

namespace stream_executor {
namespace host {

HostStream::HostStream(StreamExecutor* executor)
    : StreamCommon(executor),
      thread_(tsl::Env::Default()->StartThread({}, "host_executor",
                                               [this]() { WorkLoop(); })) {}

HostStream::~HostStream() {
  {
    absl::MutexLock lock(&mu_);
    work_queue_.push(WorkItem(nullptr));
  }
  // thread_'s destructor blocks until the thread finishes running.
  thread_.reset();
  parent()->DeallocateStream(this);
}

absl::Status HostStream::Memcpy(DeviceMemoryBase* gpu_dst,
                                const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  // Enqueue this [asynchronous] "device-to-device" (i.e., host-to-host, given
  // the nature of the HostExecutor) memcpy  on the stream (HostStream)
  // associated with the HostExecutor.
  EnqueueTask([src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return absl::OkStatus();
}

absl::Status HostStream::Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  EnqueueTask([host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return absl::OkStatus();
}

absl::Status HostStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  EnqueueTask([dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return absl::OkStatus();
}

absl::Status HostStream::Memset32(DeviceMemoryBase* location, uint32_t pattern,
                                  uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  EnqueueTask([gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return absl::OkStatus();
}

absl::Status HostStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  EnqueueTask([gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return absl::OkStatus();
}

absl::Status HostStream::WaitFor(Stream* other) {
  auto event = std::make_shared<absl::Notification>();
  static_cast<HostStream*>(other)->EnqueueTask([event]() { event->Notify(); });
  EnqueueTask([event]() { event->WaitForNotification(); });
  return absl::OkStatus();
}

absl::Status HostStream::WaitFor(Event* event) {
  std::shared_ptr<absl::Notification> notification =
      static_cast<HostEvent*>(event)->notification();
  EnqueueTask([notification]() { notification->WaitForNotification(); });
  return absl::OkStatus();
}

bool HostStream::EnqueueTask(absl::AnyInvocable<void() &&> task) {
  return EnqueueTaskWithStatus([task = std::move(task)]() mutable {
    std::move(task)();
    return absl::OkStatus();
  });
}

absl::Status HostStream::RecordEvent(Event* event) {
  std::shared_ptr<absl::Notification> notification =
      static_cast<HostEvent*>(event)->notification();
  EnqueueTask([notification]() {
    CHECK(!notification->HasBeenNotified());
    notification->Notify();
  });
  return absl::OkStatus();
}

absl::Status HostStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  if (EnqueueTaskWithStatus(std::move(callback))) {
    return absl::OkStatus();
  }
  return absl::InternalError("Failed to host callback.");
}

bool HostStream::EnqueueTaskWithStatus(
    absl::AnyInvocable<absl::Status() &&> task) {
  CHECK(task != nullptr);
  absl::MutexLock lock(&mu_);
  work_queue_.push(WorkItem(std::move(task)));
  return true;
}

bool HostStream::WorkAvailable() { return !work_queue_.empty(); }

void HostStream::WorkLoop() {
  // Set denormal and rounding behavior to match the default TF ThreadPool
  // behavior.
  // TODO(phawkins, jlebar): it's not clear this is the best place to set this.
  tsl::port::ScopedFlushDenormal flush;
  tsl::port::ScopedSetRound round(FE_TONEAREST);
  while (true) {
    std::queue<WorkItem> queue;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(absl::Condition(this, &HostStream::WorkAvailable));
      std::swap(queue, work_queue_);
    }
    while (!queue.empty()) {
      WorkItem& work_item = queue.front();
      if (!work_item.task) {
        return;
      }
      {  // Don't destroy the context until the task is done.
        tsl::WithContext with_context(work_item.context);
        status_.Update(std::move(work_item.task)());
      }
      queue.pop();
    }
  }
}

absl::Status HostStream::BlockUntilDone() {
  absl::Notification done;
  absl::Status status;
  EnqueueTask([&done, &status, this]() {
    // This task is always executed synchronously before 'status_' is updated
    // with the result of the task (always OK() in this case), so we don't need
    // to worry about locking access to 'status_'.
    status = status_;
    status_ = absl::OkStatus();
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}

}  // namespace host
}  // namespace stream_executor

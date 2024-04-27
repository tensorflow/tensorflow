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

#include <cfenv>  // NOLINT
#include <cstddef>
#include <queue>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/env.h"
#include "tsl/platform/setround.h"

namespace stream_executor {
namespace host {

HostStream::HostStream()
    : thread_(tsl::Env::Default()->StartThread({}, "host_executor",
                                               [this]() { WorkLoop(); })) {}

HostStream::~HostStream() {
  {
    absl::MutexLock lock(&mu_);
    work_queue_.push(nullptr);
  }
  // thread_'s destructor blocks until the thread finishes running.
  thread_.reset();
}

bool HostStream::EnqueueTask(absl::AnyInvocable<void() &&> task) {
  return EnqueueTaskWithStatus([task = std::move(task)]() mutable {
    std::move(task)();
    return absl::OkStatus();
  });
}

bool HostStream::EnqueueTaskWithStatus(
    absl::AnyInvocable<absl::Status() &&> task) {
  CHECK(task != nullptr);
  absl::MutexLock lock(&mu_);
  work_queue_.push(std::move(task));
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
    std::queue<absl::AnyInvocable<absl::Status() &&>> queue;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(absl::Condition(this, &HostStream::WorkAvailable));
      std::swap(queue, work_queue_);
    }
    while (!queue.empty()) {
      absl::AnyInvocable<absl::Status() &&>& fn = queue.front();
      if (!fn) {
        return;
      }
      status_.Update(std::move(fn)());
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

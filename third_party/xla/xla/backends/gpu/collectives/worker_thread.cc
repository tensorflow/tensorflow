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

#include "xla/backends/gpu/collectives/worker_thread.h"

#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/env.h"

namespace xla::gpu {

WorkerThread::WorkerThread(tsl::Env& env, absl::string_view thread_name) {
  auto work_loop = [this] {
    VLOG(1) << "Started WorkerThread";
    absl::MutexLock l(&mu_);
    while (true) {
      // Wait until the WorkerThread has been shut down or until work is ready.
      auto shutdown_or_work_ready = [this]() {
        mu_.AssertHeld();
        return shutdown_ || !work_.empty();
      };
      mu_.Await(absl::Condition(&shutdown_or_work_ready));

      // If the WorkerThread is shutting down, cancel and notify all pending
      // work.
      if (shutdown_) {
        VLOG(1) << "WorkerThread shutting down";
        for (WorkItem* w : work_) {
          w->status = absl::InternalError("shutdown");
          w->done.Notify();
        }
        return;
      }

      // Complete the work item at the front of the queue.
      VLOG(3) << "WorkerThread executing work item";
      WorkItem* w = work_.front();
      work_.pop_front();
      w->status = std::move(w->f)();
      w->done.Notify();
    }
  };

  thread_ = absl::WrapUnique(env.StartThread(
      tsl::ThreadOptions(), std::string(thread_name), work_loop));
}

WorkerThread::~WorkerThread() {
  {
    absl::MutexLock l(&mu_);
    shutdown_ = true;
  }

  // Wait for the background thread to terminate.
  thread_ = nullptr;
}

absl::Status WorkerThread::Run(absl::AnyInvocable<absl::Status() &&> f) {
  WorkItem w;
  w.f = std::move(f);
  {
    absl::MutexLock l(&mu_);
    work_.push_back(&w);
  }
  w.done.WaitForNotification();
  return w.status;
}

}  // namespace xla::gpu

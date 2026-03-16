/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/worker_thread.h"

#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/env.h"

namespace xla {

WorkerThread::WorkerThread(tsl::Env* env, const std::string& name)
    : WorkerThread(env, tsl::ThreadOptions(), name) {}

WorkerThread::WorkerThread(tsl::Env* env, const tsl::ThreadOptions& options,
                           const std::string& name) {
  thread_.reset(env->StartThread(options, name, [this]() { WorkLoop(); }));
}

WorkerThread::~WorkerThread() {
  absl::MutexLock lock(mu_);
  work_queue_.push(absl::AnyInvocable<void() &&>{nullptr});
}

void WorkerThread::Schedule(absl::AnyInvocable<void() &&> fn) {
  CHECK(fn != nullptr);
  absl::MutexLock lock(mu_);
  work_queue_.push(std::move(fn));
}

void AsyncValueContinuation::Run() && {
  if (fn) {
    for (const tsl::RCReference<tsl::AsyncValue>& dep : deps) {
      tsl::BlockUntilReady(&(*dep));
    }
    std::move(fn)();
  }
}

void WorkerThread::ScheduleWithOrderedContinuations(
    absl::AnyInvocable<AsyncValueContinuation() &&> fn) {
  CHECK(fn != nullptr);
  absl::MutexLock lock(mu_);
  work_queue_.push(std::move(fn));
}

bool WorkerThread::WorkAvailable() {
  if (!continuation_queue_.empty() &&
      completed_continuation_waiter_ ==
          continuation_queue_.front().deps.size()) {
    return true;
  }
  return !work_queue_.empty();
}

void WorkerThread::ScheduleNextWaiter() {
  CHECK(!continuation_queue_.empty());
  AsyncValueContinuation& continuation = continuation_queue_.front();
  for (int i = completed_continuation_waiter_ + 1;; ++i) {
    if (i == continuation.deps.size()) {
      completed_continuation_waiter_ = i;
      break;
    }
    if (continuation.deps[i]->IsAvailable()) {
      completed_continuation_waiter_ = i;
    } else {
      auto dep = continuation.deps[i];
      mu_.unlock();
      dep->AndThen([this]() {
        absl::MutexLock lock(mu_);
        ScheduleNextWaiter();
      });
      mu_.lock();
      break;
    }
  }
}

void WorkerThread::WorkLoop() {
  bool shutdown = false;
  absl::MutexLock lock(mu_);
  while (true) {
    if (shutdown && continuation_queue_.empty()) {
      break;
    }
    mu_.Await(absl::Condition(this, &WorkerThread::WorkAvailable));
    if (!continuation_queue_.empty() &&
        completed_continuation_waiter_ ==
            continuation_queue_.front().deps.size()) {
      AsyncValueContinuation& continuation = continuation_queue_.front();
      mu_.unlock();
      std::move(continuation.fn)();
      continuation.fn = nullptr;
      mu_.lock();
      continuation_queue_.pop();
      ++debug_queue_idx;
      completed_continuation_waiter_ = -1;
      if (!continuation_queue_.empty()) {
        ScheduleNextWaiter();
      }
    } else {
      WorkUnit work = std::move(work_queue_.front());
      work_queue_.pop();
      if (std::holds_alternative<absl::AnyInvocable<void() &&>>(work)) {
        auto fn = std::get<absl::AnyInvocable<void() &&>>(std::move(work));
        if (!fn) {
          shutdown = true;
        } else {
          mu_.unlock();
          std::move(fn)();
          fn = nullptr;
          mu_.lock();
        }
      } else if (std::holds_alternative<
                     absl::AnyInvocable<AsyncValueContinuation() &&>>(work)) {
        mu_.unlock();
        AsyncValueContinuation continuation =
            std::get<absl::AnyInvocable<AsyncValueContinuation() &&>>(
                std::move(work))();
        work = absl::AnyInvocable<void() &&>{nullptr};
        mu_.lock();
        continuation_queue_.push(std::move(continuation));
        if (continuation_queue_.size() == 1) {
          ScheduleNextWaiter();
        }
      }
    }
  }
}

}  // namespace xla

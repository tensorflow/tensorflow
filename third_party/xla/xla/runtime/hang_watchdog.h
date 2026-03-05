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

#ifndef XLA_RUNTIME_HANG_WATCHDOG_H_
#define XLA_RUNTIME_HANG_WATCHDOG_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

// HangWatchdog is responsible for watching XLA actions to make progress and
// finish in a reasonable time. If the action doesn't complete in the given
// timeout, then the watchdog calls the cancellation callback, which typically
// aborts the process if there is no other way to safely cancel the action.
//
// For example, if XLA:GPU execution doesn't finish under a minute, it is
// reasonable to assume that it deadlocked in one of the GPU kernels or inside
// a collective operation. There is no way to recover from such a deadlock
// except aborting the process. In a distributed environment, XLA tracks alive
// devices using `IncarnationId`, but this might not be enough to detect hangs,
// as all processes are alive but not making any progress.
//
// IMPORTANT: Cancellation callbacks must be cheap; running expensive work
// inside the callback can delay processing of other guards. Be careful with
// setting AsyncValue/Future in the callback, as it will process all OnReady
// callbacks in the caller thread.
class HangWatchdog {
 public:
  // Forward declare.
  struct Guard;

  // Cancel callback has to be copyable because we pass it to thread pool.
  using CancelCallback = absl::AnyInvocable<void() &&>;

  HangWatchdog(tsl::Env* env, absl::string_view name, size_t num_threads = 1);

  // Returns a cancel callback that will abort the process. If `pre_abort` is
  // provided, it will be called before the process is aborted.
  static CancelCallback Abort(absl::string_view action, absl::Duration duration,
                              CancelCallback pre_abort = nullptr);

  // Issues a watch guard to the caller indicating that the watchdog has started
  // tracking the action and expects it to finish in the given duration. If the
  // action doesn't finish within the given duration, the watchdog will call the
  // `cancel` callback. The action is considered completed when the watch guard
  // is destructed. It is up to the caller to make sure that the guard is
  // destroyed when the action is completed.
  std::shared_ptr<Guard> Watch(absl::string_view action,
                               absl::Duration duration, CancelCallback cancel);

 private:
  // Schedules a single check into the thread pool. If the guard is still alive
  // and the deadline hasn't passed, sleeps for `sleep_interval` and then
  // recursively schedules the next check with a doubled interval. This avoids
  // head-of-line blocking a thread pool thread for the entire watch duration.
  void ScheduleCheck(std::weak_ptr<Guard> guard, absl::Duration sleep_interval);

  // Finds and erases one timed-out guard. Returns the timed-out guard (or
  // nullptr) and the nearest deadline among processed non-timed-out guards.
  std::pair<std::shared_ptr<Guard>, absl::Time> ExtractTimedOutGuard();

  absl::Mutex mu_;
  std::vector<std::weak_ptr<Guard>> guards_ ABSL_GUARDED_BY(mu_);

  // Thread pool must be declared last so it is destroyed first, joining all
  // pending ScheduleCheck tasks before `mu_` and `guards_` are destroyed.
  tsl::thread::ThreadPool thread_pool_;
};

}  // namespace xla

#endif  // XLA_RUNTIME_HANG_WATCHDOG_H_

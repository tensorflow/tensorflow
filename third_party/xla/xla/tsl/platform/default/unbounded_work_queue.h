/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_TSL_PLATFORM_DEFAULT_UNBOUNDED_WORK_QUEUE_H_
#define XLA_TSL_PLATFORM_DEFAULT_UNBOUNDED_WORK_QUEUE_H_

#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/env.h"

namespace tsl {

// An `UnboundedWorkQueue` provides a mechanism for temporally multiplexing a
// potentially large number of "logical" threads onto a smaller number of
// "physical" threads. The multiplexing is achieved by maintaining an internal
// pool of long-running "physical" threads that are used to execute the
// "logical" threads.  Like a regular thread, a "logical" thread may block on
// other threads, and the size of the pool will increase to ensure that progress
// is made. This mechanism is recommended in situations where short-lived
// threads are created repeatedly, to avoid the overhead and memory
// fragmentation that can result from excessive thread creation.
class UnboundedWorkQueue {
 public:
  UnboundedWorkQueue(Env* env, absl::string_view thread_name,
                     const ThreadOptions& thread_options = {});
  ~UnboundedWorkQueue();

  using WorkFunction = std::function<void()>;

  // Schedule `fn` on a thread.  `fn` may perform blocking work, so if all the
  // existing threads are blocked or busy, this may spawn a new thread which
  // will be added to the thread pool managed by this work queue.
  void Schedule(WorkFunction fn);

 private:
  void PooledThreadFunc();

  bool HasWorkOrIsCancelled() const ABSL_SHARED_LOCKS_REQUIRED(work_queue_mu_) {
    return !work_queue_.empty() || cancelled_;
  }

  Env* const env_;  // Not owned.
  const std::string thread_name_;
  const ThreadOptions thread_options_;
  absl::Mutex work_queue_mu_;
  size_t num_idle_threads_ ABSL_GUARDED_BY(work_queue_mu_) = 0;
  bool cancelled_ ABSL_GUARDED_BY(work_queue_mu_) = false;
  std::deque<WorkFunction> work_queue_ ABSL_GUARDED_BY(work_queue_mu_);
  absl::Mutex thread_pool_mu_;
  std::vector<std::unique_ptr<Thread>> thread_pool_
      ABSL_GUARDED_BY(thread_pool_mu_);
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_DEFAULT_UNBOUNDED_WORK_QUEUE_H_

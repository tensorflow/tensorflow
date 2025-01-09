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

#include "xla/tsl/platform/default/unbounded_work_queue.h"

#include <utility>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/env.h"
#include "tsl/platform/numa.h"

namespace tsl {

UnboundedWorkQueue::UnboundedWorkQueue(Env* env, absl::string_view thread_name,
                                       const ThreadOptions& thread_options)
    : env_(env), thread_name_(thread_name), thread_options_(thread_options) {}

UnboundedWorkQueue::~UnboundedWorkQueue() {
  {
    absl::MutexLock l(&work_queue_mu_);
    // Wake up all `PooledThreadFunc` threads and cause them to terminate before
    // joining them when `threads_` is cleared.
    cancelled_ = true;
    if (!work_queue_.empty()) {
      LOG(ERROR) << "UnboundedWorkQueue named \"" << thread_name_ << "\" was "
                 << "deleted with pending work in its queue. This may indicate "
                 << "a potential use-after-free bug.";
    }
  }

  {
    absl::MutexLock l(&thread_pool_mu_);
    // Clear the list of pooled threads, which will eventually terminate due to
    // the previous notification.
    //
    // NOTE: It is safe to do this while holding `thread_pool_mu_`, because
    // no subsequent calls to `this->Schedule()` should be issued after the
    // destructor starts.
    thread_pool_.clear();
  }
}

void UnboundedWorkQueue::Schedule(WorkFunction fn) {
  // Enqueue a work item for the new thread's function, and wake up a
  // cached thread to process it.
  absl::MutexLock l(&work_queue_mu_);
  work_queue_.push_back(std::move(fn));
  // NOTE: The queue may be non-empty, so we must account for queued work when
  // considering how many threads are free.
  if (work_queue_.size() > num_idle_threads_) {
    // Spawn a new physical thread to process the given function.
    // NOTE: `PooledThreadFunc` will eventually increment `num_idle_threads_`
    // at the beginning of its work loop.
    Thread* new_thread =
        env_->StartThread({}, thread_name_, [this]() { PooledThreadFunc(); });

    absl::MutexLock l(&thread_pool_mu_);
    thread_pool_.emplace_back(new_thread);
  }
}

void UnboundedWorkQueue::PooledThreadFunc() {
  // If specified, make sure the thread runs on the correct NUMA node.
  if (thread_options_.numa_node != tsl::port::kNUMANoAffinity) {
    tsl::port::NUMASetThreadNodeAffinity(thread_options_.numa_node);
  }

  while (true) {
    WorkFunction fn;
    {
      absl::MutexLock l(&work_queue_mu_);
      ++num_idle_threads_;
      // Wait for a new work function to be submitted, or the cache to be
      // destroyed.
      work_queue_mu_.Await(
          absl::Condition(this, &UnboundedWorkQueue::HasWorkOrIsCancelled));
      if (cancelled_) {
        return;
      }
      fn = std::move(work_queue_.front());
      work_queue_.pop_front();
      --num_idle_threads_;
    }

    fn();
  }
}

}  // namespace tsl

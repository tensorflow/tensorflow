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

#include "tensorflow/core/kernels/data/unbounded_thread_pool.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace data {

// A lightweight wrapper for creating logical threads in a `UnboundedThreadPool`
// that can be shared (e.g.) in an `IteratorContext`.
class UnboundedThreadPool::LogicalThreadFactory : public ThreadFactory {
 public:
  explicit LogicalThreadFactory(UnboundedThreadPool* pool) : pool_(pool) {}

  std::unique_ptr<Thread> StartThread(const string& name,
                                      std::function<void()> fn) override {
    return pool_->RunOnPooledThread(std::move(fn));
  }

 private:
  UnboundedThreadPool* const pool_;  // Not owned.
};

// A logical implementation of the `tensorflow::Thread` interface that uses
// physical threads in an `UnboundedThreadPool` to perform the work.
//
// NOTE: This object represents a logical thread of control that may be mapped
// onto the same physical thread as other work items that are submitted to the
// same `UnboundedThreadPool`.
class UnboundedThreadPool::LogicalThreadWrapper : public Thread {
 public:
  explicit LogicalThreadWrapper(std::shared_ptr<Notification> join_notification)
      : join_notification_(std::move(join_notification)) {}

  ~LogicalThreadWrapper() override {
    // NOTE: The `Thread` destructor is expected to "join" the created thread,
    // but the physical thread may continue to execute after the work for this
    // thread is complete. We simulate this by waiting on a notification that
    // the `CachedThreadFunc` will notify when the thread's work function is
    // complete.
    join_notification_->WaitForNotification();
  }

 private:
  std::shared_ptr<Notification> join_notification_;
};

UnboundedThreadPool::~UnboundedThreadPool() {
  {
    mutex_lock l(work_queue_mu_);
    // Wake up all `CachedThreadFunc` threads and cause them to terminate before
    // joining them when `threads_` is cleared.
    cancelled_ = true;
    work_queue_cv_.notify_all();
    if (!work_queue_.empty()) {
      LOG(ERROR) << "UnboundedThreadPool named \"" << thread_name_ << "\" was "
                 << "deleted with pending work in its queue. This may indicate "
                 << "a potential use-after-free bug.";
    }
  }

  {
    mutex_lock l(thread_pool_mu_);
    // Clear the list of pooled threads, which will eventually terminate due to
    // the previous notification.
    //
    // NOTE: It is safe to do this while holding `pooled_threads_mu_`, because
    // no subsequent calls to `this->StartThread()` should be issued after the
    // destructor starts.
    thread_pool_.clear();
  }
}

std::shared_ptr<ThreadFactory> UnboundedThreadPool::get_thread_factory() {
  return std::make_shared<LogicalThreadFactory>(this);
}

size_t UnboundedThreadPool::size() {
  tf_shared_lock l(thread_pool_mu_);
  return thread_pool_.size();
}

std::unique_ptr<Thread> UnboundedThreadPool::RunOnPooledThread(
    std::function<void()> fn) {
  auto join_notification = std::make_shared<Notification>();
  bool all_threads_busy;
  {
    // Enqueue a work item for the new thread's function, and wake up a
    // cached thread to process it.
    mutex_lock l(work_queue_mu_);
    work_queue_.push_back({std::move(fn), join_notification});
    work_queue_cv_.notify_one();
    // NOTE: The queue may be non-empty, so we must account for queued work when
    // considering how many threads are free.
    all_threads_busy = work_queue_.size() > num_idle_threads_;
  }

  if (all_threads_busy) {
    // Spawn a new physical thread to process the given function.
    // NOTE: `PooledThreadFunc` will eventually increment `num_idle_threads_`
    // at the beginning of its work loop.
    Thread* new_thread = env_->StartThread(
        {}, thread_name_,
        std::bind(&UnboundedThreadPool::PooledThreadFunc, this));

    mutex_lock l(thread_pool_mu_);
    thread_pool_.emplace_back(new_thread);
  }

  return absl::make_unique<LogicalThreadWrapper>(std::move(join_notification));
}

void UnboundedThreadPool::PooledThreadFunc() {
  while (true) {
    WorkItem work_item;
    {
      mutex_lock l(work_queue_mu_);
      ++num_idle_threads_;
      while (!cancelled_ && work_queue_.empty()) {
        // Wait for a new work function to be submitted, or the cache to be
        // destroyed.
        work_queue_cv_.wait(l);
      }
      if (cancelled_) {
        return;
      }
      work_item = std::move(work_queue_.front());
      work_queue_.pop_front();
      --num_idle_threads_;
    }

    work_item.work_function();

    // Notify any thread that has "joined" the cached thread for this work item.
    work_item.done_notification->Notify();
  }
}

}  // namespace data
}  // namespace tensorflow

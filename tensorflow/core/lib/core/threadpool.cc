/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/core/threadpool.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/lib/core/nonblocking_threadpool.h"

#include <deque>

namespace tensorflow {
namespace thread {

struct ThreadPool::Impl {
  // Note to subclass implementers: Subclasses MUST finish all submitted
  // jobs in the destructor before returning from it.
  virtual ~Impl() {}

  virtual bool HasPendingClosures() const = 0;

  virtual void Schedule(std::function<void()> fn) = 0;
};

struct ThreadPoolDefaultImpl : public ThreadPool::Impl {
  ThreadPoolDefaultImpl(Env* env, const ThreadOptions& thread_options,
                        const string& name, int num_threads)
      : name_(name) {
    CHECK_GE(num_threads, 1);
    string name_prefix = "tf_" + name_;
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(env->StartThread(thread_options, name_prefix,
                                          [this]() { WorkerLoop(); }));
    }
  }

  ~ThreadPoolDefaultImpl() {
    {
      // Wait for all work to get done.
      mutex_lock l(mu_);

      // Inform every thread to exit.
      for (size_t i = 0; i < threads_.size(); ++i) {
        pending_.push_back({nullptr, 0});
      }

      // Wakeup all waiters.
      for (auto w : waiters_) {
        w->ready = true;
        w->cv.notify_one();
      }
    }

    // Wait for threads to finish.
    for (auto t : threads_) {
      delete t;
    }
  }

  struct Waiter {
    condition_variable cv;
    bool ready;
  };

  struct Waiter;
  struct Item {
    std::function<void()> fn;
    uint64 id;
  };

  bool HasPendingClosures() const override {
    mutex_lock l(mu_);
    return pending_.size() != 0;
  }

  void Schedule(std::function<void()> fn) override {
    CHECK(fn != nullptr);
    uint64 id = 0;
    if (port::Tracing::IsActive()) {
      id = port::Tracing::UniqueId();
      port::Tracing::RecordEvent(port::Tracing::EventCategory::kScheduleClosure,
                                 id);
    }

    mutex_lock l(mu_);
    pending_.push_back({fn, id});
    if (!waiters_.empty()) {
      Waiter* w = waiters_.back();
      waiters_.pop_back();
      w->ready = true;
      w->cv.notify_one();
    }
  }

  void WorkerLoop() {
    port::Tracing::RegisterCurrentThread(name_.c_str());
    mutex_lock l(mu_);
    Waiter w;
    while (true) {
      while (pending_.empty()) {
        // Wait for work to be assigned to me
        w.ready = false;
        waiters_.push_back(&w);
        while (!w.ready) {
          w.cv.wait(l);
        }
      }
      // Pick up pending work
      Item item = pending_.front();
      pending_.pop_front();
      if (item.fn == nullptr) {
        break;
      }
      mu_.unlock();
      if (item.id != 0) {
        port::Tracing::ScopedActivity region(
            port::Tracing::EventCategory::kRunClosure, item.id);
        item.fn();
      } else {
        item.fn();
      }
      mu_.lock();
    }
  }

  const string name_;
  mutable mutex mu_;
  std::vector<Thread*> threads_;  // All threads
  std::vector<Waiter*> waiters_;  // Stack of waiting threads.
  std::deque<Item> pending_;      // Queue of pending work
};

struct ThreadPoolNonBlockingImpl : public ThreadPool::Impl {
  ThreadPoolNonBlockingImpl(Env* env, const ThreadOptions& thread_options,
                            const string& name, int num_threads)
      : pool(env, name, thread_options, num_threads) {
    CHECK_GE(num_threads, 1);
  }

  bool HasPendingClosures() const override { return pool.jobs_running(); }

  void Schedule(std::function<void()> fn) override {
    pool.Schedule(std::move(fn));
  }

  NonBlockingThreadPool pool;
};

// Hook for unit tests, not exposed.  Not thread safe.
// Tells us which thread pool to use.  Possible values
// are:
//
//   nullptr = get it from TF_THREAD_POOL environment var, or if
//   not use DEFAULT_THREAD_POOL_IMPL;
//   non-null = use that implementation

const char* THREAD_POOL_IMPL_NAME = nullptr;
const char* DEFAULT_THREAD_POOL_IMPL = "mutex_based";

const char* get_thread_pool_impl_name() {
  if (THREAD_POOL_IMPL_NAME == nullptr) {
    static const char* env_name = getenv("TF_THREAD_POOL");
    return env_name ? env_name : DEFAULT_THREAD_POOL_IMPL;
  }
  return THREAD_POOL_IMPL_NAME;
}

ThreadPool::Impl* CreateDefaultThreadPoolImpl(
    Env* env, const string& name, const ThreadOptions& thread_options,
    int num_threads) {
  const char* impl_name = get_thread_pool_impl_name();
  if (strcmp(impl_name, "mutex_based") == 0) {
    return new ThreadPoolDefaultImpl(env, thread_options, name, num_threads);
  } else if (strcmp(impl_name, "lock_free") == 0) {
    static std::atomic_flag had_message = ATOMIC_FLAG_INIT;
    if (!had_message.test_and_set()) {
      LOG(INFO) << "Using experimental lock-free thread pool implementation";
    }

    return new ThreadPoolNonBlockingImpl(env, thread_options, name,
                                         num_threads);
  } else {
    LOG(FATAL) << "unknown value for TF_THREAD_POOL environment variable: '"
               << impl_name
               << "': accepted values are 'mutex_based' (safe, default), "
               << "'lock_free' (experimental, faster for >= 8 cores)";
  }
}

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads) {
  impl_.reset(
      CreateDefaultThreadPoolImpl(env, name, thread_options, num_threads));
}

ThreadPool::~ThreadPool() {}

bool ThreadPool::HasPendingClosures() const {
  return impl_->HasPendingClosures();
}

void ThreadPool::Schedule(std::function<void()> fn) {
  return impl_->Schedule(std::move(fn));
}

}  // namespace thread
}  // namespace tensorflow

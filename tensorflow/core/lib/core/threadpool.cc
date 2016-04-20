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

#ifdef TENSORFLOW_USE_EIGEN_THREADPOOL
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#else
#include <deque>
#include <thread>
#include <vector>
#endif

#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace thread {

#ifdef TENSORFLOW_USE_EIGEN_THREADPOOL

struct EigenEnvironment {
  typedef Thread EnvThread;
  struct Task {
    std::function<void()> f;
    uint64 trace_id;
  };

  Env* const env_;
  const ThreadOptions thread_options_;
  const string name_;

  EigenEnvironment(Env* env, const ThreadOptions& thread_options,
                   const string& name)
      : env_(env), thread_options_(thread_options), name_(name) {}

  EnvThread* CreateThread(std::function<void()> f) {
    return env_->StartThread(thread_options_, name_, [=]() {
      // Set the processor flag to flush denormals to zero
      port::ScopedFlushDenormal flush;
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    if (port::Tracing::IsActive()) {
      id = port::Tracing::UniqueId();
      port::Tracing::RecordEvent(port::Tracing::EventCategory::kScheduleClosure,
                                 id);
    }
    return Task{std::move(f), id};
  }

  void ExecuteTask(const Task& t) {
    if (t.trace_id != 0) {
      port::Tracing::ScopedActivity region(
          port::Tracing::EventCategory::kRunClosure, t.trace_id);
      t.f();
    } else {
      t.f();
    }
  }
};

struct ThreadPool::Impl : Eigen::ThreadPoolTempl<EigenEnvironment> {
  Impl(Env* env, const ThreadOptions& thread_options, const string& name,
       int num_threads)
      : Eigen::ThreadPoolTempl<EigenEnvironment>(
            num_threads, EigenEnvironment(env, thread_options, name)) {}
};

#else

struct ThreadPool::Impl {
  Impl(Env* env, const ThreadOptions& thread_options, const string& name,
       int num_threads);
  ~Impl();
  void Schedule(std::function<void()> fn);

 private:
  struct Waiter {
    condition_variable cv;
    bool ready;
  };

  struct Task {
    std::function<void()> fn;
    uint64 id;
  };

  void WorkerLoop();

  const string name_;
  mutex mu_;
  std::vector<Thread*> threads_;  // All threads
  std::vector<Waiter*> waiters_;  // Stack of waiting threads.
  std::deque<Task> pending_;      // Queue of pending work
};

ThreadPool::Impl::Impl(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads)
    : name_(name) {
  for (int i = 0; i < num_threads; i++) {
    threads_.push_back(
        env->StartThread(thread_options, name, [this]() { WorkerLoop(); }));
  }
}

ThreadPool::Impl::~Impl() {
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

void ThreadPool::Impl::Schedule(std::function<void()> fn) {
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

void ThreadPool::Impl::WorkerLoop() {
  // Set the processor flag to flush denormals to zero
  port::ScopedFlushDenormal flush;

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
    Task t = pending_.front();
    pending_.pop_front();
    if (t.fn == nullptr) {
      break;
    }
    mu_.unlock();
    if (t.id != 0) {
      port::Tracing::ScopedActivity region(
          port::Tracing::EventCategory::kRunClosure, t.id);
      t.fn();
    } else {
      t.fn();
    }
    mu_.lock();
  }
}
#endif

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads) {
  CHECK_GE(num_threads, 1);
  impl_.reset(
      new ThreadPool::Impl(env, thread_options, "tf_" + name, num_threads));
}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  impl_->Schedule(std::move(fn));
}

}  // namespace thread
}  // namespace tensorflow

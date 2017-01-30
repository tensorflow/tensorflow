/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"


namespace tensorflow {
namespace thread {

struct EigenEnvironment {
  typedef Thread EnvThread;
  struct TaskImpl {
    std::function<void()> f;
    Context context;
    uint64 trace_id;
  };
  struct Task {
    std::unique_ptr<TaskImpl> f;
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
      // Set the C++ rounding mode to ROUND TO NEAREST
      port::ScopedSetRound round;
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
    return Task{
        std::unique_ptr<TaskImpl>(new TaskImpl{
            std::move(f), Context(ContextKind::kThread), id,
        }),
    };
  }

  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    if (t.f->trace_id != 0) {
      port::Tracing::ScopedActivity region(
          port::Tracing::EventCategory::kRunClosure, t.f->trace_id);
      t.f->f();
    } else {
      t.f->f();
    }
  }
};

struct ThreadPool::Impl : Eigen::ThreadPoolTempl<EigenEnvironment> {
  Impl(Env* env, const ThreadOptions& thread_options, const string& name,
       int num_threads)
      : Eigen::ThreadPoolTempl<EigenEnvironment>(
            num_threads, EigenEnvironment(env, thread_options, name)) {}

  void ParallelFor(int64 total, int64 cost_per_unit,
                   std::function<void(int64, int64)> fn) {
    CHECK_GE(total, 0);
    CHECK_EQ(total, (int64)(Eigen::Index)total);
    Eigen::ThreadPoolDevice device(this, this->NumThreads());
    device.parallelFor(
        total, Eigen::TensorOpCost(0, 0, cost_per_unit),
        [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
  }
};

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

void ThreadPool::ParallelFor(int64 total, int64 cost_per_unit,
                             std::function<void(int64, int64)> fn) {
  impl_->ParallelFor(total, cost_per_unit, std::move(fn));
}

void ThreadPool::ParallelForWithWorkerId(
    int64 total, int64 cost_per_unit,
    const std::function<void(int64, int64, int)>& fn) {
  impl_->ParallelFor(total, cost_per_unit,
                     [this, &fn](int64 start, int64 limit) {
                       // ParallelFor may use the current thread to do some
                       // work synchronously. When calling CurrentThreadId()
                       // from outside of the thread pool, we get -1, so we can
                       // shift every id up by 1.
                       int id = CurrentThreadId() + 1;
                       fn(start, limit, id);
                     });
}

int ThreadPool::NumThreads() const { return impl_->NumThreads(); }

int ThreadPool::CurrentThreadId() const { return impl_->CurrentThreadId(); }

}  // namespace thread
}  // namespace tensorflow

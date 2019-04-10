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
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"
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
      // Set the processor flag to flush denormals to zero.
      port::ScopedFlushDenormal flush;
      // Set the processor rounding mode to ROUND TO NEAREST.
      port::ScopedSetRound round(FE_TONEAREST);
      if (thread_options_.numa_node != port::kNUMANoAffinity) {
        port::NUMASetThreadNodeAffinity(thread_options_.numa_node);
      }
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    if (tracing::EventCollector::IsEnabled()) {
      id = tracing::GetUniqueArg();
      tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);
    }
    return Task{
        std::unique_ptr<TaskImpl>(new TaskImpl{
            std::move(f),
            Context(ContextKind::kThread),
            id,
        }),
    };
  }

  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure,
                                 t.f->trace_id);
    t.f->f();
  }
};

struct ThreadPool::Impl : Eigen::ThreadPoolTempl<EigenEnvironment> {
  Impl(Env* env, const ThreadOptions& thread_options, const string& name,
       int num_threads, bool low_latency_hint, Eigen::Allocator* allocator)
      : Eigen::ThreadPoolTempl<EigenEnvironment>(
            num_threads, low_latency_hint,
            EigenEnvironment(env, thread_options, name)),
        allocator_(allocator) {}

  void ParallelFor(int64 total, int64 cost_per_unit,
                   std::function<void(int64, int64)> fn) {
    CHECK_GE(total, 0);
    CHECK_EQ(total, (int64)(Eigen::Index)total);
    Eigen::ThreadPoolDevice device(this, this->NumThreads(), allocator_);
    device.parallelFor(
        total, Eigen::TensorOpCost(0, 0, cost_per_unit),
        [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
  }

  Eigen::Allocator* allocator_;
};

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads,
                       bool low_latency_hint, Eigen::Allocator* allocator) {
  CHECK_GE(num_threads, 1);
  impl_.reset(new ThreadPool::Impl(env, thread_options, "tf_" + name,
                                   num_threads, low_latency_hint, allocator));
}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  impl_->Schedule(std::move(fn));
}

int ThreadPool::NumShardsUsedByTransformRangeConcurrently(
    const int64 block_size, const int64 total) {
  if (block_size <= 0 || total <= 1 || total <= block_size ||
      NumThreads() == 1) {
    return 1;
  }
  return (total + block_size - 1) / block_size;
}

// This functionality is similar to parallelFor, except that reasoning about
// the number of shards used is significantly easier.
void ThreadPool::TransformRangeConcurrently(
    const int64 block_size, const int64 total,
    const std::function<void(int64, int64)>& fn) {
  const int num_shards_used =
      NumShardsUsedByTransformRangeConcurrently(block_size, total);
  if (num_shards_used == 1) {
    fn(0, total);
    return;
  }

  // Adapted from Eigen's parallelFor implementation.
  BlockingCounter counter(num_shards_used);
  std::function<void(int64, int64)> handle_range =
      [=, &handle_range, &counter, &fn](int64 first, int64 last) {
        while (last - first > block_size) {
          // Find something near the midpoint which is a multiple of block size.
          const int64 mid = first + ((last - first) / 2 + block_size - 1) /
                                        block_size * block_size;
          Schedule([=, &handle_range]() { handle_range(mid, last); });
          last = mid;
        }
        // Single block or less, execute directly.
        fn(first, last);
        counter.DecrementCount();  // The shard is done.
      };
  if (num_shards_used <= NumThreads()) {
    // Avoid a thread hop by running the root of the tree and one block on the
    // main thread.
    handle_range(0, total);
  } else {
    // Execute the root in the thread pool to avoid running work on more than
    // numThreads() threads.
    Schedule([=, &handle_range]() { handle_range(0, total); });
  }
  counter.Wait();
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

void ThreadPool::ScheduleWithHint(std::function<void()> fn, int start,
                                  int limit) {
  impl_->ScheduleWithHint(std::move(fn), start, limit);
}

void ThreadPool::SetStealPartitions(
    const std::vector<std::pair<unsigned, unsigned>>& partitions) {
  impl_->SetStealPartitions(partitions);
}

Eigen::ThreadPoolInterface* ThreadPool::AsEigenThreadPool() {
  return impl_.get();
}
}  // namespace thread
}  // namespace tensorflow

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

#include "tensorflow/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS

#include "absl/types/optional.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/tsl/platform/blocking_counter.h"
#include "tensorflow/tsl/platform/denormal.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/numa.h"
#include "tensorflow/tsl/platform/setround.h"

namespace tsl {
// TODO(aminim): remove after tensorflow/core/platform/context.h migration.
using tensorflow::Context;
using tensorflow::ContextKind;
using tensorflow::WithContext;
// TODO(aminim): remove after tensorflow/core/platform/setround.h migration.
namespace port {
using namespace tensorflow::port;  // NOLINT
}  // namespace port
// TODO(aminim): remove after tensorflow/core/platform/tracing.h migration.
namespace tracing {
using namespace tensorflow::tracing;  // NOLINT
}  // namespace tracing

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
      tsl::port::ScopedSetRound round(FE_TONEAREST);
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

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads,
                       bool low_latency_hint, Eigen::Allocator* allocator) {
  CHECK_GE(num_threads, 1);
  eigen_threadpool_.reset(new Eigen::ThreadPoolTempl<EigenEnvironment>(
      num_threads, low_latency_hint,
      EigenEnvironment(env, thread_options, "tf_" + name)));
  underlying_threadpool_ = eigen_threadpool_.get();
  threadpool_device_.reset(new Eigen::ThreadPoolDevice(underlying_threadpool_,
                                                       num_threads, allocator));
}

ThreadPool::ThreadPool(thread::ThreadPoolInterface* user_threadpool) {
  underlying_threadpool_ = user_threadpool;
  threadpool_device_.reset(new Eigen::ThreadPoolDevice(
      underlying_threadpool_, underlying_threadpool_->NumThreads(), nullptr));
}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  underlying_threadpool_->Schedule(std::move(fn));
}

int ThreadPool::NumShardsUsedByFixedBlockSizeScheduling(
    const int64_t total, const int64_t block_size) {
  if (block_size <= 0 || total <= 1 || total <= block_size ||
      NumThreads() == 1) {
    return 1;
  }
  return (total + block_size - 1) / block_size;
}

int ThreadPool::NumShardsUsedByTransformRangeConcurrently(
    const int64_t block_size, const int64_t total) {
  return NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
}

void ThreadPool::ParallelFor(int64_t total,
                             const SchedulingParams& scheduling_params,
                             const std::function<void(int64_t, int64_t)>& fn) {
  switch (scheduling_params.strategy()) {
    case SchedulingStrategy::kAdaptive: {
      if (scheduling_params.cost_per_unit().has_value()) {
        ParallelFor(total, *scheduling_params.cost_per_unit(), fn);
      }
      break;
    }
    case SchedulingStrategy::kFixedBlockSize: {
      if (scheduling_params.block_size().has_value()) {
        ParallelForFixedBlockSizeScheduling(
            total, *scheduling_params.block_size(), fn);
      }
      break;
    }
  }
}

void ThreadPool::TransformRangeConcurrently(
    const int64_t block_size, const int64_t total,
    const std::function<void(int64_t, int64_t)>& fn) {
  ParallelFor(total,
              SchedulingParams(SchedulingStrategy::kFixedBlockSize,
                               absl::nullopt /* cost_per_unit */, block_size),
              fn);
}

// This functionality is similar to parallelFor, except that reasoning about
// the number of shards used is significantly easier.
void ThreadPool::ParallelForFixedBlockSizeScheduling(
    const int64_t total, const int64_t block_size,
    const std::function<void(int64_t, int64_t)>& fn) {
  const int num_shards_used =
      NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
  if (num_shards_used == 1) {
    fn(0, total);
    return;
  }

  // Adapted from Eigen's parallelFor implementation.
  BlockingCounter counter(num_shards_used);
  std::function<void(int64_t, int64_t)> handle_range =
      [=, &handle_range, &counter, &fn](int64_t first, int64_t last) {
        while (last - first > block_size) {
          // Find something near the midpoint which is a multiple of block size.
          const int64_t mid = first + ((last - first) / 2 + block_size - 1) /
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

void ThreadPool::ParallelFor(int64_t total, int64_t cost_per_unit,
                             const std::function<void(int64_t, int64_t)>& fn) {
  CHECK_GE(total, 0);
  CHECK_EQ(total, (int64_t)(Eigen::Index)total);
  threadpool_device_->parallelFor(
      total, Eigen::TensorOpCost(0, 0, cost_per_unit),
      [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
}

void ThreadPool::ParallelForWithWorkerId(
    int64_t total, int64_t cost_per_unit,
    const std::function<void(int64_t, int64_t, int)>& fn) {
  CHECK_GE(total, 0);
  CHECK_EQ(total, (int64_t)(Eigen::Index)total);

  threadpool_device_->parallelFor(total,
                                  Eigen::TensorOpCost(0, 0, cost_per_unit),
                                  [this, &fn](int64_t start, int64_t limit) {
                                    // ParallelFor may use the current thread to
                                    // do some work synchronously. When calling
                                    // CurrentThreadId() from outside of the
                                    // thread pool, we get -1, so we can shift
                                    // every id up by 1.
                                    int id = CurrentThreadId() + 1;
                                    fn(start, limit, id);
                                  });
}

void ThreadPool::ParallelForWithWorkerId(
    int64_t total, const SchedulingParams& scheduling_params,
    const std::function<void(int64_t, int64_t, int)>& fn) {
  ParallelFor(total, scheduling_params,
              [this, &fn](int64_t start, int64_t limit) {
                // We may use the current thread to do some work synchronously.
                // When calling CurrentThreadId() from outside of the thread
                // pool, we get -1, so we can shift every id up by 1.
                int id = CurrentThreadId() + 1;
                fn(start, limit, id);
              });
}

int ThreadPool::NumThreads() const {
  return underlying_threadpool_->NumThreads();
}

int ThreadPool::CurrentThreadId() const {
  return underlying_threadpool_->CurrentThreadId();
}

void ThreadPool::ScheduleWithHint(std::function<void()> fn, int start,
                                  int limit) {
  underlying_threadpool_->ScheduleWithHint(std::move(fn), start, limit);
}

void ThreadPool::SetStealPartitions(
    const std::vector<std::pair<unsigned, unsigned>>& partitions) {
  // ThreadPool::SetStealPartitions is only called in the constructor of
  // RunHandlerPool::Impl, which currently instantiates ThreadPool using a
  // constructor that does not take user_threadpool. Thus we assume
  // eigen_threadpool_ is not null here.
  DCHECK(eigen_threadpool_ != nullptr);
  eigen_threadpool_->SetStealPartitions(partitions);
}

Eigen::ThreadPoolInterface* ThreadPool::AsEigenThreadPool() const {
  DCHECK(underlying_threadpool_ != nullptr);
  return underlying_threadpool_;
}
}  // namespace thread
}  // namespace tsl

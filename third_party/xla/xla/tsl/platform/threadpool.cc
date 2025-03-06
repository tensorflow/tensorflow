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

#include "xla/tsl/platform/threadpool.h"

#include <cfenv>  // NOLINT
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/optimization.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool_interface.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/context.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/setround.h"
#include "tsl/platform/tracing.h"

#ifdef DNNL_AARCH64_USE_ACL
#include "tsl/platform/cpu_info.h"
#endif  // DNNL_AARCH64_USE_ACL

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

#ifdef TENSORFLOW_THREADSCALING_EXPERIMENTAL
ABSL_FLAG(float, tensorflow_num_threads_scale_factor, 1.0,
          "Allows to scale all Tensorflow ThreadPools. Total number of threads "
          "in a given ThreadPool equals to num_threads * "
          "tensorflow_num_threads_scale_factor. Default scale factor of 1 is a "
          "no-op.");
#endif  // TENSORFLOW_THREADSCALING_EXPERIMENTAL

namespace tsl::thread {

struct EigenEnvironment {
  using EnvThread = Thread;

  struct TaskImpl {
    std::function<void()> fn;
    Context context;
    uint64 trace_id;
  };

  struct Task {
    Task() = default;

    Task(std::function<void()> fn, Context context, uint64 trace_id)
        : f(TaskImpl{std::move(fn), std::move(context), trace_id}) {}

    Task(Task&&) = default;
    Task& operator=(Task&&) = default;

    std::optional<TaskImpl> f;
  };

  Env* const env;
  const ThreadOptions thread_options;
  const std::string name;

  EigenEnvironment(Env* env, const ThreadOptions& thread_options,
                   std::string name)
      : env(env), thread_options(thread_options), name(std::move(name)) {}

  EnvThread* CreateThread(std::function<void()> f) {
    return env->StartThread(thread_options, name, [this, f = std::move(f)]() {
      // Set the processor flag to flush denormals to zero.
      port::ScopedFlushDenormal flush;
      // Set the processor rounding mode to ROUND TO NEAREST.
      tsl::port::ScopedSetRound round(FE_TONEAREST);
      if (thread_options.numa_node != port::kNUMANoAffinity) {
        port::NUMASetThreadNodeAffinity(thread_options.numa_node);
      }
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    if (ABSL_PREDICT_FALSE(tracing::EventCollector::IsEnabled())) {
      id = tracing::GetUniqueArg();
      tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);
    }
    return Task(std::move(f), Context(ContextKind::kThread), id);
  }

  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure,
                                 t.f->trace_id);
    t.f->fn();
  }
};

ThreadPool::ThreadPool(Env* env, const std::string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const std::string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const std::string& name, int num_threads,
                       bool low_latency_hint, Eigen::Allocator* allocator) {
  CHECK_GE(num_threads, 1);

#ifdef DNNL_AARCH64_USE_ACL
  // To avoid cost of swapping in and out threads from running processes
  // we do not use all available cores to parallelise TF operations.
  if (num_threads == tsl::port::NumTotalCPUs() && num_threads >= 16) {
    num_threads = num_threads - 1;
  }
#endif  // DNNL_AARCH64_USE_ACL

#ifdef TENSORFLOW_THREADSCALING_EXPERIMENTAL
  CHECK_GT(absl::GetFlag(FLAGS_tensorflow_num_threads_scale_factor), 0);
  num_threads *= absl::GetFlag(FLAGS_tensorflow_num_threads_scale_factor);
  if (num_threads < 1) num_threads = 1;
#endif  // TENSORFLOW_THREADSCALING_EXPERIMENTAL

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
                               /*cost_per_unit=*/std::nullopt, block_size),
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

Eigen::ThreadPoolInterface* ThreadPool::AsEigenThreadPool() const {
  DCHECK(underlying_threadpool_ != nullptr);
  return underlying_threadpool_;
}

}  // namespace tsl::thread

/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_WORK_QUEUE_H_
#define XLA_BACKENDS_CPU_RUNTIME_WORK_QUEUE_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

// A work queue that partitions `num_tasks` tasks into `num_partitions`
// partitions processed by parallel workers.
class WorkQueue {
  // Align all atomic counters to a cache line boundary to avoid false
  // sharing between multiple worker threads.
  static constexpr size_t kAtomicAlignment =
#if defined(__cpp_lib_hardware_interference_size)
      std::hardware_destructive_interference_size;
#else
      64;
#endif

 public:
  WorkQueue(size_t num_tasks, size_t num_partitions);

  // Returns the next task in the given partition. Returns std::nullopt
  // if the partition is complete.
  std::optional<size_t> Pop(size_t partition_index);

  size_t num_partitions() const { return partitions_.size(); }

  bool empty() const { return empty_.load(std::memory_order_relaxed); }

 private:
  friend class Worker;

  struct Partition {
    void Initialize(size_t begin, size_t end);

    // Tracks index of the next task in the assigned partition.
    alignas(kAtomicAlignment) std::atomic<size_t> index;
    size_t begin;
    size_t end;
  };

  absl::FixedArray<Partition, 32> partitions_;
  alignas(kAtomicAlignment) std::atomic<size_t> empty_;
};

// Worker processes tasks from the work queue starting from the assigned
// work partition. Once the assigned partition is complete it tries to pop
// the task from the next partition. Once the work queue is empty (the worker
// wraps around to the initial partition) it returns and empty task.
class Worker {
 public:
  Worker(size_t worker_index, WorkQueue* queue);

  std::optional<size_t> Pop();

  // Schedule `count_down.count()` workers into the Eigen thread pool that
  // process `num_tasks` parallel tasks and count down for each completed
  // worker.
  template <typename ParallelTask>
  static void Parallelize(const Eigen::ThreadPoolDevice* device,
                          tsl::CountDownAsyncValueRef<tsl::Chain> count_down,
                          size_t num_tasks, ParallelTask&& parallel_task);

  // Schedule `num_workers` workers into the Eigen thread pool that process
  // `num_tasks` parallel tasks and return an async value that becomes
  // available when all workers are completed.
  template <typename ParallelTask>
  static tsl::AsyncValueRef<tsl::Chain> Parallelize(
      const Eigen::ThreadPoolDevice* device, size_t num_workers,
      size_t num_tasks, ParallelTask&& parallel_task);

 private:
  template <typename ParallelTask>
  struct ParallelizeContext;

  template <typename ParallelTask>
  static absl::Status ExecuteInline(size_t num_tasks,
                                    ParallelTask&& parallel_task);

  template <typename ParallelTask>
  static void ParallelizeWithContext(ParallelizeContext<ParallelTask>* ctx,
                                     uint16_t start_index, uint16_t end_index);

  size_t worker_index_;
  size_t partition_index_;
  WorkQueue* queue_;
};

inline void WorkQueue::Partition::Initialize(size_t begin, size_t end) {
  index.store(begin, std::memory_order_relaxed);
  this->begin = begin;
  this->end = end;
}

inline WorkQueue::WorkQueue(size_t num_tasks, size_t num_partitions)
    : partitions_(num_partitions), empty_(num_tasks == 0) {
  size_t partition_size = tsl::MathUtil::CeilOfRatio(num_tasks, num_partitions);
  for (size_t i = 0, begin = 0, end = partition_size; i < num_partitions;
       ++i, begin = end, end = std::min(num_tasks, end + partition_size)) {
    partitions_[i].Initialize(begin, end);
  }
}

inline std::optional<size_t> WorkQueue::Pop(size_t partition_index) {
  DCHECK(partition_index < partitions_.size()) << "Invalid partition index";
  Partition& partition = partitions_.data()[partition_index];

  // Check if partition is already empty.
  if (size_t index = partition.index.load(std::memory_order_relaxed);
      index >= partition.end) {
    return std::nullopt;
  }

  // Try to acquire the next task in the partition.
  size_t index = partition.index.fetch_add(1, std::memory_order_relaxed);
  return index >= partition.end ? std::nullopt : std::make_optional(index);
}

inline Worker::Worker(size_t worker_index, WorkQueue* queue)
    : worker_index_(worker_index),
      partition_index_(worker_index),
      queue_(queue) {}

inline std::optional<size_t> Worker::Pop() {
  std::optional<size_t> task = queue_->Pop(partition_index_);
  if (task) return task;

  // If work queue is empty, we are not going to find any more tasks.
  if (queue_->empty()) return std::nullopt;

  while (!task.has_value()) {
    // Wrap around to the first partition.
    if (ABSL_PREDICT_FALSE(++partition_index_ >= queue_->num_partitions())) {
      partition_index_ = 0;
    }

    // We checked all partitions and got back to the partition we started from.
    if (ABSL_PREDICT_FALSE(partition_index_ == worker_index_)) {
      queue_->empty_.store(true, std::memory_order_relaxed);
      break;
    }

    task = queue_->Pop(partition_index_);
  }

  return task;
}

template <typename ParallelTask>
struct Worker::ParallelizeContext {
  ParallelizeContext(const Eigen::ThreadPoolDevice* device,
                     tsl::CountDownAsyncValueRef<tsl::Chain> count_down,
                     size_t num_tasks, ParallelTask&& parallel_task);

  const Eigen::ThreadPoolDevice* device;
  tsl::CountDownAsyncValueRef<tsl::Chain> count_down;

  WorkQueue work_queue;
  ParallelTask parallel_task;
};

template <typename ParallelTask>
Worker::ParallelizeContext<ParallelTask>::ParallelizeContext(
    const Eigen::ThreadPoolDevice* device,
    tsl::CountDownAsyncValueRef<tsl::Chain> count_down, size_t num_tasks,
    ParallelTask&& parallel_task)
    : device(device),
      count_down(std::move(count_down)),
      work_queue(num_tasks, /*num_partitions=*/this->count_down.count()),
      parallel_task(std::forward<ParallelTask>(parallel_task)) {}

template <typename ParallelTask>
void Worker::ParallelizeWithContext(ParallelizeContext<ParallelTask>* ctx,
                                    uint16_t start_index, uint16_t end_index) {
  DCHECK_LT(start_index, end_index) << "Invalid worker index range";

  using R = std::invoke_result_t<ParallelTask, size_t>;
  static_assert(std::is_same_v<R, absl::Status> || std::is_void_v<R>,
                "Unsupported parallel task return type");

  auto count_down = [&](size_t count, absl::Status status) {
    // If count down is completed, delete the context.
    if (ctx->count_down.CountDown(count, std::move(status))) delete ctx;
  };

  // Recursively split assigned workers into two halves and schedule the
  // right half into the thread pool.
  while (end_index - start_index > 1) {
    // If work queue is empty, we don't need to keep enqueuing more workers and
    // can simply count down for the remaining workers.
    if (ABSL_PREDICT_FALSE(ctx->work_queue.empty())) {
      count_down(end_index - start_index, absl::OkStatus());
      return;
    }

    uint16_t mid_index = (start_index + end_index) / 2;
    ctx->device->enqueueNoNotification([ctx, mid_index, end_index] {
      ParallelizeWithContext(ctx, mid_index, end_index);
    });
    end_index = mid_index;
  }

  // Execute the `start_index` worker in the caller thread.
  Worker worker(start_index, &ctx->work_queue);
  while (std::optional<size_t> task = worker.Pop()) {
    if constexpr (std::is_same_v<R, absl::Status>) {
      absl::Status status = ctx->parallel_task(*task);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        count_down(1, std::move(status));
        return;
      }
    } else {
      ctx->parallel_task(*task);
    }
  }

  // Count down for the one executed worker.
  count_down(1, absl::OkStatus());
}

template <typename ParallelTask>
ABSL_ATTRIBUTE_ALWAYS_INLINE absl::Status Worker::ExecuteInline(
    size_t num_tasks, ParallelTask&& parallel_task) {
  using R = std::invoke_result_t<ParallelTask, size_t>;
  static_assert(std::is_same_v<R, absl::Status> || std::is_void_v<R>,
                "Unsupported parallel task return type");

  for (size_t i = 0; i < num_tasks; ++i) {
    if constexpr (std::is_same_v<R, absl::Status>) {
      absl::Status status = parallel_task(i);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        return status;
      }
    } else {
      parallel_task(i);
    }
  }

  return absl::OkStatus();
}

template <typename ParallelTask>
ABSL_ATTRIBUTE_ALWAYS_INLINE void Worker::Parallelize(
    const Eigen::ThreadPoolDevice* device,
    tsl::CountDownAsyncValueRef<tsl::Chain> count_down, size_t num_tasks,
    ParallelTask&& parallel_task) {
  size_t num_workers = count_down.count();
  DCHECK_LE(num_workers, num_tasks);

  // Short-circuit single-threaded execution.
  if (ABSL_PREDICT_FALSE(num_workers == 1)) {
    count_down.CountDown(
        ExecuteInline(num_tasks, std::forward<ParallelTask>(parallel_task)));
    return;
  }

  DCHECK_LE(num_workers, std::numeric_limits<uint16_t>::max());
  if (ABSL_PREDICT_FALSE(num_workers > std::numeric_limits<uint16_t>::max())) {
    count_down.CountDown(num_workers - std::numeric_limits<uint16_t>::max());
  }

  auto ctx = std::make_unique<ParallelizeContext<ParallelTask>>(
      device, std::move(count_down), num_tasks,
      std::forward<ParallelTask>(parallel_task));

  ParallelizeWithContext(ctx.release(), 0, num_workers);
}

template <typename ParallelTask>
ABSL_ATTRIBUTE_ALWAYS_INLINE tsl::AsyncValueRef<tsl::Chain> Worker::Parallelize(
    const Eigen::ThreadPoolDevice* device, size_t num_workers, size_t num_tasks,
    ParallelTask&& parallel_task) {
  // Short-circuit single-threaded execution.
  if (ABSL_PREDICT_FALSE(num_workers == 1)) {
    if (absl::Status status =
            ExecuteInline(num_tasks, std::forward<ParallelTask>(parallel_task));
        ABSL_PREDICT_FALSE(!status.ok())) {
      return status;
    }
    return tsl::MakeAvailableAsyncValueRef<tsl::Chain>();
  }

  DCHECK_LE(num_workers, std::numeric_limits<uint16_t>::max());
  if (ABSL_PREDICT_FALSE(num_workers > std::numeric_limits<uint16_t>::max())) {
    num_workers = std::numeric_limits<uint16_t>::max();
  }

  tsl::CountDownAsyncValueRef<tsl::Chain> count_down(num_workers);
  auto execute_event = count_down.AsRef();

  Parallelize(device, std::move(count_down), num_tasks,
              std::forward<ParallelTask>(parallel_task));

  return execute_event;
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_WORK_QUEUE_H_

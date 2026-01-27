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
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "Eigen/ThreadPool"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/platform/logging.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

// A work queue that partitions `num_work_items` work items into
// `num_partitions` partitions processed by parallel workers.
class WorkQueue {
 public:
  WorkQueue(size_t num_work_items, size_t num_partitions);

  // Returns the next work item in the given partition. Returns std::nullopt
  // if the partition is complete.
  std::optional<size_t> Pop(size_t partition_index);

  // Return the partition [begin, end) work items range.
  std::pair<size_t, size_t> partition_range(size_t partition_index) const;

  size_t num_partitions() const { return partitions_.size(); }

  bool IsEmpty() const { return empty_.load(std::memory_order_relaxed); }

 private:
  friend class Worker;

  struct Partition {
    void Initialize(size_t begin, size_t end);

    // Tracks index of the next work item in the assigned partition.
    ABSL_CACHELINE_ALIGNED std::atomic<size_t> index;
    size_t begin;
    size_t end;
  };

  // Sets an empty work queue flag to stop worker threads from looping through
  // all partitions looking for work.
  void SetEmpty() { empty_.store(true, std::memory_order_relaxed); }

  // Notify that one of the workers switched to the work stealing mode.
  void NotifyWorkStealingWorker();

  // Decrements the number of work stealing workers by at most `max_workers` and
  // returns the number of decremented work stealing workers.
  size_t DecrementWorkStealingWorkers(size_t max_workers);

  absl::FixedArray<Partition, 32> partitions_;
  ABSL_CACHELINE_ALIGNED std::atomic<bool> empty_;
  ABSL_CACHELINE_ALIGNED std::atomic<size_t> num_work_stealing_workers_;
};

// Worker processes work items from the work queue starting from the assigned
// work partition. Once the assigned partition is complete it tries to pop
// the work item from the next partition. Once the work queue is empty (the
// worker wraps around to the initial partition) it returns and empty work item.
class Worker {
 public:
  Worker(size_t worker_index, WorkQueue* queue);

  // Pops a work item from the work queue. If `notify_work_stealing` is true,
  // the worker will notify the work queue when it switches to the work
  // stealing mode. Worker parallelization has an optimization to avoid
  // scheduling more workers if there are workers in the work stealing mode.
  std::optional<size_t> Pop(bool notify_work_stealing = true);

  // Schedule `num_workers` workers into the Eigen thread pool that process
  // `num_work_items` parallel work items and return an async value that becomes
  // available when all workers are completed.
  template <typename ParallelWork>
  static tsl::AsyncValueRef<tsl::Chain> Parallelize(
      Eigen::ThreadPoolInterface* thread_pool, size_t num_workers,
      size_t num_work_items, ParallelWork&& parallel_work);

 private:
  template <typename ParallelWork>
  struct ParallelizeContext;

  template <typename ParallelWork>
  static absl::Status ExecuteInline(size_t num_work_items,
                                    ParallelWork&& parallel_work);

  template <typename ParallelWork>
  static void Parallelize(std::shared_ptr<ParallelizeContext<ParallelWork>> ctx,
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

inline WorkQueue::WorkQueue(size_t num_work_items, size_t num_partitions)
    : partitions_(num_partitions),
      empty_(num_work_items == 0),
      num_work_stealing_workers_(0) {
  size_t partition_size = num_work_items / num_partitions;
  size_t rem_work_items = num_work_items % num_partitions;
  for (size_t i = 0, begin = 0, end = 0; i < num_partitions; ++i, begin = end) {
    end = begin + partition_size + ((i < rem_work_items) ? 1 : 0);
    partitions_[i].Initialize(begin, end);
  }
}

inline std::optional<size_t> WorkQueue::Pop(size_t partition_index) {
  DCHECK(partition_index < partitions_.size()) << "Invalid partition index";
  Partition& partition = partitions_.data()[partition_index];

  // Check if partition is already empty.
  if (size_t index = partition.index.load(std::memory_order_relaxed);
      ABSL_PREDICT_FALSE(index >= partition.end)) {
    return std::nullopt;
  }

  // Try to acquire the next work item in the partition.
  size_t index = partition.index.fetch_add(1, std::memory_order_relaxed);
  return ABSL_PREDICT_FALSE(index >= partition.end) ? std::nullopt
                                                    : std::make_optional(index);
}

inline std::pair<size_t, size_t> WorkQueue::partition_range(
    size_t partition_index) const {
  DCHECK(partition_index < partitions_.size()) << "Invalid partition index";
  return {partitions_[partition_index].begin, partitions_[partition_index].end};
}

inline void WorkQueue::NotifyWorkStealingWorker() {
  num_work_stealing_workers_.fetch_add(1, std::memory_order_relaxed);
}

inline size_t WorkQueue::DecrementWorkStealingWorkers(size_t max_workers) {
  size_t n = num_work_stealing_workers_.load(std::memory_order_relaxed);

  size_t decrement = std::min(n, max_workers);
  while (decrement && !num_work_stealing_workers_.compare_exchange_weak(
                          n, n - decrement, std::memory_order_relaxed,
                          std::memory_order_relaxed)) {
    decrement = std::min(n, max_workers);
  }

  return decrement;
}

inline Worker::Worker(size_t worker_index, WorkQueue* queue)
    : worker_index_(worker_index),
      partition_index_(worker_index),
      queue_(queue) {}

inline std::optional<size_t> Worker::Pop(bool notify_work_stealing) {
  std::optional<size_t> work_item = queue_->Pop(partition_index_);
  if (ABSL_PREDICT_TRUE(work_item)) {
    return work_item;
  }

  // If we didn't find a work item in the initially assigned partition, notify
  // the work queue that we are switching to work stealing mode.
  if (ABSL_PREDICT_FALSE(notify_work_stealing &&
                         partition_index_ == worker_index_)) {
    queue_->NotifyWorkStealingWorker();
  }

  while (!work_item.has_value() && !queue_->IsEmpty()) {
    // Wrap around to the first partition.
    if (ABSL_PREDICT_FALSE(++partition_index_ >= queue_->num_partitions())) {
      partition_index_ = 0;
    }

    // We checked all partitions and got back to the partition we started from.
    if (ABSL_PREDICT_FALSE(partition_index_ == worker_index_)) {
      queue_->SetEmpty();
      break;
    }

    work_item = queue_->Pop(partition_index_);
  }

  return work_item;
}

template <typename ParallelWork>
struct Worker::ParallelizeContext {
  ParallelizeContext(Eigen::ThreadPoolInterface* thread_pool,
                     tsl::CountDownAsyncValueRef<tsl::Chain> count_down,
                     size_t num_work_items, ParallelWork&& parallel_work);

  Eigen::ThreadPoolInterface* thread_pool;
  tsl::CountDownAsyncValueRef<tsl::Chain> count_down;

  WorkQueue work_queue;
  ParallelWork parallel_work;
};

template <typename ParallelWork>
Worker::ParallelizeContext<ParallelWork>::ParallelizeContext(
    Eigen::ThreadPoolInterface* thread_pool,
    tsl::CountDownAsyncValueRef<tsl::Chain> count_down, size_t num_work_items,
    ParallelWork&& parallel_work)
    : thread_pool(thread_pool),
      count_down(std::move(count_down)),
      work_queue(num_work_items, /*num_partitions=*/this->count_down.count()),
      parallel_work(std::forward<ParallelWork>(parallel_work)) {}

template <typename ParallelWork>
void Worker::Parallelize(std::shared_ptr<ParallelizeContext<ParallelWork>> ctx,
                         uint16_t start_index, uint16_t end_index) {
  DCHECK_LT(start_index, end_index) << "Invalid worker index range";

  using R = std::invoke_result_t<ParallelWork, size_t>;
  static_assert(std::is_same_v<R, absl::Status> || std::is_void_v<R>,
                "Unsupported parallel work return type");

  // Recursively split assigned workers into two halves and schedule the
  // right half into the thread pool.
  while (end_index - start_index > 1) {
    // If work queue is empty, we don't need to keep scheduling more workers.
    if (ABSL_PREDICT_FALSE(ctx->work_queue.IsEmpty())) {
      return;
    }

    // If we have workers in the work stealing mode, we can skip scheduling
    // more workers as existing workers will process remaining partitions. By
    // doing this optimization we avoid unnecessary thread pool overheads.
    size_t skip_workers =
        ctx->work_queue.DecrementWorkStealingWorkers(end_index - start_index);
    if (ABSL_PREDICT_FALSE(skip_workers > 0)) {
      DCHECK_LE(skip_workers, end_index - start_index);
      end_index -= skip_workers;

      // Return if there is no more work to do.
      if (start_index == end_index) {
        return;
      }

      // Execute the last remaining worker in the caller thread.
      if (end_index - start_index == 1) {
        break;
      }
    }

    DCHECK_GE(end_index - start_index, 1);
    uint16_t mid_index = (start_index + end_index) / 2;
    ctx->thread_pool->Schedule([ctx, mid_index, end_index] {
      Parallelize(ctx, mid_index, end_index);
    });
    end_index = mid_index;
  }

  // Execute the `start_index` worker in the caller thread.
  Worker worker(start_index, &ctx->work_queue);
  size_t num_processed_work_items = 0;

  // Keep track of the first error status encountered by any of the workers.
  absl::Status status;

  while (std::optional<size_t> work_item = worker.Pop()) {
    if constexpr (std::is_same_v<R, absl::Status>) {
      if (ABSL_PREDICT_TRUE(status.ok())) {
        status.Update(ctx->parallel_work(*work_item));
      }
    } else {
      ctx->parallel_work(*work_item);
    }
    ++num_processed_work_items;
  }

  ctx->count_down.CountDown(num_processed_work_items, std::move(status));
}

template <typename ParallelWork>
ABSL_ATTRIBUTE_ALWAYS_INLINE absl::Status Worker::ExecuteInline(
    size_t num_work_items, ParallelWork&& parallel_work) {
  using R = std::invoke_result_t<ParallelWork, size_t>;
  static_assert(std::is_same_v<R, absl::Status> || std::is_void_v<R>,
                "Unsupported parallel work return type");

  for (size_t i = 0; i < num_work_items; ++i) {
    if constexpr (std::is_same_v<R, absl::Status>) {
      absl::Status status = parallel_work(i);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        return status;
      }
    } else {
      parallel_work(i);
    }
  }

  return absl::OkStatus();
}

template <typename ParallelWork>
ABSL_ATTRIBUTE_ALWAYS_INLINE tsl::AsyncValueRef<tsl::Chain> Worker::Parallelize(
    Eigen::ThreadPoolInterface* thread_pool, size_t num_workers,
    size_t num_work_items, ParallelWork&& parallel_work) {
  // Short-circuit single-threaded execution.
  if (ABSL_PREDICT_FALSE(num_workers == 1)) {
    if (absl::Status status = ExecuteInline(
            num_work_items, std::forward<ParallelWork>(parallel_work));
        ABSL_PREDICT_FALSE(!status.ok())) {
      return status;
    }
    return tsl::MakeAvailableAsyncValueRef<tsl::Chain>();
  }

  DCHECK_LE(num_workers, std::numeric_limits<uint16_t>::max());
  if (ABSL_PREDICT_FALSE(num_workers > std::numeric_limits<uint16_t>::max())) {
    num_workers = std::numeric_limits<uint16_t>::max();
  }
  // Ensure we don't launch more workers than work items. Extra workers would be
  // idle or cause out-of-bounds partition access.
  num_workers = std::min(num_work_items, num_workers);

  tsl::CountDownAsyncValueRef<tsl::Chain> count_down(num_work_items);
  auto execute_event = count_down.AsRef();

  auto ctx = std::make_shared<ParallelizeContext<ParallelWork>>(
      thread_pool, std::move(count_down), num_work_items,
      std::forward<ParallelWork>(parallel_work));

  Parallelize(std::move(ctx), 0, num_workers);

  return execute_event;
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_WORK_QUEUE_H_

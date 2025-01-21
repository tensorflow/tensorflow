/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/xnnpack/parallel_loop_runner.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/time/time.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

using Task = std::function<void(size_t task_index)>;

// Returns non-reference-counted async value ref in constructed state.
//
// Returned async value is a per-process singleton stored in a storage with a
// static duration, and can be safely compared using pointer equality.
static tsl::AsyncValueRef<tsl::Chain> OkDoneEventSingleton() {
  static tsl::AsyncValueOwningRef<tsl::Chain>* singleton = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<tsl::Chain>();
    return new tsl::AsyncValueOwningRef<tsl::Chain>(
        tsl::MakeAvailableAsyncValueRef<tsl::Chain>(*storage));
  }();
  return singleton->AsRef();
}

ParallelLoopRunner::ParallelLoopRunner(
    const Eigen::ThreadPoolDevice* device,
    std::optional<absl::Duration> worker_timeslice)
    : done_event_(OkDoneEventSingleton()),
      device_(device),
      worker_timeslice_(worker_timeslice) {}

tsl::AsyncValueRef<tsl::Chain> ParallelLoopRunner::ResetDoneEvent() {
  auto done_event = std::move(done_event_);
  done_event_ = OkDoneEventSingleton();
  return done_event;
}

size_t ParallelLoopRunner::num_threads() const {
  return device_.load()->numThreadsInPool();
}

tsl::AsyncValueRef<tsl::Chain> ParallelLoopRunner::TakeDoneEvent(
    ParallelLoopRunner&& runner) {
  return std::move(runner.done_event_);
}

void ParallelLoopRunner::WorkQueue::Partition::Initialize(size_t begin,
                                                          size_t end) {
  index.store(begin, std::memory_order_relaxed);
  this->begin = begin;
  this->end = end;
}

ParallelLoopRunner::WorkQueue::WorkQueue(size_t num_tasks,
                                         size_t num_partitions)
    : partitions_(num_partitions), empty_(num_tasks == 0) {
  size_t partition_size = tsl::MathUtil::CeilOfRatio(num_tasks, num_partitions);
  for (size_t i = 0, begin = 0, end = partition_size; i < num_partitions;
       ++i, begin = end, end = std::min(num_tasks, end + partition_size)) {
    partitions_[i].Initialize(begin, end);
  }
}

std::optional<size_t> ParallelLoopRunner::WorkQueue::Pop(
    size_t partition_index) {
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

ParallelLoopRunner::Worker::Worker(size_t worker_index, WorkQueue* queue)
    : worker_index_(worker_index),
      partition_index_(worker_index),
      queue_(queue) {}

std::optional<size_t> ParallelLoopRunner::Worker::Pop() {
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

template <typename ParallelizeContext>
static void Parallelize(ParallelizeContext* ctx, uint16_t start_index,
                        uint16_t end_index) {
  CHECK_LT(start_index, end_index) << "Invalid worker index range";

  auto count_down = [&](size_t count) {
    // If count down is completed, delete the context.
    if (ctx->count_down.CountDown(count)) delete ctx;
  };

  // Recursively split assigned workers into two halves and schedule the
  // right half into the thread pool.
  while (end_index - start_index > 1) {
    // If work queue is empty, we don't need to keep enqueuing more workers and
    // can simply count down for the remaining workers.
    if (ABSL_PREDICT_FALSE(ctx->work_queue.empty())) {
      count_down(end_index - start_index);
      return;
    }

    uint16_t mid_partition = (start_index + end_index) / 2;
    ctx->device->enqueueNoNotification([ctx, mid_partition, end_index] {
      Parallelize(ctx, mid_partition, end_index);
    });
    end_index = mid_partition;
  }

  // Execute the `start_index` worker in the caller thread.
  ParallelLoopRunner::Worker worker(start_index, &ctx->work_queue);
  while (std::optional<size_t> task = worker.Pop()) {
    ctx->parallel_task(*task);
  }

  // Count down for the one executed worker.
  count_down(1);
}

template <typename ParallelTask>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::Parallelize(
    tsl::CountDownAsyncValueRef<tsl::Chain> count_down, size_t num_workers,
    size_t num_tasks, ParallelTask&& parallel_task) {
  DCHECK_EQ(count_down.count(), num_workers)
      << "Number of workers must match the count down counter";

  // Short-circuit single-threaded execution.
  if (ABSL_PREDICT_FALSE(num_workers == 1)) {
    for (size_t i = 0; i < num_tasks; ++i) {
      parallel_task(i);
    }
    count_down.CountDown();
    return;
  }

  struct ParallelizeContext {
    ParallelizeContext(const Eigen::ThreadPoolDevice* device,
                       tsl::CountDownAsyncValueRef<tsl::Chain> count_down,
                       size_t num_workers, size_t num_tasks,
                       ParallelTask&& parallel_task)
        : device(device),
          num_workers(num_workers),
          work_queue(num_tasks, /*num_partitions=*/num_workers),
          count_down(std::move(count_down)),
          parallel_task(std::forward<ParallelTask>(parallel_task)) {}

    const Eigen::ThreadPoolDevice* device;

    size_t num_workers;
    WorkQueue work_queue;

    tsl::CountDownAsyncValueRef<tsl::Chain> count_down;
    ParallelTask parallel_task;
  };

  auto ctx = std::make_unique<ParallelizeContext>(
      device_, std::move(count_down), num_workers, num_tasks,
      std::forward<ParallelTask>(parallel_task));

  DCHECK_LE(num_workers, std::numeric_limits<uint16_t>::max());
  xla::cpu::Parallelize(ctx.release(), 0, num_workers);
}

template <typename Task>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::ScheduleOne(Task&& task) {
  auto event = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
  done_event_.AndThen([event, task = std::forward<Task>(task)] {
    task();
    event.SetStateConcrete();
  });
  done_event_ = std::move(event);
}

// Compute the number of workers that should be used for parallel operation, by
// executing the first task, measuring the compute time and estimating how many
// workers are needed, so that each worker will handle `worker_timeslice` amount
// of compute.
template <typename ParallelTask>
ABSL_ATTRIBUTE_ALWAYS_INLINE size_t
ComputeOptimalNumWorkers(absl::Duration worker_timeslice, size_t num_threads,
                         size_t num_tasks, ParallelTask& parallel_task) {
  // Run first task in the caller thread, to estimate the number of parallel
  // workers that should be used for parallel operation.
  uint64_t start_ns = tsl::Env::Default()->NowNanos();
  parallel_task(0);
  uint64_t end_ns = tsl::Env::Default()->NowNanos();

  // We assume that all tasks take roughly the same amount of compute and we
  // can estimate the total workload duration by multiplying the number of
  // remaining tasks by the duration of a single task.
  size_t workload_ns = (num_tasks - 1) * (end_ns - start_ns);
  size_t timeslice_ns = absl::ToInt64Nanoseconds(worker_timeslice);

  // Get the number of workers, so that each worker will take roughly
  // `worker_timeslice` amount of compute. Don't create more workers than
  // the number of threads in the thread pool or the number of tasks.
  size_t num_workers =
      std::min(std::min(num_tasks - 1, num_threads),
               tsl::MathUtil::CeilOfRatio(workload_ns, timeslice_ns));
  return std::min(num_workers, size_t{std::numeric_limits<uint16_t>::max()});
}

template <typename ParallelTask>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::ScheduleAll(
    size_t num_tasks, ParallelTask&& parallel_task) {
  DCHECK_GT(num_tasks, 1) << "Expected at least two task";

  // If done event is already available and we have a worker timeslice, we can
  // compute the optimal number of workers for the parallel operation and
  // potentially avoid allocating count down counter altogether.
  if (ABSL_PREDICT_TRUE(done_event_.IsConcrete() && worker_timeslice_)) {
    size_t optimal_num_workers = ComputeOptimalNumWorkers(
        *worker_timeslice_, num_threads(), num_tasks, parallel_task);

    // Execute remaining tasks in the caller thread if we have a single worker.
    if (ABSL_PREDICT_TRUE(optimal_num_workers == 1)) {
      for (size_t i = 1; i < num_tasks; ++i) {
        parallel_task(i);
      }
      return;
    }

    tsl::CountDownAsyncValueRef<tsl::Chain> count_down(optimal_num_workers);
    done_event_ = count_down.AsRef();

    // Parallelize the remaining tasks (skip the first task that was executed
    // when we were computing the number of workers).
    Parallelize(std::move(count_down), optimal_num_workers, num_tasks - 1,
                [parallel_task = std::forward<ParallelTask>(parallel_task)](
                    size_t task_index) { parallel_task(task_index + 1); });
    return;
  }

  // If `done_event_` is not available, we start with at most `num_threads()`
  // workers as we can't run more parallel workers than the number of threads in
  // the thread pool. Later we might adjust the number of workers when it's safe
  // to execute the first task to measure the execution time.
  size_t num_workers = std::min(std::min(num_tasks, num_threads()),
                                size_t{std::numeric_limits<uint16_t>::max()});

  tsl::CountDownAsyncValueRef<tsl::Chain> count_down(num_workers);
  auto count_down_done = count_down.AsRef();

  auto schedule_all =
      [this, num_workers, num_tasks, count_down = std::move(count_down),
       parallel_task = std::forward<ParallelTask>(parallel_task)]() mutable {
        // If we don't have a worker timeslice, we can parallelize the task
        // immediately using pre-computed number of workers.
        if (ABSL_PREDICT_FALSE(!worker_timeslice_)) {
          Parallelize(std::move(count_down), num_workers, num_tasks,
                      std::move(parallel_task));
          return;
        }

        // Compute the optimal number of workers by executing the first task.
        size_t optimal_num_workers = ComputeOptimalNumWorkers(
            *worker_timeslice_, num_threads(), num_tasks, parallel_task);
        DCHECK_LE(optimal_num_workers, num_workers);

        // Count down for the workers that we don't need.
        count_down.CountDown(num_workers - optimal_num_workers);

        // Parallelize the remaining tasks (skip the first task that was
        // executed when we were computing the number of workers).
        Parallelize(std::move(count_down), optimal_num_workers, num_tasks - 1,
                    [parallel_task = std::move(parallel_task)](
                        size_t task_index) { parallel_task(task_index + 1); });
      };

  done_event_.AndThen(std::move(schedule_all));
  done_event_ = std::move(count_down_done);
}

namespace {

// Multidimensional index types for the parallel loop runner tasks. We launch
// tasks using one-dimensional `task_index` and convert it into a
// multidimensional index type depending on the loop type.

struct Task1DTile1DIndex {
  size_t offset;
  size_t extent;
};

struct Task2DTile1DIndex {
  size_t i;
  size_t offset_j;
  size_t extent_j;
};

struct Task3DTile2DIndex {
  size_t i;
  size_t offset_j;
  size_t offset_k;
  size_t extent_j;
  size_t extent_k;
};

}  // namespace

static Task1DTile1DIndex Delinearize(size_t task_index, size_t range,
                                     size_t tile) {
  size_t offset = task_index * tile;
  size_t extent = std::min(range - offset, tile);
  return {offset, extent};
}

static size_t NumTasks(size_t range_i, size_t range_j, size_t tile_j) {
  size_t num_tile_j_tasks = tsl::MathUtil::CeilOfRatio(range_j, tile_j);
  size_t num_tasks = range_i * num_tile_j_tasks;
  DCHECK_GT(num_tasks, 0) << "Expected at least one tile task";
  return num_tasks;
}

static Task2DTile1DIndex Delinearize(size_t task_index, size_t range_i,
                                     size_t range_j, size_t tile_j) {
  size_t num_tile_j_tasks = tsl::MathUtil::CeilOfRatio(range_j, tile_j);
  DCHECK_GT(num_tile_j_tasks, 0) << "Expected at least one tile j task";

  // Compute task indices along the `i` and `j` dimensions.
  size_t task_i = task_index / num_tile_j_tasks;
  size_t task_j = task_index % num_tile_j_tasks;

  // Convert task index into the offset and extent along the `j` dimension.
  size_t offset_j = task_j * tile_j;
  size_t extent_j = std::min(range_j - offset_j, tile_j);

  return {task_i, offset_j, extent_j};
}

static size_t NumTasks(size_t range_i, size_t range_j, size_t range_k,
                       size_t tile_j, size_t tile_k) {
  size_t num_tile_j_tasks = tsl::MathUtil::CeilOfRatio(range_j, tile_j);
  size_t num_tile_k_tasks = tsl::MathUtil::CeilOfRatio(range_k, tile_k);
  size_t num_tasks = range_i * num_tile_j_tasks * num_tile_k_tasks;
  DCHECK_GT(num_tasks, 0) << "Expected at least one tile task";
  return num_tasks;
}

static Task3DTile2DIndex Delinearize(size_t task_index, size_t range_i,
                                     size_t range_j, size_t range_k,
                                     size_t tile_j, size_t tile_k) {
  size_t num_tile_j_tasks = tsl::MathUtil::CeilOfRatio(range_j, tile_j);
  size_t num_tile_k_tasks = tsl::MathUtil::CeilOfRatio(range_k, tile_k);
  size_t num_tile_tasks = num_tile_j_tasks * num_tile_k_tasks;

  DCHECK_GT(num_tile_j_tasks, 0) << "Expected at least one tile j task";
  DCHECK_GT(num_tile_k_tasks, 0) << "Expected at least one tile k task";

  // Compute task indices along the `i`, `j` and `k` dimensions.
  size_t task_i = task_index / num_tile_tasks;
  task_index %= num_tile_tasks;

  size_t task_j = task_index / num_tile_k_tasks;
  task_index %= num_tile_k_tasks;

  size_t task_k = task_index;

  // Convert task indices into the offset and extent along the `j` and `k`
  // dimensions.
  size_t offset_j = task_j * tile_j;
  size_t offset_k = task_k * tile_k;
  size_t extent_j = std::min(range_j - offset_j, tile_j);
  size_t extent_k = std::min(range_k - offset_k, tile_k);

  return {task_i, offset_j, offset_k, extent_j, extent_k};
}

// In the `Parallelize` implementations below:
//
// (1) If done event is already available, execute the task immediately in the
//     caller thread. In this case we don't need to overwrite the done event,
//     because the existing one will correctly represent the state of the
//     parallel loop runner (all scheduled loops are ready).
//
// (2) If done event is not available, we have to overwrite it with a new one
//     that will be set to concrete state after the task is executed.

void ParallelLoopRunner::Parallelize(size_t range, Task1D task) {
  DCHECK(done_event_) << "Parallel loop runner is in moved-from state";
  DCHECK_GT(range, 0) << "Expected at least one task";

  // Fast path for the degenerate parallel loop with single task.
  if (ABSL_PREDICT_TRUE(range == 1)) {
    // Execute task in the caller thread if done event is already available.
    if (ABSL_PREDICT_TRUE(done_event_.IsConcrete())) {
      task(0);
      return;
    }

    // Schedule task when done event becomes available.
    ScheduleOne([task = std::move(task)] { task(0); });
    return;
  }

  ScheduleAll(range, std::move(task));
}

void ParallelLoopRunner::Parallelize(size_t range, size_t tile,
                                     Task1DTile1D task) {
  DCHECK(done_event_) << "Parallel loop runner is in moved-from state";

  size_t num_tasks = tsl::MathUtil::CeilOfRatio(range, tile);
  DCHECK_GT(num_tasks, 0) << "Expected at least one task";

  // Fast path for the degenerate parallel loop with single task.
  if (ABSL_PREDICT_TRUE(num_tasks == 1)) {
    // Execute task in the caller thread if done event is already available.
    if (ABSL_PREDICT_TRUE(done_event_.IsConcrete())) {
      task(0, range);
      return;
    }

    // Schedule task when done event becomes available.
    ScheduleOne([range, task = std::move(task)] { task(0, range); });
    return;
  }

  auto parallel_task = [range, tile,
                        task = std::move(task)](size_t task_index) {
    auto x = Delinearize(task_index, range, tile);
    task(x.offset, x.extent);
  };

  ScheduleAll(num_tasks, std::move(parallel_task));
}

void ParallelLoopRunner::Parallelize(size_t range_i, size_t range_j,
                                     size_t tile_j, Task2DTile1D task) {
  DCHECK(done_event_) << "Parallel loop runner is in moved-from state";
  size_t num_tasks = NumTasks(range_i, range_j, tile_j);

  // Fast path for the degenerate parallel loop with single task.
  if (ABSL_PREDICT_TRUE(num_tasks == 1)) {
    // Execute task in the caller thread if done event is already available.
    if (ABSL_PREDICT_TRUE(done_event_.IsConcrete())) {
      task(0, 0, range_j);
      return;
    }

    // Schedule task when done event becomes available.
    ScheduleOne([range_j, task = std::move(task)] { task(0, 0, range_j); });
    return;
  }

  auto parallel_task = [range_i, range_j, tile_j,
                        task = std::move(task)](size_t task_index) {
    auto x = Delinearize(task_index, range_i, range_j, tile_j);
    task(x.i, x.offset_j, x.extent_j);
  };

  ScheduleAll(num_tasks, std::move(parallel_task));
}

void ParallelLoopRunner::Parallelize(size_t range_i, size_t range_j,
                                     size_t range_k, size_t tile_j,
                                     size_t tile_k, Task3DTile2D task) {
  DCHECK(done_event_) << "Parallel loop runner is in moved-from state";
  size_t num_tasks = NumTasks(range_i, range_j, range_k, tile_j, tile_k);

  // Fast path for the degenerate parallel loop with single task.
  if (ABSL_PREDICT_TRUE(num_tasks == 1)) {
    // Execute task in the caller thread if done event is already available.
    if (ABSL_PREDICT_TRUE(done_event_.IsConcrete())) {
      task(0, 0, 0, range_j, range_k);
      return;
    }

    // Schedule task when done event becomes available.
    ScheduleOne([range_j, range_k, task = std::move(task)] {
      task(0, 0, 0, range_j, range_k);
    });
    return;
  }

  auto parallel_task = [range_i, range_j, range_k, tile_j, tile_k,
                        task = std::move(task)](size_t task_index) {
    auto x = Delinearize(task_index, range_i, range_j, range_k, tile_j, tile_k);
    task(x.i, x.offset_j, x.offset_k, x.extent_j, x.extent_k);
  };

  ScheduleAll(num_tasks, std::move(parallel_task));
}

}  // namespace xla::cpu

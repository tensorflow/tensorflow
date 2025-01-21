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
#include "xla/backends/cpu/runtime/xnnpack/work_queue.h"
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
    Worker::Parallelize(
        device_, std::move(count_down), num_tasks - 1,
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
          Worker::Parallelize(device_, std::move(count_down), num_tasks,
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
        Worker::Parallelize(
            device_, std::move(count_down), num_tasks - 1,
            [parallel_task = std::move(parallel_task)](size_t task_index) {
              parallel_task(task_index + 1);
            });
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

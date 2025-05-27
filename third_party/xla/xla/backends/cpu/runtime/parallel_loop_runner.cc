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

#include "xla/backends/cpu/runtime/parallel_loop_runner.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "xla/backends/cpu/runtime/work_queue.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/platform/logging.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

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

ParallelLoopRunner::ParallelLoopRunner(const Eigen::ThreadPoolDevice* device)
    : done_event_(OkDoneEventSingleton()), device_(device) {}

tsl::AsyncValueRef<tsl::Chain> ParallelLoopRunner::ResetDoneEvent() {
  auto done_event = std::move(done_event_);
  done_event_ = OkDoneEventSingleton();
  return done_event;
}

size_t ParallelLoopRunner::num_threads() const {
  return device_.load()->numThreadsInPool();
}

bool ParallelLoopRunner::is_in_runner() const {
  return device_.load()->currentThreadId() > -1;
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

template <typename ParallelTask>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::ScheduleAll(
    size_t num_tasks, ParallelTask&& parallel_task) {
  DCHECK_GT(num_tasks, 1) << "Expected at least two task";

  // Use at most `num_threads()` workers as we can't run more parallel workers
  // than the number of threads in the thread pool.
  size_t num_workers = std::min(std::min(num_tasks, num_threads()),
                                size_t{std::numeric_limits<uint16_t>::max()});

  tsl::CountDownAsyncValueRef<tsl::Chain> count_down(num_workers);
  auto count_down_done = count_down.AsRef();

  auto parallelize = [this, num_tasks, count_down = std::move(count_down),
                      parallel_task =
                          std::forward<ParallelTask>(parallel_task)] {
    Worker::Parallelize(device_, std::move(count_down), num_tasks,
                        std::move(parallel_task));
  };

  done_event_.AndThen(std::move(parallelize));
  done_event_ = std::move(count_down_done);
}

// A collection of helper macros to define parallel task structs for ND loops
// with different types of dimensions.

#define DEFINE_PARALLEL_TASK_1D(TASK, DIM0)                                 \
  struct ParallelLoopRunner::Parallel##TASK {                               \
    ABSL_ATTRIBUTE_ALWAYS_INLINE void operator()(size_t task_index) const { \
      std::apply(task, Delinearize(task_index, this->i));                   \
    }                                                                       \
    DIM0 i;                                                                 \
    TASK task;                                                              \
  }

#define DEFINE_PARALLEL_TASK_2D(TASK, DIM0, DIM1)                           \
  struct ParallelLoopRunner::Parallel##TASK {                               \
    ABSL_ATTRIBUTE_ALWAYS_INLINE void operator()(size_t task_index) const { \
      std::apply(task, Delinearize(task_index, this->i, this->j));          \
    }                                                                       \
    DIM0 i;                                                                 \
    DIM1 j;                                                                 \
    TASK task;                                                              \
  }

#define DEFINE_PARALLEL_TASK_3D(TASK, DIM0, DIM1, DIM2)                     \
  struct ParallelLoopRunner::Parallel##TASK {                               \
    ABSL_ATTRIBUTE_ALWAYS_INLINE void operator()(size_t task_index) const { \
      std::apply(task, Delinearize(task_index, this->i, this->j, this->k)); \
    }                                                                       \
    DIM0 i;                                                                 \
    DIM1 j;                                                                 \
    DIM2 k;                                                                 \
    TASK task;                                                              \
  }

#define DEFINE_PARALLEL_TASK_4D(TASK, DIM0, DIM1, DIM2, DIM3)                  \
  struct ParallelLoopRunner::Parallel##TASK {                                  \
    ABSL_ATTRIBUTE_ALWAYS_INLINE void operator()(size_t task_index) const {    \
      std::apply(task,                                                         \
                 Delinearize(task_index, this->i, this->j, this->k, this->l)); \
    }                                                                          \
    DIM0 i;                                                                    \
    DIM1 j;                                                                    \
    DIM2 k;                                                                    \
    DIM3 l;                                                                    \
    TASK task;                                                                 \
  }

#define DEFINE_PARALLEL_TASK_5D(TASK, DIM0, DIM1, DIM2, DIM3, DIM4)         \
  struct ParallelLoopRunner::Parallel##TASK {                               \
    ABSL_ATTRIBUTE_ALWAYS_INLINE void operator()(size_t task_index) const { \
      std::apply(task, Delinearize(task_index, this->i, this->j, this->k,   \
                                   this->l, this->m));                      \
    }                                                                       \
    DIM0 i;                                                                 \
    DIM1 j;                                                                 \
    DIM2 k;                                                                 \
    DIM3 l;                                                                 \
    DIM4 m;                                                                 \
    TASK task;                                                              \
  }

DEFINE_PARALLEL_TASK_1D(Task1D, RangeDim);
DEFINE_PARALLEL_TASK_1D(Task1DTile1D, TileDim);

DEFINE_PARALLEL_TASK_2D(Task2D, RangeDim, RangeDim);
DEFINE_PARALLEL_TASK_2D(Task2DTile1D, RangeDim, TileDim);
DEFINE_PARALLEL_TASK_2D(Task2DTile2D, TileDim, TileDim);

DEFINE_PARALLEL_TASK_3D(Task3D, RangeDim, RangeDim, RangeDim);
DEFINE_PARALLEL_TASK_3D(Task3DTile1D, RangeDim, RangeDim, TileDim);
DEFINE_PARALLEL_TASK_3D(Task3DTile2D, RangeDim, TileDim, TileDim);

DEFINE_PARALLEL_TASK_4D(Task4DTile2D, RangeDim, RangeDim, TileDim, TileDim);

DEFINE_PARALLEL_TASK_5D(Task5D, RangeDim, RangeDim, RangeDim, RangeDim,
                        RangeDim);
DEFINE_PARALLEL_TASK_5D(Task5DTile2D, RangeDim, RangeDim, RangeDim, TileDim,
                        TileDim);

#undef DEFINE_PARALLEL_TASK_1D
#undef DEFINE_PARALLEL_TASK_2D
#undef DEFINE_PARALLEL_TASK_3D
#undef DEFINE_PARALLEL_TASK_4D
#undef DEFINE_PARALLEL_TASK_5D

// Parallelize `task` over dimensions `dims` using `ParallelTask`.
//
// (1) If done event is already available, execute the task immediately in the
//     caller thread. In this case we don't need to overwrite the done event,
//     because the existing one will correctly represent the state of the
//     parallel loop runner (all scheduled loops are ready).
//
// (2) If done event is not available, we have to overwrite it with a new one
//     that will be set to concrete state after the task is executed.
//
// We wrap all tasks into structs conforming to the `ParallelTest` API, so that
// in profiles we can see human-readable names of the tasks instead of lambdas.
template <typename ParallelTask, typename... Dims, typename Task>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::Parallelize(Dims... dims,
                                                                  Task&& task) {
  DCHECK(done_event_) << "Parallel loop runner is in moved-from state";

  size_t num_tasks = NumTasks(dims...);
  DCHECK_GT(num_tasks, 0) << "Expected at least one task";

  // Fast path for the degenerate parallel loop with a single task.
  if (ABSL_PREDICT_TRUE(num_tasks == 1)) {
    // Converts the dimension into the first task index.
    auto to_first_task_index = [](auto dim) {
      if constexpr (std::is_same_v<decltype(dim), RangeDim>) {
        return RangeIndex{0};
      } else {
        return TileIndex{0, dim.range};
      }
    };

    // Execute task in the caller thread if done event is already available.
    if (ABSL_PREDICT_TRUE(done_event_.IsConcrete())) {
      task(to_first_task_index(dims)...);
      return;
    }

    // Schedule task when done event becomes available.
    ScheduleOne([task = std::forward<Task>(task),
                 idxs = std::make_tuple(to_first_task_index(dims)...)] {
      std::apply([&task](auto... idxs) { task(idxs...); }, idxs);
    });
    return;
  }

  ScheduleAll(num_tasks, ParallelTask{dims..., std::forward<Task>(task)});
}

// Parallelize `task` over dynamic dimensions `dims` using `ParallelTask`.
template <typename ParallelTask, typename... Dims, typename Task>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::ParallelizeDynamic(
    Dims... dims, Task&& task) {
  // We target 4 tasks per thread to enable load balancing and keep task
  // scheduling overheads low.
  static constexpr size_t kTasksPerThread = 4;
  std::apply(
      [&](auto... dynamic_dims) {
        Parallelize<ParallelTask, Dims...>(dynamic_dims...,
                                           std::forward<Task>(task));
      },
      DynamicDimensions(kTasksPerThread * num_threads(), dims...));
}

void ParallelLoopRunner::Parallelize(RangeDim i, Task1D task) {
  Parallelize<ParallelTask1D, RangeDim>(i, std::move(task));
}

void ParallelLoopRunner::Parallelize(RangeDim i, RangeDim j, Task2D task) {
  Parallelize<ParallelTask2D, RangeDim, RangeDim>(i, j, std::move(task));
}

void ParallelLoopRunner::Parallelize(TileDim i, Task1DTile1D task) {
  Parallelize<ParallelTask1DTile1D, TileDim>(i, std::move(task));
}

void ParallelLoopRunner::ParallelizeDynamic(TileDim i, Task1DTile1D task) {
  ParallelizeDynamic<ParallelTask1DTile1D, TileDim>(i, std::move(task));
}

void ParallelLoopRunner::Parallelize(RangeDim i, TileDim j, Task2DTile1D task) {
  Parallelize<ParallelTask2DTile1D, RangeDim, TileDim>(i, j, std::move(task));
}

void ParallelLoopRunner::ParallelizeDynamic(RangeDim i, TileDim j,
                                            Task2DTile1D task) {
  ParallelizeDynamic<ParallelTask2DTile1D, RangeDim, TileDim>(i, j,
                                                              std::move(task));
}

void ParallelLoopRunner::Parallelize(TileDim i, TileDim j, Task2DTile2D task) {
  Parallelize<ParallelTask2DTile2D, TileDim, TileDim>(i, j, std::move(task));
}

void ParallelLoopRunner::ParallelizeDynamic(TileDim i, TileDim j,
                                            Task2DTile2D task) {
  ParallelizeDynamic<ParallelTask2DTile2D, TileDim, TileDim>(i, j,
                                                             std::move(task));
}

void ParallelLoopRunner::Parallelize(RangeDim i, RangeDim j, RangeDim k,
                                     Task3D task) {
  Parallelize<ParallelTask3D, RangeDim, RangeDim, RangeDim>(i, j, k,
                                                            std::move(task));
}

void ParallelLoopRunner::Parallelize(RangeDim i, RangeDim j, TileDim k,
                                     Task3DTile1D task) {
  Parallelize<ParallelTask3DTile1D, RangeDim, RangeDim, TileDim>(
      i, j, k, std::move(task));
}

void ParallelLoopRunner::Parallelize(RangeDim i, TileDim j, TileDim k,
                                     Task3DTile2D task) {
  Parallelize<ParallelTask3DTile2D, RangeDim, TileDim, TileDim>(
      i, j, k, std::move(task));
}

void ParallelLoopRunner::ParallelizeDynamic(RangeDim i, TileDim j, TileDim k,
                                            Task3DTile2D task) {
  ParallelizeDynamic<ParallelTask3DTile2D, RangeDim, TileDim, TileDim>(
      i, j, k, std::move(task));
}

void ParallelLoopRunner::Parallelize(RangeDim i, RangeDim j, TileDim k,
                                     TileDim l, Task4DTile2D task) {
  Parallelize<ParallelTask4DTile2D, RangeDim, RangeDim, TileDim, TileDim>(
      i, j, k, l, std::move(task));
}

void ParallelLoopRunner::ParallelizeDynamic(RangeDim i, RangeDim j, TileDim k,
                                            TileDim l, Task4DTile2D task) {
  ParallelizeDynamic<ParallelTask4DTile2D, RangeDim, RangeDim, TileDim,
                     TileDim>(i, j, k, l, std::move(task));
}

void ParallelLoopRunner::Parallelize(RangeDim i, RangeDim j, RangeDim k,
                                     RangeDim l, RangeDim m, Task5D task) {
  Parallelize<ParallelTask5D, RangeDim, RangeDim, RangeDim, RangeDim, RangeDim>(
      i, j, k, l, m, std::move(task));
}

void ParallelLoopRunner::Parallelize(RangeDim i, RangeDim j, RangeDim k,
                                     TileDim l, TileDim m, Task5DTile2D task) {
  Parallelize<ParallelTask5DTile2D, RangeDim, RangeDim, RangeDim, TileDim,
              TileDim>(i, j, k, l, m, std::move(task));
}

}  // namespace xla::cpu

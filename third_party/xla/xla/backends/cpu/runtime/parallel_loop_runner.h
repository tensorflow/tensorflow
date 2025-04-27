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

#ifndef XLA_BACKENDS_CPU_RUNTIME_PARALLEL_LOOP_RUNNER_H_
#define XLA_BACKENDS_CPU_RUNTIME_PARALLEL_LOOP_RUNNER_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/lib/math/math_util.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace xla::cpu {

// Parallel loop runner uses underlying Eigen ThreadPoolDevice to execute
// parallel loops providing implicit synchronization: the next parallel loop
// starts execution only after all tasks from the previous loop are completed.
//
// Scheduled parallel loops execute asynchronously without blocking the caller
// thread. It is the user's responsibility to ensure that all values captured by
// the task are valid until the task is completed.
//
// Parallel loop runner is an implementation of the `pthreadpool` API adaptor
// for XLA:CPU runtime.
//
// ParallelLoopRunner uses "persistent workers" to execute parallel loops.
// Workers get scheduled into the underlying thread pool and when they start
// executing they pop tasks from the shared work queue. With this approach we
// avoid scheduling closures into the thread pool for each parallel task,
// because fixed thread pool overheads are high and XNNPACK operations tend to
// launch many parallel loops with larget number of very small tasks.
//
// Parallel loop runner can be configured by the `worker_timeslice` parameter,
// that defines the approximate amount of compute (in terms of wall time) that
// each persistent worker will handle. We rely on this parameter to avoid
// scheduling too many workers into the thread pool, because for tiny tasks the
// overheads can be prohibitively expensive.
//
// WARNING: ParallelLoopRunner is not thread-safe, and must be externally
// synchronized by the user.
class ParallelLoopRunner {
 public:
  explicit ParallelLoopRunner(const Eigen::ThreadPoolDevice* device);

  // Takes ownership of the runner and returns a done event. After the done
  // event is transferred to the caller, it is illegal to schedule more parallel
  // loops on the moved-from runner.
  static tsl::AsyncValueRef<tsl::Chain> TakeDoneEvent(
      ParallelLoopRunner&& runner);

  //===--------------------------------------------------------------------===//
  // Parallel dimensions and task coordinates APIs.
  //===--------------------------------------------------------------------===//

  // Parallel dimension iterated in [0, range) range in parallel.
  struct RangeDim {
    size_t range;
  };

  // Tiled dimension iterated in [0, range) range in `tile`-sized chunks.
  struct TileDim {
    size_t range;
    size_t tile;
  };

  // Parallel task index along the range dimension.
  struct RangeIndex {
    size_t offset;
  };

  // Parallel task index along the tile dimension.
  struct TileIndex {
    size_t offset;
    size_t count;
  };

  // Mapping from parallel loop dimension to the parallel task index. Defined
  // as template specializations below.
  template <typename Dim>
  struct TaskIndex;

  static size_t DimSize(RangeDim dim) { return dim.range; }

  static size_t DimSize(TileDim dim) {
    return tsl::MathUtil::CeilOfRatio(dim.range, dim.tile);
  }

  // Returns the number of tasks to be launched for the given dimensions.
  template <typename... Dims>
  static size_t NumTasks(Dims... dims);

  // Delinearizes linear `task_index` into the parallel task coordinates.
  template <typename... Dims>
  static std::tuple<typename TaskIndex<Dims>::Index...> Delinearize(
      size_t task_index, Dims... dims);

  // Adjusts tile dimensions to fit the product of all dimensions into the
  // desired number of tasks. Used in `ParallelizeDynamic` versions of the
  // parallel loop APIs, to minimize the task scheduling overheads.
  template <typename... Dims>
  static std::tuple<Dims...> DynamicDimensions(size_t target_num_tasks,
                                               Dims... dims);

  //===--------------------------------------------------------------------===//
  // Parallel loop APIs.
  //===--------------------------------------------------------------------===//

  using Task1D = std::function<void(RangeIndex i)>;

  using Task2D = std::function<void(RangeIndex i, RangeIndex j)>;

  using Task3D = std::function<void(RangeIndex i, RangeIndex j, RangeIndex k)>;

  using Task1DTile1D = std::function<void(TileIndex i)>;

  using Task2DTile1D = std::function<void(RangeIndex i, TileIndex j)>;

  using Task2DTile2D = std::function<void(TileIndex i, TileIndex j)>;

  using Task3DTile1D =
      std::function<void(RangeIndex i, RangeIndex j, TileIndex k)>;

  using Task3DTile2D =
      std::function<void(RangeIndex i, TileIndex j, TileIndex k)>;

  using Task4DTile2D =
      std::function<void(RangeIndex i, RangeIndex j, TileIndex k, TileIndex l)>;

  using Task5D = std::function<void(RangeIndex i, RangeIndex j, RangeIndex k,
                                    RangeIndex l, RangeIndex m)>;

  using Task5DTile2D = std::function<void(
      RangeIndex i, RangeIndex j, RangeIndex k, TileIndex l, TileIndex m)>;

  // IMPORTANT: For `dynamic` versions of the parallel loops, the runner is free
  // to adjust `count` for tiled dimensions to minimize the number of launched
  // tasks. Today we don't take advantage of this feature, and always launch the
  // same number of tasks as in regular parallel loops.

  // Launches `task` in parallel for each element of the `i` dimension.
  void Parallelize(RangeDim i, Task1D task);

  // Launches `task` in parallel for each element of the `i` and `j` dimensions.
  void Parallelize(RangeDim i, RangeDim j, Task2D task);

  // Launches `task` in parallel for each element of the `i` dimension.
  void Parallelize(TileDim i, Task1DTile1D task);

  // Launches `task` in parallel for each element of the `i` dimension.
  void ParallelizeDynamic(TileDim i, Task1DTile1D task);

  // Launches `task` in parallel for each element of the `i` and `j` dimensions.
  void Parallelize(RangeDim i, TileDim j, Task2DTile1D task);

  // Launches `task` in parallel for each element of the `i` and `j` dimensions.
  void ParallelizeDynamic(RangeDim i, TileDim j, Task2DTile1D task);

  // Launches `task` in parallel for each element of the `i` and `j` dimensions.
  void Parallelize(TileDim i, TileDim j, Task2DTile2D task);

  // Launches `task` in parallel for each element of the `i` and `j` dimensions.
  void ParallelizeDynamic(TileDim i, TileDim j, Task2DTile2D task);

  // Launches `task` in parallel for each element of the `i`, `j` and `k`
  // dimensions.
  void Parallelize(RangeDim i, RangeDim j, RangeDim k, Task3D task);

  // Launches `task` in parallel for each element of the `i`, `j` and `k`
  // dimensions.
  void Parallelize(RangeDim i, RangeDim j, TileDim k, Task3DTile1D task);

  // Launches `task` in parallel for each element of the `i`, `j` and `k`
  // dimensions.
  void Parallelize(RangeDim i, TileDim j, TileDim k, Task3DTile2D task);

  // Launches `task` in parallel for each element of the `i`, `j` and `k`
  // dimensions.
  void ParallelizeDynamic(RangeDim i, TileDim j, TileDim k, Task3DTile2D task);

  // Launches `task` in parallel for each element of the `i`, `j`, `k` and `l`
  // dimensions.
  void Parallelize(RangeDim i, RangeDim j, TileDim k, TileDim l,
                   Task4DTile2D task);

  // Launches `task` in parallel for each element of the `i`, `j`, `k` and `l`
  // dimensions.
  void ParallelizeDynamic(RangeDim i, RangeDim j, TileDim k, TileDim l,
                          Task4DTile2D task);

  // Launches `task` in parallel for each element of the `i`, `j`, `k`, `l` and
  // `m` dimensions.
  void Parallelize(RangeDim i, RangeDim j, RangeDim k, RangeDim l, RangeDim m,
                   Task5D task);

  // Launches `task` in parallel for each element of the `i`, `j`, `k`, `l` and
  // `m` dimensions.
  void Parallelize(RangeDim i, RangeDim j, RangeDim k, TileDim l, TileDim m,
                   Task5DTile2D task);

  // Resets the parallel loop runner `done_event` and returns the previous one
  // to the caller.
  tsl::AsyncValueRef<tsl::Chain> ResetDoneEvent();

  tsl::AsyncValueRef<tsl::Chain> done_event() const { return done_event_; }

  const Eigen::ThreadPoolDevice* device() const { return device_; }
  void set_device(const Eigen::ThreadPoolDevice* device) { device_ = device; }

  // Returns the number of threads in the underlying thread pool.
  size_t num_threads() const;

  // Returns true if the current thread belongs to the underlying thread pool.
  bool is_in_runner() const;

 private:
  // Forward declarations of the parallel tasks.
  struct ParallelTask1D;
  struct ParallelTask2D;
  struct ParallelTask3D;
  struct ParallelTask5D;
  struct ParallelTask1DTile1D;
  struct ParallelTask2DTile1D;
  struct ParallelTask2DTile2D;
  struct ParallelTask3DTile1D;
  struct ParallelTask3DTile2D;
  struct ParallelTask4DTile2D;
  struct ParallelTask5DTile2D;

  // Schedules `task` as the AndThen callback of the `done_event_`. Updates
  // `done_event_` to the new completion event.
  template <typename Task>
  void ScheduleOne(Task&& task);

  // Schedules `num_tasks` invocation of the `parallel_task` into the Eigen
  // thread pool when the `done_event_` becomes available. Updates `done_event_`
  // to the new completion event.
  template <typename ParallelTask>
  void ScheduleAll(size_t num_tasks, ParallelTask&& parallel_task);

  // Internal implementation of the parallel loop APIs.
  template <typename ParallelTask, typename... Dims, typename Task>
  void Parallelize(Dims... dims, Task&& task);

  // Internal implementation of the dynamic parallel loop APIs.
  template <typename ParallelTask, typename... Dims, typename Task>
  void ParallelizeDynamic(Dims... dims, Task&& task);

  // Async value that signals completion of the last scheduled parallel loop.
  tsl::AsyncValueRef<tsl::Chain> done_event_;

  // We keep a pointer to the Eigen thread pool device as an atomic variable
  // because we might update it between concurrent runs of XNNPACK operations
  // and non-atomic access to the `device_` pointer might lead to a data race.
  //
  // In practice PjRt CPU client owns the intra-op thread pool and passes it to
  // XLA via Thunk::ExecuteParams, and PjRt client might have multiple thread
  // pools for different NUMA nodes, and we have to be able to switch between
  // them from run to run.
  std::atomic<const Eigen::ThreadPoolDevice*> device_;
};

// An explicit specialization shall be declared in the namespace of which the
// template is a member, or, for member templates, in the namespace of which the
// enclosing class or enclosing class template is a member.

template <>
struct ParallelLoopRunner::TaskIndex<ParallelLoopRunner::RangeDim> {
  using Index = RangeIndex;
};

template <>
struct ParallelLoopRunner::TaskIndex<ParallelLoopRunner::TileDim> {
  using Index = TileIndex;
};

//===----------------------------------------------------------------------===//
// Parallel dimensions and task coordinates APIs.
//===----------------------------------------------------------------------===//

namespace internal {

template <typename Dim>
auto TaskStrides(Dim dim) {
  return std::array<size_t, 1>{1};
}

template <typename Dim, typename... Dims>
auto TaskStrides(Dim dim, Dims... dims) {
  std::array<size_t, 1 + sizeof...(Dims)> strides = {
      ParallelLoopRunner::NumTasks(dims...)};
  absl::c_copy(TaskStrides(dims...), &strides[1]);
  return strides;
}

template <size_t n>
auto TaskCoordinate(size_t task_index, std::array<size_t, n> strides) {
  std::array<size_t, n> coordinate;
  for (size_t d = 0; d < n; ++d) {
    coordinate[d] = task_index / strides[d];
    task_index %= strides[d];
  }
  return coordinate;
}

}  // namespace internal

template <typename... Dims>
size_t ParallelLoopRunner::NumTasks(Dims... dims) {
  return (DimSize(dims) * ...);
}

template <typename... Dims>
std::tuple<typename ParallelLoopRunner::TaskIndex<Dims>::Index...>
ParallelLoopRunner::Delinearize(size_t task_index, Dims... dims) {
  // Convert linear task index into the multidimensional parallel task index.
  auto strides = internal::TaskStrides(dims...);
  auto coord = internal::TaskCoordinate(task_index, strides);

  size_t d = 0;
  auto to_task_index = [&](auto dim) {
    size_t dim_index = coord[d++];
    DCHECK_LE(dim_index, DimSize(dim)) << "Dimension index is out of bounds";

    if constexpr (std::is_same_v<decltype(dim), RangeDim>) {
      return RangeIndex{dim_index};
    } else if constexpr (std::is_same_v<decltype(dim), TileDim>) {
      size_t offset = dim_index * dim.tile;
      return TileIndex{offset, std::min(dim.range - offset, dim.tile)};
    } else {
      static_assert(sizeof(decltype(dim)) == 0, "Unsupported dimension type");
    }
  };

  return std::make_tuple(to_task_index(dims)...);
}

template <typename... Dims>
std::tuple<Dims...> ParallelLoopRunner::DynamicDimensions(
    size_t target_num_tasks, Dims... dims) {
  constexpr size_t num_dims = sizeof...(Dims);

  // XNNPACK tends to choose too small tile sizes that creates too many tasks.
  // For dynamic versions of parallel loops we prefer tile sizes to be at
  // least 128 elements. We do further adjustments later to fit the number of
  // tasks into the target number of tasks.
  auto update_min_tile = [](auto& dim) {
    if constexpr (std::is_same_v<std::decay_t<decltype(dim)>, TileDim>) {
      static constexpr size_t kMinTileSize = 128;
      size_t multiple = tsl::MathUtil::CeilOfRatio(kMinTileSize, dim.tile);
      dim.tile = std::min(dim.range, dim.tile * multiple);
    }
  };
  (update_min_tile(dims), ...);

  // We can't adjust range dimensions and must execute a task for each index.
  auto to_range_tasks = [](auto dim) -> size_t {
    if constexpr (std::is_same_v<std::decay_t<decltype(dim)>, RangeDim>) {
      return dim.range;
    }
    return 1;
  };

  // Tile dimensions can be dynamically adjusted to reduce the number of tasks.
  auto to_tile_tasks = [](auto dim) -> size_t {
    if constexpr (std::is_same_v<std::decay_t<decltype(dim)>, TileDim>) {
      return tsl::MathUtil::CeilOfRatio(dim.range, dim.tile);
    }
    return 1;
  };

  std::array<size_t, num_dims> range_tasks = {to_range_tasks(dims)...};
  std::array<size_t, num_dims> tile_tasks = {to_tile_tasks(dims)...};

  size_t num_range_tasks =
      absl::c_accumulate(range_tasks, 1, std::multiplies<>());

  // The target number of tasks that should be assigned to tile dimensions.
  size_t target_dyn_tasks =
      tsl::MathUtil::CeilOfRatio(target_num_tasks, num_range_tasks);

  // Compute the target number of tile tasks for each tile dimension.
  std::array<size_t, num_dims> target_tile_tasks;

  for (size_t d = 0; d < num_dims; ++d) {
    if (target_dyn_tasks == 1 || tile_tasks[d] == 1) {
      // If we don't have any dyn task to assign or tile tasks to adjust, we
      // should process tile dimension as a single task.
      target_tile_tasks[d] = 1;

    } else if (target_dyn_tasks <= tile_tasks[d]) {
      // Assign the remaining dyn tasks to the current tile dimension.
      target_tile_tasks[d] = target_dyn_tasks;
      target_dyn_tasks = 1;

    } else {
      // Keep the number of tile tasks the same for current dimension and assign
      // the remaining dyn tasks to the inner tile dimensions.
      target_tile_tasks[d] = tile_tasks[d];
      target_dyn_tasks =
          tsl::MathUtil::CeilOfRatio(target_dyn_tasks, tile_tasks[d]);
    }
  }

  // Update tile dimensions to adjust the tile size to get the target number
  // of tasks per dimension.
  size_t d = 0;
  auto update_target_tile = [&](auto& dim) {
    if constexpr (std::is_same_v<std::decay_t<decltype(dim)>, TileDim>) {
      size_t target_tile_size =
          tsl::MathUtil::CeilOfRatio(dim.range, target_tile_tasks[d]);
      size_t multiple = tsl::MathUtil::CeilOfRatio(target_tile_size, dim.tile);
      dim.tile = dim.tile * multiple;
    } else {
      DCHECK_EQ(target_tile_tasks[d], 1)
          << "Target tile tasks for range dimensions must be 1";
    }
    ++d;
  };
  (update_target_tile(dims), ...);

  return std::make_tuple(dims...);
}

constexpr bool operator==(ParallelLoopRunner::RangeDim a,
                          ParallelLoopRunner::RangeDim b) {
  return a.range == b.range;
}

constexpr bool operator==(ParallelLoopRunner::TileDim a,
                          ParallelLoopRunner::TileDim b) {
  return a.range == b.range && a.tile == b.tile;
}

constexpr bool operator==(ParallelLoopRunner::RangeIndex a,
                          ParallelLoopRunner::RangeIndex b) {
  return a.offset == b.offset;
}

constexpr bool operator==(ParallelLoopRunner::TileIndex a,
                          ParallelLoopRunner::TileIndex b) {
  return a.offset == b.offset && a.count == b.count;
}

template <typename Sink>
void AbslStringify(Sink& sink, ParallelLoopRunner::RangeDim dim) {
  absl::Format(&sink, "RangeDim{range=%zu}", dim.range);
}

template <typename Sink>
void AbslStringify(Sink& sink, ParallelLoopRunner::TileDim dim) {
  absl::Format(&sink, "TileDim{range=%zu, tile=%zu}", dim.range, dim.tile);
}

template <typename Sink>
void AbslStringify(Sink& sink, ParallelLoopRunner::RangeIndex index) {
  absl::Format(&sink, "RangeIndex{offset=%zu}", index.offset);
}

template <typename Sink>
void AbslStringify(Sink& sink, ParallelLoopRunner::TileIndex index) {
  absl::Format(&sink, "TileIndex{offset=%zu, count=%zu}", index.offset,
               index.count);
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_PARALLEL_LOOP_RUNNER_H_

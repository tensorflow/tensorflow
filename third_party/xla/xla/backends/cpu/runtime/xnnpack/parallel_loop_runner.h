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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_PARALLEL_LOOP_RUNNER_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_PARALLEL_LOOP_RUNNER_H_

#include <atomic>
#include <cstddef>
#include <functional>
#include <new>
#include <optional>

#include "absl/container/fixed_array.h"
#include "absl/time/time.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"

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
// Parallel loop runner can be configured by the `worker_timeslice` parameter,
// that defines the approximate amount of compute (in terms of wall time) that
// each persistent worker will handle. We rely on this parameter to avoid
// scheduling too many workers into the thread pool, because for tiny tasks the
// overheads can be prohibitively expensive.
//
// WARNING: ParallelLoopRunner is not thread-safe, and must be externally
// synchronized by the user.
class ParallelLoopRunner {
  // Align all atomic counters to a cache line boundary to avoid false
  // sharing between multiple worker threads.
  static constexpr size_t kAtomicAlignment =
#if defined(__cpp_lib_hardware_interference_size)
      std::hardware_destructive_interference_size;
#else
      64;
#endif

 public:
  explicit ParallelLoopRunner(
      const Eigen::ThreadPoolDevice* device,
      std::optional<absl::Duration> worker_timeslice = std::nullopt);

  // Takes ownership of the runner and returns a done event. After the done
  // event is transferred to the caller, it is illegal to schedule more parallel
  // loops on the moved-from runner.
  static tsl::AsyncValueRef<tsl::Chain> TakeDoneEvent(
      ParallelLoopRunner&& runner);

  using Task1D = std::function<void(size_t offset)>;

  using Task1DTile1D = std::function<void(size_t offset, size_t extent)>;

  using Task2DTile1D =
      std::function<void(size_t offset_i, size_t offset_j, size_t extent_j)>;

  using Task3DTile2D =
      std::function<void(size_t offset_i, size_t offset_j, size_t offset_k,
                         size_t extent_j, size_t extent_k)>;

  // This function implements a parallel version of a following loop:
  //
  //   for (size_t i = 0; i < range; i++)
  //     task(i);
  void Parallelize(size_t range, Task1D task);

  // This function implements a parallel version of a following loop:
  //
  //   for (size_t i = 0; i < range; i += tile)
  //     task(i, std::min(range - i, tile));
  void Parallelize(size_t range, size_t tile, Task1DTile1D task);

  // This function implements a parallel version of a following loop:
  //
  //   for (size_t i = 0; i < range_i; i++)
  //     for (size_t j = 0; j < range_j; j += tile_j)
  //       task(i, j, min(range_j - j, tile_j));
  void Parallelize(size_t range_i, size_t range_j, size_t tile_j,
                   Task2DTile1D task);

  // This function implements a parallel version of a following loop:
  //
  //   for (size_t i = 0; i < range_i; i++)
  //     for (size_t j = 0; j < range_j; j += tile_j)
  //       for (size_t k = 0; k < range_k; k += tile_k)
  //         task(i, j, k, min(range_j - j, tile_j), min(range_k - k, tile_k));
  void Parallelize(size_t range_i, size_t range_j, size_t range_k,
                   size_t tile_j, size_t tile_k, Task3DTile2D task);

  // Resets the parallel loop runner `done_event` and returns the previous one
  // to the caller.
  tsl::AsyncValueRef<tsl::Chain> ResetDoneEvent();

  tsl::AsyncValueRef<tsl::Chain> done_event() const { return done_event_; }

  const Eigen::ThreadPoolDevice* device() const { return device_; }
  void set_device(const Eigen::ThreadPoolDevice* device) { device_ = device; }

  size_t num_threads() const;

  // ParallelLoopRunner uses "persistent workers" to execute parallel loops.
  // Workers get scheduled into the underlying thread pool and when they start
  // executing they pop tasks from the shared work queue. With this approach we
  // avoid scheduling closures into the thread pool for each parallel task,
  // because fixed thread pool overheads are high and XNNPACK operations tend to
  // launch many parallel loops with a large number of very small tasks.

  // Forward declare.
  class Worker;

  // A work queue that partitions `num_tasks` tasks into `num_partitions`
  // partitions processed by parallel workers.
  class WorkQueue {
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
  // wraps around to the initial partition) it returns an empty task.
  class Worker {
   public:
    Worker(size_t worker_index, WorkQueue* queue);

    std::optional<size_t> Pop();

   private:
    size_t worker_index_;
    size_t partition_index_;
    WorkQueue* queue_;
  };

 private:
  // Schedules `num_workers` persistent workers into the Eigen thread pool
  // that process `num_tasks` parallel tasks.
  template <typename ParallelTask>
  void Parallelize(tsl::CountDownAsyncValueRef<tsl::Chain> count_down,
                   size_t num_workers, size_t num_tasks,
                   ParallelTask&& parallel_task);

  // Schedules `task` as the AndThen callback of the `done_event_`. Updates
  // `done_event_` to the new completion event.
  template <typename Task>
  void ScheduleOne(Task&& task);

  // Schedules `num_tasks` invocation of the `parallel_task` into the Eigen
  // thread pool when the `done_event_` becomes available. Updates `done_event_`
  // to the new completion event.
  template <typename ParallelTask>
  void ScheduleAll(size_t num_tasks, ParallelTask&& parallel_task);

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

  // The approximate amount of compute (in terms of wall time) that each
  // persistent worker should handle.
  std::optional<absl::Duration> worker_timeslice_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_PARALLEL_LOOP_RUNNER_H_

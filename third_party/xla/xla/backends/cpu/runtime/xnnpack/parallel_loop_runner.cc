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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include "absl/base/optimization.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/lib/math/math_util.h"
#include "tsl/platform/logging.h"

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

// Schedules tasks in the [start_index, end_index) range into the Eigen thread
// pool using recursive work splitting. Executes the `start_index` task in the
// caller thread.
static void ScheduleRange(tsl::CountDownAsyncValueRef<tsl::Chain> count_down,
                          Eigen::ThreadPoolDevice* device, size_t start_index,
                          size_t end_index, Task task) {
  CHECK_LT(start_index, end_index) << "Invalid task index range";  // Crash OK
  while (end_index - start_index > 1) {
    uint64_t mid_index = (start_index + end_index) / 2;
    device->enqueueNoNotification([device, mid_index, end_index, task,
                                   count_down] {
      ScheduleRange(std::move(count_down), device, mid_index, end_index, task);
    });
    end_index = mid_index;
  }
  task(start_index);
  count_down.CountDown();
}

ParallelLoopRunner::ParallelLoopRunner(Eigen::ThreadPoolDevice* device)
    : done_event_(OkDoneEventSingleton()), device_(device) {}

size_t ParallelLoopRunner::num_threads() const {
  return device_->numThreadsInPool();
}

tsl::AsyncValueRef<tsl::Chain> ParallelLoopRunner::TakeDoneEvent(
    ParallelLoopRunner&& runner) {
  return std::move(runner.done_event_);
}

void ParallelLoopRunner::Parallelize(size_t range, size_t tile, Task1D task) {
  DCHECK(done_event_) << "Parallel loop runner is in moved-from state";

  size_t num_tasks = tsl::MathUtil::CeilOfRatio(range, tile);
  DCHECK_GT(num_tasks, 0) << "Expected at least one task";

  // Fast path for the degenerate parallel loop with single task.
  if (ABSL_PREDICT_TRUE(num_tasks == 1)) {
    DCHECK_EQ(range, tile) << "Expected range to be equal to tile";

    if (ABSL_PREDICT_TRUE(done_event_.IsConcrete())) {
      // If done event is already available, execute the task immediately in the
      // caller thread. In this case we don't need to overwrite the done event,
      // because the existing one will correctly represent the state of the
      // parallel loop runner (all scheduled loops are ready).
      task(0, range);

    } else {
      // If done event is not available, we have to overwrite it with a new one
      // that will be set to concrete state after the task is executed.
      auto done_event = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
      done_event_.AndThen([range, done_event, task = std::move(task)] {
        task(0, range);
        done_event.SetStateConcrete();
      });
      done_event_ = std::move(done_event);
    }

    return;
  }

  // Schedule `num_tasks` into the underlying thread pool when done event
  // becomes available.
  tsl::CountDownAsyncValueRef<tsl::Chain> count_down(num_tasks);
  auto done_event = count_down.AsRef();

  done_event_.AndThen([this, num_tasks, range, tile, task = std::move(task),
                       count_down = std::move(count_down)] {
    ScheduleRange(std::move(count_down), device_, 0, num_tasks,
                  [range, tile, task = std::move(task)](size_t task_index) {
                    size_t offset = task_index * tile;
                    size_t extent = std::min(range - offset, tile);
                    task(offset, extent);
                  });
  });
  done_event_ = std::move(done_event);
}

}  // namespace xla::cpu

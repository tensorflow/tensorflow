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

#include "xla/hlo/utils/concurrency/tsl_task_executor.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/utils/concurrency/type_adapters.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"

namespace xla::concurrency {
namespace {

int ResolveParallelism(std::optional<int> parallelism) {
  if (!parallelism.has_value() || *parallelism <= 0 ||
      *parallelism > tsl::port::MaxParallelism()) {
    return tsl::port::MaxParallelism();
  }
  return *parallelism;
}

// Run all actions in a loop within a single schedulable unit.
// This way we guarantee sequential execution.
void DispatchSequentialRun(tsl::thread::ThreadPool* thread_pool,
                           absl::Status& final_status,
                           absl::BlockingCounter& finished_counter,
                           std::vector<Task>& original_actions) {
  thread_pool->Schedule(
      [&final_status, &finished_counter,
       actions = TurnMoveOnlyToCopyableWithCaching<absl::Status>::FromVector(
           std::move(original_actions))]() mutable {
        for (auto& action : actions) {
          auto action_status = std::move(action)();
          if (!action_status.ok()) {
            final_status = action_status;
            finished_counter
                .DecrementCount();  // this will unblock the caller; count == 1
            return;
          }
        }
        final_status = absl::OkStatus();
        finished_counter.DecrementCount();
      });
}

// Run each action as a separately schedulable unit.
void DispatchParallelRun(tsl::thread::ThreadPool* thread_pool,
                         absl::Status& final_status,
                         absl::BlockingCounter& finished_counter,
                         absl::Mutex& mu_final_status,
                         std::vector<Task>& actions) {
  // When using `tsl::thread::ThreadPool` directly we need to count successful
  // tasks and signal finish once all are done. Without `finished_conuter` we
  // do not know when to set `absl::OkStatus()` on the latch.
  for (auto& action : actions) {
    thread_pool->Schedule([&final_status, &finished_counter, &mu_final_status,
                           action = TurnMoveOnlyToCopyableWithCaching(
                               std::move(action))]() mutable {
      // Pseudo-cancellation.
      // The actions will not be invoked. However, the `ThreadPool` will
      // iterate through all the scheduled tasks and check the status.
      // Cancellation complexity is O(#tasks).
      absl::Status current_status = absl::OkStatus();
      {
        absl::ReaderMutexLock reader_lock{&mu_final_status};
        current_status = final_status;
      }
      if (current_status.ok()) {
        auto action_status = std::move(action)();
        if (!action_status.ok()) {
          absl::MutexLock write_lock{&mu_final_status};
          final_status = action_status;
        }
      }
      // Must be the last thing we touch.
      finished_counter.DecrementCount();
    });
  }
}

}  // namespace

TslTaskExecutor::TslTaskExecutor(std::optional<int> max_parallelism) {
  auto parallelism = ResolveParallelism(max_parallelism);

  thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), kThreadPoolName, parallelism);
}

absl::Status TslTaskExecutor::ExecuteIndependentTasks(
    std::vector<Task> tasks, std::optional<int> parallelism) {
  auto actual_parallelism = ResolveParallelism(parallelism);

  if (actual_parallelism == 1) {  // NOMUTANTS -- Functionally equivalent code
                                  // paths; but the other is parallelized.
    // Enforce sequential execution for debugging.
    absl::BlockingCounter finished_counter(1);
    absl::Status final_status = absl::OkStatus();
    DispatchSequentialRun(thread_pool_.get(), final_status, finished_counter,
                          tasks);
    finished_counter.Wait();
    return final_status;
  }

  absl::Status final_status = absl::OkStatus();
  {
    absl::BlockingCounter finished_counter(tasks.size());
    absl::Mutex mu_final_status;

    DispatchParallelRun(thread_pool_.get(), final_status, finished_counter,
                        mu_final_status, tasks);
    // Wait for all tasks to finish, so `latch` can be destroyed.
    finished_counter.Wait();
  }
  return final_status;
}
}  // namespace xla::concurrency

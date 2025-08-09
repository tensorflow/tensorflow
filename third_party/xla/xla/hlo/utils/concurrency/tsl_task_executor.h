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

#ifndef XLA_HLO_UTILS_CONCURRENCY_TSL_TASK_EXECUTOR_H_
#define XLA_HLO_UTILS_CONCURRENCY_TSL_TASK_EXECUTOR_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::concurrency {

// Tasks must signal a status. We promise to call tasks at most once.
using Task = absl::AnyInvocable<absl::Status() &&>;

// A thread pool with a higher-level API for parallelization of compiler passes.
// Not thread safe.
//
// All calls are synchronous. Specifically, the call to parallelize work blocks
// until the work is done, or canceled due to failure of any of the submitted
// tasks. Once a parallelization call unblocks implementatinos must guarantee
// that no value caputerd by any of the submitted tasks would be accessed going
// forward. Specifically, any captured values can be destroyed after the
// parallelization call returns, even when the work is cancelled.
//
// This design is chosen for simplicity & expediency. It has obvious downside
// that blocking until all work is done will result in many threads idling
// towards the end of the execution.
//
// Features
// - Batch submitted for execution fails if any individual task fails.
// - Guarantees in-order processing of tasks when `parallelism` is 1.
class TslTaskExecutor {
 public:
  // Runs all the actions on `parallelism` theads. If fewer threads are
  // available, runs on as many as it has.
  //
  // When `parallelism` == 1 sequential execution is guaranteed.
  absl::Status ExecuteIndependentTasks(
      std::vector<Task> tasks, std::optional<int> parallelism = std::nullopt);

  explicit TslTaskExecutor(std::optional<int> max_parallelism = std::nullopt);

 private:
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;

  // std::string because `tsl::thread::ThreadPool` wants a string and not a
  // view.
  const std::string kThreadPoolName = "TslTaskExecutor";
};

}  // namespace xla::concurrency
#endif  // XLA_HLO_UTILS_CONCURRENCY_TSL_TASK_EXECUTOR_H_

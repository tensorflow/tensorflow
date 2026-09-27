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

#include <cstdint>
#include <optional>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::concurrency {

class ABSL_DEPRECATED(
    "Prefer `xla::concurrency::DefaultExecutor()` if you want to use a default "
    "executor available to the XLA process. Otherwise use TSL ThreadPool "
    "directly, or any other tsl::Executor implementation.") TslTaskExecutor
    : public tsl::Executor {
 public:
  explicit TslTaskExecutor(
      std::optional<int32_t> max_parallelism = std::nullopt,
      absl::string_view name = "TslTaskExecutor");

  void Execute(Task task) final;

 private:
  tsl::thread::ThreadPool thread_pool_;
};

}  // namespace xla::concurrency
#endif  // XLA_HLO_UTILS_CONCURRENCY_TSL_TASK_EXECUTOR_H_

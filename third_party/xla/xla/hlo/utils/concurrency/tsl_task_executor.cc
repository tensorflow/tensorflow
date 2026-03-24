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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"

namespace xla::concurrency {

static int32_t ResolveParallelism(std::optional<int32_t> parallelism) {
  if (!parallelism.has_value() || *parallelism <= 0 ||
      *parallelism > tsl::port::MaxParallelism()) {
    return tsl::port::MaxParallelism();
  }
  return *parallelism;
}

TslTaskExecutor::TslTaskExecutor(std::optional<int32_t> max_parallelism,
                                 absl::string_view name)
    : thread_pool_(tsl::Env::Default(), std::string(name),
                   ResolveParallelism(max_parallelism)) {}

void TslTaskExecutor::Execute(Task task) {
  thread_pool_.AsExecutor()->Execute(std::move(task));
}

}  // namespace xla::concurrency

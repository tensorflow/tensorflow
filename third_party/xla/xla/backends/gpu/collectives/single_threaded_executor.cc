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

#include "xla/backends/gpu/collectives/single_threaded_executor.h"

#include <utility>

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/platform/threadpool_async_executor.h"

namespace xla::gpu {

SingleThreadedExecutor::SingleThreadedExecutor(tsl::Env& env)
    : thread_pool_(&env, "SingleThreadedExecutor", 1),
      executor_(&thread_pool_) {}

void SingleThreadedExecutor::Execute(SingleThreadedExecutor::Task task) {
  executor_.Execute(std::move(task));
}

}  // namespace xla::gpu

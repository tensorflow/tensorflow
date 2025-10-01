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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_SINGLE_THREADED_EXECUTOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_SINGLE_THREADED_EXECUTOR_H_

#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/platform/threadpool_async_executor.h"

namespace xla::gpu {

// An Executor that executes all tasks on a single thread.
//
// Tasks are executed concurrently to the thread that calls the Execute method,
// but tasks are not executed concurrently to each other.
class SingleThreadedExecutor : public tsl::AsyncValue::Executor {
 public:
  explicit SingleThreadedExecutor(tsl::Env& env = *tsl::Env::Default());
  void Execute(Task task) override;

 private:
  tsl::thread::ThreadPool thread_pool_;
  tsl::thread::ThreadPoolAsyncExecutor executor_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_SINGLE_THREADED_EXECUTOR_H_

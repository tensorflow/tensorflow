/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PLATFORM_THREADPOOL_ASYNC_EXECUTOR_H_
#define XLA_TSL_PLATFORM_THREADPOOL_ASYNC_EXECUTOR_H_

#include <utility>

#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/platform/threadpool.h"

namespace tsl::thread {

// An adaptor for a ThreadPool that converts it into the AsyncValue:Executor.
//
// AsncValue::Executor task is a move-only absl::AnyInvocable, and ThreadPool
// expects a copyable std::function. This class adapts the two and makes sure
// that the task is deleted when it's done executing.
class ThreadPoolAsyncExecutor : public AsyncValue::Executor {
 public:
  explicit ThreadPoolAsyncExecutor(ThreadPool* thread_pool)
      : thread_pool_(thread_pool) {}

  void Execute(Task task) final {
    auto* task_ptr = new Task(std::move(task));
    thread_pool_->Schedule([task_ptr] {
      (*task_ptr)();
      delete task_ptr;
    });
  }

 private:
  ThreadPool* thread_pool_;
};

}  // namespace tsl::thread

#endif  // XLA_TSL_PLATFORM_THREADPOOL_ASYNC_EXECUTOR_H_

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

#ifndef XLA_TSL_CONCURRENCY_EXECUTOR_H_
#define XLA_TSL_CONCURRENCY_EXECUTOR_H_

#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/functional/any_invocable.h"

namespace tsl {

// Executor allows to customize where the `AsyncValue` and `Future` callbacks
// are executed. By default the callback is executed on the caller thread
// if async value (or future) is already available, or on a thread that sets
// async value (or future) available (setting a value or an error), which can
// accidentally lead to executing a very expensive computations on an IO thread.
//
// IMPORTANT: It's the caller responsibility to ensure that executor passed to
// all `AndThen`, `Map` or `OnReady` function calls stay alive while async
// values (or futures) have unresolved callbacks waiting to be invoked.
class Executor {
 public:
  using Task = absl::AnyInvocable<void() &&>;

  virtual ~Executor() = default;

  virtual void Execute(Task task) = 0;
};

// Executor that executes tasks inline in the caller thread.
class InlineExecutor final : public Executor {
 public:
  // Returns a singleton instance of the inline executor.
  static InlineExecutor& Instance() {
    static absl::NoDestructor<InlineExecutor> executor;
    return *executor;
  }

  void Execute(Task task) final { std::move(task)(); }
};

}  // namespace tsl

#endif  // XLA_TSL_CONCURRENCY_EXECUTOR_H_

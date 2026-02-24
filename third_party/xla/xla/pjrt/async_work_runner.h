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

#ifndef XLA_PJRT_ASYNC_WORK_RUNNER_H_
#define XLA_PJRT_ASYNC_WORK_RUNNER_H_

#include <utility>

#include "absl/base/macros.h"
#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

// Async work runner abstracts away the implementation of the underlying thread
// pool (or concurrent work queue).
class AsyncWorkRunner : public tsl::Executor {
 public:
  AsyncWorkRunner() = default;
  virtual ~AsyncWorkRunner() = default;

  // Executes `task` when all dependencies become ready.
  void ExecuteWhenReady(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> dependencies,
      Task task) {
    tsl::RunWhenReady(dependencies, [this, task = std::move(task)]() mutable {
      this->Execute(std::move(task));
    });
  }

  ABSL_DEPRECATE_AND_INLINE()
  void Schedule(absl::AnyInvocable<void() &&> work) {
    Execute(std::move(work));
  }

  ABSL_DEPRECATE_AND_INLINE()
  void ScheduleWhenReady(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
      absl::AnyInvocable<void() &&> work) {
    ExecuteWhenReady(values, std::move(work));
  }
};

}  // namespace xla

#endif  // XLA_PJRT_ASYNC_WORK_RUNNER_H_

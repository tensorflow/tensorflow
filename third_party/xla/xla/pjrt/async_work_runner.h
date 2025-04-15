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

#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

// Async work runner abstracts away the implementation of the underlying thread
// pool (or concurrent work queue).
class AsyncWorkRunner {
 public:
  virtual ~AsyncWorkRunner() = default;

  // `work` euqueued by `Schedule` may run on the calling thread.
  virtual void Schedule(absl::AnyInvocable<void()> work) = 0;
  virtual void ScheduleWhenReady(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
      absl::AnyInvocable<void()> work) = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_ASYNC_WORK_RUNNER_H_

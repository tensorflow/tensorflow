/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_NEVER_RUN_ON_FIBER_H_
#define XLA_PJRT_NEVER_RUN_ON_FIBER_H_

#include <optional>
#include <type_traits>

#include "absl/synchronization/notification.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/tsl/platform/env.h"

namespace xla {

// Synchronously invokes `f` while ensuring that it never runs on a
// cooperatively scheduled fiber.
//
// This is useful for safely invoking functions that may perform blocking
// operations that are not compatible with cooperatively scheduled fibers such
// as pthread synchronization primitives vs. Google's fibers.
template <typename F>
std::invoke_result_t<F> NeverRunOnFiber(AsyncWorkRunner* async_work_runner,
                                        F&& f) {
  using T = std::invoke_result_t<F>;
  if (tsl::Env::Default()->IsCurrentThreadFiber()) {
    if constexpr (std::is_void_v<T>) {
      absl::Notification done;
      async_work_runner->Schedule([&]() {
        std::forward<F>(f)();
        done.Notify();
      });
      done.WaitForNotification();
    } else {
      std::optional<T> result;
      absl::Notification done;
      async_work_runner->Schedule([&]() {
        result = std::forward<F>(f)();
        done.Notify();
      });
      done.WaitForNotification();
      return *std::move(result);
    }
  }
  return std::forward<F>(f)();
}

}  // namespace xla

#endif  // XLA_PJRT_NEVER_RUN_ON_FIBER_H_

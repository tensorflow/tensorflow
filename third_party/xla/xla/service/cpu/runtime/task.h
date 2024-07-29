/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_TASK_H_
#define XLA_SERVICE_CPU_RUNTIME_TASK_H_

#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"

namespace xla::cpu {

// Converts absl::AnyInvocable to a std::function. absl::AnyInvocable is not
// copyable, and we need to wrap it into a std::shared_ptr to be able to pass
// to thread pools (i.e. Eigen) that expect copyable std::function task type.
inline auto ToCopyableTask(absl::AnyInvocable<void()> task) {
  return [shared_task = std::make_shared<decltype(task)>(std::move(task))] {
    (*shared_task)();
  };
}

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_TASK_H_

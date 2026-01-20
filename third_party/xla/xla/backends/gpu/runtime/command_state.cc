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

#include "xla/backends/gpu/runtime/command_state.h"

#include <atomic>
#include <cstdint>
#include <memory>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"

namespace xla::gpu {

CommandStateManager::TypeId CommandStateManager::GetNextTypeId() {
  static absl::NoDestructor<std::atomic<int64_t>> counter;
  return TypeId(counter->fetch_add(1));
}

CommandState* absl_nullable CommandStateManager::GetOrNull(
    const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
    TypeId type_id) {
  Key key = {cmd, command_buffer, type_id};
  if (auto it = state_.find(key); it != state_.end()) {
    return it->second.get();
  }
  return nullptr;
}

CommandState* absl_nonnull CommandStateManager::GetOrCreate(
    const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
    TypeId type_id, absl::FunctionRef<std::unique_ptr<CommandState>()> create) {
  Key key = {cmd, command_buffer, type_id};
  if (auto it = state_.find(key); it != state_.end()) {
    return it->second.get();
  }
  return state_.try_emplace(key, create()).first->second.get();
}

}  // namespace xla::gpu

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

#ifndef XLA_SERVICE_CPU_RUNTIME_WHILE_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_WHILE_THUNK_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime/thunk_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// While loop written as two thunk sequences:
//
// while (condition_thunk.Execute(...) && condition_buffer) {
//   body_thunk.Execute(...);
// }
//
// Condition buffer must be a i1 (bool) buffer that holds a loop predicate.
class WhileThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<WhileThunk>> Create(
      Info info, BufferAllocation::Slice cond_buffer,
      ThunkSequence cond_sequence, ThunkSequence body_sequence);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;
  ResourceUses resource_uses() const final;

 private:
  WhileThunk(Info info, BufferAllocation::Slice cond_buffer,
             ThunkExecutor cond_executor, ThunkExecutor body_executor);

  BufferAllocation::Slice cond_buffer_;
  ThunkExecutor cond_executor_;
  ThunkExecutor body_executor_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_WHILE_THUNK_H_

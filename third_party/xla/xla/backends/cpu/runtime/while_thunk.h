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

#ifndef XLA_BACKENDS_CPU_RUNTIME_WHILE_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_WHILE_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/service/buffer_assignment.h"
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
      ThunkSequence cond_sequence, ThunkSequence body_sequence,
      std::optional<int64_t> trip_count = std::nullopt);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;
  ResourceUses resource_uses() const final;

 private:
  WhileThunk(Info info, BufferAllocation::Slice cond_buffer,
             ThunkExecutor cond_executor, ThunkExecutor body_executor,
             std::optional<int64_t> trip_count);

  tsl::AsyncValueRef<ExecuteEvent> ExecuteForLoop(const ExecuteParams& params,
                                                  int64_t trip_count);

  tsl::AsyncValueRef<ExecuteEvent> ExecuteWhileLoop(const ExecuteParams& params,
                                                    bool* condition);

  // If `cond` or `body` thunk sequence return unavailable async values, then
  // we execute the while loop asynchronously by chaining `Execute` calls via
  // `AndThen` callbacks. This execution mode adds significant overheads, so we
  // try to avoid it when possible and run everything in the caller thread.

  tsl::AsyncValueRef<ExecuteEvent> ExecuteAsyncForLoop(
      const ExecuteParams& params, tsl::AsyncValueRef<ExecuteEvent> dependency,
      int64_t loop_counter, int64_t trip_count);

  tsl::AsyncValueRef<ExecuteEvent> ExecuteAsyncWhileLoop(
      const ExecuteParams& params, tsl::AsyncValueRef<ExecuteEvent> dependency,
      bool* condition);

  BufferAllocation::Slice cond_buffer_;
  ThunkExecutor cond_executor_;
  ThunkExecutor body_executor_;

  // Statically known trip count. If available, WhileThunk::Execute will not
  // execute `cond_executor_` and simply call `body_executor_` `trip_count`
  // times (effectively converting while loop into a for loop).
  std::optional<int64_t> trip_count_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_WHILE_THUNK_H_

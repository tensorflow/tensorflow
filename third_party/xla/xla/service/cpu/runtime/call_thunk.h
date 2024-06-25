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

#ifndef XLA_SERVICE_CPU_RUNTIME_CALL_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_CALL_THUNK_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime/thunk_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// A thunk constructed from a call instruction that simply calls a thunk
// sequence emitted from the called computation.
class CallThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<CallThunk>> Create(
      Info info, ThunkSequence called_sequence);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

 private:
  CallThunk(Info info, ThunkExecutor called_executor);

  ThunkExecutor called_executor_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_CALL_THUNK_H_

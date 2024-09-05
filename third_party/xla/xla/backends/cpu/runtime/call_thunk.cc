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

#include "xla/backends/cpu/runtime/call_thunk.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<CallThunk>> CallThunk::Create(
    Info info, ThunkSequence called_sequence) {
  TF_ASSIGN_OR_RETURN(auto called_executor,
                      ThunkExecutor::Create(std::move(called_sequence)));
  return absl::WrapUnique(
      new CallThunk(std::move(info), std::move(called_executor)));
}

CallThunk::CallThunk(Info info, ThunkExecutor called_executor)
    : Thunk(Kind::kCall, std::move(info)),
      called_executor_(std::move(called_executor)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> CallThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });
  return called_executor_.Execute(params);
}

CallThunk::BufferUses CallThunk::buffer_uses() const {
  return called_executor_.buffer_uses();
}

CallThunk::ResourceUses CallThunk::resource_uses() const {
  return called_executor_.resource_uses();
}

}  // namespace xla::cpu

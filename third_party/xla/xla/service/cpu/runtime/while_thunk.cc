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

#include "xla/service/cpu/runtime/while_thunk.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime/thunk_executor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<WhileThunk>> WhileThunk::Create(
    Info info, BufferAllocation::Slice cond_buffer, ThunkSequence cond_sequence,
    ThunkSequence body_sequence) {
  TF_ASSIGN_OR_RETURN(ThunkExecutor cond_executor,
                      ThunkExecutor::Create(std::move(cond_sequence)));
  TF_ASSIGN_OR_RETURN(ThunkExecutor body_executor,
                      ThunkExecutor::Create(std::move(body_sequence)));
  return absl::WrapUnique(new WhileThunk(std::move(info), cond_buffer,
                                         std::move(cond_executor),
                                         std::move(body_executor)));
}

WhileThunk::WhileThunk(Info info, BufferAllocation::Slice cond_buffer,
                       ThunkExecutor cond_executor, ThunkExecutor body_executor)
    : Thunk(Kind::kWhile, std::move(info)),
      cond_buffer_(cond_buffer),
      cond_executor_(std::move(cond_executor)),
      body_executor_(std::move(body_executor)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> WhileThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase cond_data,
      params.buffer_allocations->GetDeviceAddress(cond_buffer_));

  bool* condition = reinterpret_cast<bool*>(cond_data.opaque());

  // TODO(ezhulenev): Remove `BlockUntilReady` calls and make WhileThunk
  // asynchronous by chaining `Execute` calls via `AndThen` callbacks.

  auto init_event = cond_executor_.Execute(params);
  tsl::BlockUntilReady(init_event);
  if (init_event.IsError()) return init_event.GetError();

  while (*condition) {
    auto body_event = body_executor_.Execute(params);
    tsl::BlockUntilReady(body_event);
    if (body_event.IsError()) return body_event.GetError();

    auto cond_event = cond_executor_.Execute(params);
    tsl::BlockUntilReady(cond_event);
    if (cond_event.IsError()) return cond_event.GetError();
  }

  return OkExecuteEvent();
}

WhileThunk::BufferUses WhileThunk::buffer_uses() const {
  BufferUses buffer_uses = {{cond_buffer_, BufferUse::kWrite}};

  BufferUses cond_uses = cond_executor_.buffer_uses();
  buffer_uses.insert(buffer_uses.end(), cond_uses.begin(), cond_uses.end());

  BufferUses body_uses = body_executor_.buffer_uses();
  buffer_uses.insert(buffer_uses.end(), body_uses.begin(), body_uses.end());

  return buffer_uses;
}

}  // namespace xla::cpu

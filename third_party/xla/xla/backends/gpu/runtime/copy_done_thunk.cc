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

#include "xla/backends/gpu/runtime/copy_done_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

CopyDoneThunk::CopyDoneThunk(
    ThunkInfo thunk_info, std::shared_ptr<CopyThunk::AsyncEvents> async_events,
    int64_t copy_start_instr_id)
    : Thunk(Thunk::kCopyDone, std::move(thunk_info)),
      async_events_(std::move(async_events)),
      copy_start_instr_id_(copy_start_instr_id) {}

absl::Status CopyDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "CopyDone thunk between a host and a device for instruction id: "
          << copy_start_instr_id_;
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Event> event,
                      async_events_->Extract(executor, copy_start_instr_id_));
  return params.stream->WaitFor(event.get());
}

absl::StatusOr<ThunkProto> CopyDoneThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CopyDoneThunkProto* copy_done_thunk_proto = proto.mutable_copy_done_thunk();
  if (auto id = GetAsyncEventsUniqueId()) {
    copy_done_thunk_proto->set_async_events_unique_id(id->value());
  }
  copy_done_thunk_proto->set_copy_start_instr_id(copy_start_instr_id_);

  return proto;
}

absl::StatusOr<std::unique_ptr<CopyDoneThunk>> CopyDoneThunk::FromProto(
    ThunkInfo thunk_info, const CopyDoneThunkProto& thunk_proto,
    CopyThunk::AsyncEventsMap& async_events_map) {
  std::shared_ptr<CopyThunk::AsyncEvents> async_events;
  if (thunk_proto.has_async_events_unique_id()) {
    auto [async_event_it, _] = async_events_map.try_emplace(
        AsyncEventsUniqueId(thunk_proto.async_events_unique_id()),
        std::make_shared<CopyThunk::AsyncEvents>());
    async_events = async_event_it->second;
  }
  return std::make_unique<CopyDoneThunk>(std::move(thunk_info),
                                         std::move(async_events),
                                         thunk_proto.copy_start_instr_id());
}

std::optional<AsyncEventsUniqueId> CopyDoneThunk::GetAsyncEventsUniqueId()
    const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}

}  // namespace gpu
}  // namespace xla

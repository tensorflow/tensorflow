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

#include "xla/backends/cpu/runtime/rng_state_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<RngGetAndUpdateStateThunk>>
RngGetAndUpdateStateThunk::Create(Info info,
                                  BufferAllocation::Slice state_buffer,
                                  int64_t delta) {
  return absl::WrapUnique(
      new RngGetAndUpdateStateThunk(std::move(info), state_buffer, delta));
}

RngGetAndUpdateStateThunk::RngGetAndUpdateStateThunk(
    Info info, BufferAllocation::Slice state_buffer, int64_t delta)
    : Thunk(Kind::kRngGetAndUpdateState, info),
      state_buffer_(state_buffer),
      rng_state_(delta) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> RngGetAndUpdateStateThunk::Execute(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase state_data,
      params.buffer_allocations->GetDeviceAddress(state_buffer_));

  if (state_data.size() != sizeof(absl::int128)) {
    return InvalidArgument("Invalid state buffer size: %d", state_data.size());
  }

  VLOG(3) << absl::StreamFormat("Rng get and update state");
  VLOG(3) << absl::StreamFormat("  state: %s (%p)", state_buffer_.ToString(),
                                state_data.opaque());

  uint64_t* data = static_cast<uint64_t*>(state_data.opaque());
  rng_state_.GetAndUpdateState(data);

  return OkExecuteEvent();
}

}  // namespace xla::cpu

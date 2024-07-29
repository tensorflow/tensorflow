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

#include "xla/service/cpu/runtime/replica_id_thunk.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<ReplicaIdThunk>> ReplicaIdThunk::Create(
    Info info, BufferAllocation::Slice replica_id_buffer) {
  return absl::WrapUnique(
      new ReplicaIdThunk(std::move(info), replica_id_buffer));
}

ReplicaIdThunk::ReplicaIdThunk(Info info,
                               BufferAllocation::Slice replica_id_buffer)
    : Thunk(Kind::kReplicaId, info), replica_id_buffer_(replica_id_buffer) {}

tsl::AsyncValueRef<ReplicaIdThunk::ExecuteEvent> ReplicaIdThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase replica_id_data,
      params.buffer_allocations->GetDeviceAddress(replica_id_buffer_));

  TF_RET_CHECK(replica_id_data.size() == sizeof(int32_t))
      << "Replica id buffer must be able to fit replica id value";

  TF_RET_CHECK(params.collective_params)
      << "Replica id requires collective params";

  TF_ASSIGN_OR_RETURN(
      int32_t replica_id,
      params.collective_params->device_assignment->ReplicaIdForDevice(
          params.collective_params->global_device_id));

  VLOG(3) << absl::StreamFormat("Replica id: %d", replica_id);
  VLOG(3) << absl::StreamFormat("  replica_id: slice %s (%p)",
                                replica_id_buffer_.ToString(),
                                replica_id_data.opaque());

  std::memcpy(replica_id_data.opaque(), &replica_id, sizeof(int32_t));
  return OkExecuteEvent();
}

ReplicaIdThunk::BufferUses ReplicaIdThunk::buffer_uses() const {
  return {BufferUse::Write(replica_id_buffer_)};
}

}  // namespace xla::cpu

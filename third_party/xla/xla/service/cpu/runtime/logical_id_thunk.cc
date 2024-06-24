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
#include "xla/service/cpu/runtime/logical_id_thunk.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/global_device_id.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

static Thunk::Kind ToThunkKind(LogicalIdKind logical_id_kind) {
  switch (logical_id_kind) {
    case LogicalIdKind::kPartitionId:
      return Thunk::Kind::kPartitionId;
    case LogicalIdKind::kReplicaId:
      return Thunk::Kind::kReplicaId;
  }
}

template <LogicalIdKind type>
absl::StatusOr<std::unique_ptr<LogicalIdThunk<type>>>
LogicalIdThunk<type>::Create(Info info,
                             BufferAllocation::Slice logical_id_buffer) {
  return absl::WrapUnique(
      new LogicalIdThunk(std::move(info), logical_id_buffer));
}

template <LogicalIdKind type>
LogicalIdThunk<type>::LogicalIdThunk(Info info,
                                     BufferAllocation::Slice logical_id_buffer)
    : Thunk(ToThunkKind(type), info), logical_id_buffer_(logical_id_buffer) {}

template <LogicalIdKind type>
static constexpr auto ToString() {
  if constexpr (type == LogicalIdKind::kPartitionId) {
    return "Partition";
  } else if constexpr (type == LogicalIdKind::kReplicaId) {
    return "Replica";
  }
}

template <LogicalIdKind type>
absl::StatusOr<int32_t> LogicalIdThunk<type>::GetIdForDevice(
    const DeviceAssignment* device_assignment, GlobalDeviceId device_id) const {
  if constexpr (type == LogicalIdKind::kPartitionId) {
    return device_assignment->PartitionIdForDevice(device_id);
  } else if constexpr (type == LogicalIdKind::kReplicaId) {
    return device_assignment->ReplicaIdForDevice(device_id);
  }
}

template <LogicalIdKind type>
tsl::AsyncValueRef<typename LogicalIdThunk<type>::ExecuteEvent>
LogicalIdThunk<type>::Execute(const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase logical_id_data,
      params.buffer_allocations->GetDeviceAddress(logical_id_buffer_));

  TF_RET_CHECK(logical_id_data.size() == sizeof(int32_t))
      << "Logical id buffer must be able to fit logical id value";

  TF_RET_CHECK(params.collective_params)
      << ToString<type>() << " id requires collective params";

  TF_ASSIGN_OR_RETURN(
      int32_t logical_id,
      GetIdForDevice(params.collective_params->device_assignment,
                     params.collective_params->global_device_id));

  VLOG(3) << absl::StreamFormat("%s id: %d", ToString<type>(), logical_id);
  VLOG(3) << absl::StreamFormat("  logical_id: slice %s (%p)",
                                logical_id_buffer_.ToString(),
                                logical_id_data.opaque());

  std::memcpy(logical_id_data.opaque(), &logical_id, sizeof(int32_t));
  return OkExecuteEvent();
}

template <LogicalIdKind type>
using BufferUses = typename LogicalIdThunk<type>::BufferUses;

template <LogicalIdKind type>
BufferUses<type> LogicalIdThunk<type>::buffer_uses() const {
  return {BufferUse::Write(logical_id_buffer_)};
}

template class LogicalIdThunk<LogicalIdKind::kReplicaId>;
template class LogicalIdThunk<LogicalIdKind::kPartitionId>;

}  // namespace xla::cpu

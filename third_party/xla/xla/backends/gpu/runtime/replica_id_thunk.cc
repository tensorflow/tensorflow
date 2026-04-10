/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/replica_id_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

absl::StatusOr<uint32_t> ReplicaOrPartitionIdThunk::ComputeId(
    const ExecuteParams& params) const {
  GlobalDeviceId global_device_id = params.collective_params->global_device_id;
  ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                   params.collective_params->device_assn->LogicalIdForDevice(
                       global_device_id));
  return static_cast<uint32_t>(kind() == Kind::kReplicaId
                                   ? logical_id.replica_id
                                   : logical_id.computation_id);
}

absl::Status ReplicaOrPartitionIdThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);
  ASSIGN_OR_RETURN(uint32_t id, ComputeId(params));
  return params.stream->Memset32(&dest_addr, id, /*size=*/4);
}

absl::StatusOr<const se::CommandBuffer::Command*>
ReplicaOrPartitionIdThunk::Record(const Thunk::ExecuteParams& execute_params,
                                  const RecordParams& record_params,
                                  RecordAction record_action,
                                  se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dest_);
  ASSIGN_OR_RETURN(uint32_t value, ComputeId(execute_params));

  VLOG(5) << (kind() == Kind::kReplicaId ? "Replica" : "Partition")
          << "IdThunk::Record: value=" << value;
  VLOG(5) << "  dest: " << dest_ << " (" << dst.opaque() << ")";

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateMemset(&dst, value, /*num_elements=*/1,
                                        create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(command_buffer->UpdateMemset(update->command, &dst, value,
                                                 /*num_elements=*/1));
    return update->command;
  }
  return Internal("Invalid record action");
}

absl::StatusOr<ThunkProto> ReplicaIdThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  auto* replica_id_thunk_proto = proto.mutable_replica_id_thunk();
  ASSIGN_OR_RETURN(*replica_id_thunk_proto->mutable_dest_buffer(),
                   dest().ToProto());
  return proto;
}

absl::StatusOr<std::unique_ptr<ReplicaIdThunk>> ReplicaIdThunk::FromProto(
    ThunkInfo thunk_info, const ReplicaIdThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  ASSIGN_OR_RETURN(BufferAllocation::Slice dest,
                   BufferAllocation::Slice::FromProto(thunk_proto.dest_buffer(),
                                                      buffer_allocations));
  return std::make_unique<ReplicaIdThunk>(std::move(thunk_info), dest);
}

absl::StatusOr<ThunkProto> PartitionIdThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  auto* partition_id_thunk_proto = proto.mutable_partition_id_thunk();
  ASSIGN_OR_RETURN(*partition_id_thunk_proto->mutable_dest_buffer(),
                   dest().ToProto());
  return proto;
}

/*static*/ absl::StatusOr<std::unique_ptr<PartitionIdThunk>>
PartitionIdThunk::FromProto(ThunkInfo thunk_info,
                            const PartitionIdThunkProto& proto,
                            absl::Span<const BufferAllocation> allocations) {
  ASSIGN_OR_RETURN(
      BufferAllocation::Slice dest_buffer,
      BufferAllocation::Slice::FromProto(proto.dest_buffer(), allocations));
  return std::make_unique<PartitionIdThunk>(std::move(thunk_info), dest_buffer);
}

}  // namespace gpu
}  // namespace xla

/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/nvshmem_recv_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

NvshmemRecvThunk::NvshmemRecvThunk(
    ThunkInfo thunk_info, const HloRecvInstruction* instr,
    int64_t replica_count, int64_t partition_count,
    const CollectiveThunk::Buffer& buffer,
    std::shared_ptr<NvshmemBufferAddresses> buffer_addresses)
    : NvshmemCollectiveThunk(Thunk::kNvshmemRecv, thunk_info,
                             IsGPUSyncCollective(*instr)),
      config_(GetP2PConfigForSendRecv(instr, instr->shape().tuple_shapes(0),
                                      replica_count, partition_count)),
      buffer_(buffer),
      hlo_name_(instr->name()),
      buffer_addresses_(std::move(buffer_addresses)) {}

NvshmemRecvThunk::NvshmemRecvThunk(
    ThunkInfo thunk_info, P2PConfig config,
    const CollectiveThunk::Buffer& buffer,
    std::shared_ptr<NvshmemBufferAddresses> absl_nonnull buffer_addresses,
    std::string hlo_name,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : NvshmemCollectiveThunk(Thunk::kNvshmemRecv, std::move(thunk_info),
                             async_events != nullptr),
      config_(std::move(config)),
      buffer_(buffer),
      hlo_name_(std::move(hlo_name)),
      buffer_addresses_(std::move(buffer_addresses)) {
  set_async_events(std::move(async_events));
}

absl::StatusOr<ThunkProto> NvshmemRecvThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  NvshmemRecvThunkProto* thunk_proto = proto.mutable_nvshmem_recv_thunk();
  *thunk_proto->mutable_config() = P2PConfigToProto(config_);
  TF_ASSIGN_OR_RETURN(*thunk_proto->mutable_buffer(), buffer_.ToProto());
  thunk_proto->set_hlo_name(hlo_name_);

  std::optional<AsyncEventsUniqueId> async_events_id = GetAsyncEventsUniqueId();
  if (async_events_id.has_value()) {
    thunk_proto->set_async_events_unique_id(async_events_id->value());
  }

  return proto;
}

absl::StatusOr<std::unique_ptr<NvshmemRecvThunk>> NvshmemRecvThunk::FromProto(
    ThunkInfo thunk_info, const NvshmemRecvThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    std::shared_ptr<NvshmemBufferAddresses> absl_nonnull buffer_addresses,
    CollectiveThunk::AsyncEventsMap& async_events_map) {
  TF_ASSIGN_OR_RETURN(P2PConfig config,
                      P2PConfigFromProto(thunk_proto.config()));
  TF_ASSIGN_OR_RETURN(CollectiveThunk::Buffer buffer,
                      CollectiveThunk::Buffer::FromProto(thunk_proto.buffer(),
                                                         buffer_allocations));

  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events;
  if (thunk_proto.has_async_events_unique_id()) {
    std::shared_ptr<CollectiveThunk::AsyncEvents>& events =
        async_events_map[AsyncEventsUniqueId{
            thunk_proto.async_events_unique_id()}];
    if (!events) {
      events = std::make_shared<CollectiveThunk::AsyncEvents>();
    }
    async_events = events;
  }

  return absl::WrapUnique(
      new NvshmemRecvThunk(std::move(thunk_info), std::move(config), buffer,
                           std::move(buffer_addresses), thunk_proto.hlo_name(),
                           std::move(async_events)));
}

absl::Status NvshmemRecvThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(NvshmemCollectiveThunk::Initialize(params));
  return absl::OkStatus();
}

absl::Status NvshmemRecvThunk::RunNvshmemCollective(const ExecuteParams& params,
                                                    se::Stream& stream) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations, {buffer_},
                             config_.config.operand_element_type));
  TF_RET_CHECK(device_buffers.size() == 1) << "Expected one buffer pair.";

  GlobalDeviceId global_device_id = params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID current_logical_id,
                      params.collective_params->device_assn->LogicalIdForDevice(
                          global_device_id));
  const int64_t current_id =
      config_.config.group_mode ==
              CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string =
      CollectiveThunk::GetDeviceString(*params.collective_params);

  int device_ordinal = stream.parent()->device_ordinal();
  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);
  DeviceBufferPair& buffer = device_buffers[0];

  VLOG(3) << "Performing Recv from device ordinal: " << device_ordinal
          << ", global_id: " << global_device_id
          << ", current_id: " << current_id << ", group mode: "
          << CollectiveOpGroupModeToString(config_.config.group_mode) << " ("
          << hlo_name_ << ")";

  const std::optional<int64_t> source_id = source_target.source;

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d", device_string,
                                current_id, source_id.value_or(-1));

  // For receive operations, we need to register the buffer with NVSHMEM
  // and store it in the global struct for the send operation to use
  if (source_id.value_or(-1) == -1) {
    VLOG(3) << "Storing destination device " << device_ordinal << " and buffer "
            << buffer.destination_buffer.opaque();
    buffer_addresses_->StoreNvshmemPtr(device_ordinal,
                                       buffer.destination_buffer.opaque());
  }

  if (!source_id) {
    VLOG(3) << "No source ID found, skipping Recv operation";
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(auto* collectives, GetNvshmemCollectivesFromRegistry());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Communicator> nvshmem_comm,
                      collectives->CreateCommunicator());
  VLOG(1) << "Running Recv operation"
          << " element_type=" << buffer.element_type
          << " destination_buffer=" << buffer.destination_buffer.opaque()
          << " source_buffer=" << buffer.source_buffer.opaque()
          << " element_count=" << buffer.element_count
          << " source_id=" << *source_id;
  auto recv_future = nvshmem_comm->Recv(
      buffer.destination_buffer, buffer.source_buffer, buffer.element_type,
      buffer.element_count, RankId(*source_id), GpuCollectives::On(stream));
  TF_RETURN_IF_ERROR(recv_future.Await());
  TF_RETURN_IF_ERROR(nvshmem_comm->Quiet(GpuCollectives::On(stream)));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

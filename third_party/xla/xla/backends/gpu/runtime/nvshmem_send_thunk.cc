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

#include "xla/backends/gpu/runtime/nvshmem_send_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

NvshmemSendThunk::NvshmemSendThunk(
    ThunkInfo thunk_info, const HloSendInstruction* instr,
    int64_t replica_count, int64_t partition_count,
    const CollectiveThunk::Buffer& buffer,
    std::shared_ptr<NvshmemBufferAddresses> buffer_addresses)
    : NvshmemCollectiveThunk(Thunk::kNvshmemSend, thunk_info,
                             /*is_p2p=*/true),
      config_(GetP2PConfigForSendRecv(instr, instr->operand(0)->shape(),
                                      replica_count, partition_count)),
      buffer_(buffer),
      hlo_name_(instr->name()),
      buffer_addresses_(std::move(buffer_addresses)) {}

NvshmemSendThunk::NvshmemSendThunk(
    ThunkInfo thunk_info, const P2PConfig& config,
    const CollectiveThunk::Buffer& buffer, std::string hlo_name,
    std::shared_ptr<NvshmemBufferAddresses> absl_nonnull buffer_addresses)
    : NvshmemCollectiveThunk(Thunk::kNvshmemSend, thunk_info,
                             /*is_p2p=*/true),
      config_(config),
      buffer_(buffer),
      hlo_name_(std::move(hlo_name)),
      buffer_addresses_(std::move(buffer_addresses)) {}

absl::StatusOr<ThunkProto> NvshmemSendThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  NvshmemSendThunkProto* nvshmem_proto = proto.mutable_nvshmem_send_thunk();
  *nvshmem_proto->mutable_p2p_config() = P2PConfigToProto(config_);
  nvshmem_proto->set_hlo_name(hlo_name_);
  TF_ASSIGN_OR_RETURN(*nvshmem_proto->mutable_buffer(), buffer_.ToProto());

  return proto;
}

absl::StatusOr<std::unique_ptr<NvshmemSendThunk>> NvshmemSendThunk::FromProto(
    ThunkInfo thunk_info, const NvshmemSendThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    std::shared_ptr<NvshmemBufferAddresses> absl_nonnull buffer_addresses) {
  TF_RET_CHECK(buffer_addresses != nullptr);
  TF_ASSIGN_OR_RETURN(P2PConfig p2p_config,
                      P2PConfigFromProto(proto.p2p_config()));

  TF_ASSIGN_OR_RETURN(
      CollectiveThunk::Buffer buffer,
      CollectiveThunk::Buffer::FromProto(proto.buffer(), buffer_allocations));

  return absl::WrapUnique(
      new NvshmemSendThunk(std::move(thunk_info), p2p_config, buffer,
                           proto.hlo_name(), std::move(buffer_addresses)));
}

absl::Status NvshmemSendThunk::Initialize(const InitializeParams& params) {
  VLOG(3) << "Initializing NvshmemSendThunk for: " << hlo_name_;
  TF_RETURN_IF_ERROR(NvshmemCollectiveThunk::Initialize(params));
  return absl::OkStatus();
}

absl::Status NvshmemSendThunk::RunNvshmemCollective(const ExecuteParams& params,
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

  // Determine the target IDs for this instance. The target ID is the ID
  // to which this instance will copy its data.
  VLOG(3) << "Performing Send from device ordinal: " << device_ordinal
          << ", global_id: " << global_device_id
          << ", current_id: " << current_id << ", group mode: "
          << CollectiveOpGroupModeToString(config_.config.group_mode) << " ("
          << hlo_name_ << ")";

  const std::optional<int64_t> target_id = source_target.target;

  VLOG(3) << absl::StreamFormat("%s : id = %d, target_id = %d", device_string,
                                current_id, target_id.value_or(-1));

  // Set up receive buffer if available (this should happen for all ranks)
  auto recv_buffer_status = buffer_addresses_->GetNvshmemPtr(device_ordinal);
  if (recv_buffer_status.ok()) {
    void* recv_buffer_ptr = recv_buffer_status.value();
    VLOG(3) << "Using existing receive buffer for send: " << recv_buffer_ptr;
    buffer.destination_buffer = se::DeviceAddressBase(
        recv_buffer_ptr, buffer.destination_buffer.size());
  } else {
    VLOG(3) << "No receive buffer found";
  }

  // Only proceed with Send operation if we have a target
  if (!target_id) {
    VLOG(3) << "No target ID found, skipping Send operation";
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(auto* collectives, GetNvshmemCollectivesFromRegistry());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Communicator> nvshmem_comm,
                      collectives->CreateCommunicator());
  VLOG(1) << "Running Send operation"
          << " element_type=" << buffer.element_type
          << " destination_buffer=" << buffer.destination_buffer.opaque()
          << " source_buffer=" << buffer.source_buffer.opaque()
          << " element_count=" << buffer.element_count
          << " target_id=" << *target_id;
  auto send_future = nvshmem_comm->Send(
      buffer.destination_buffer, buffer.source_buffer, buffer.element_type,
      buffer.element_count, RankId(*target_id), GpuCollectives::On(stream));
  TF_RETURN_IF_ERROR(send_future.Await());
  TF_RETURN_IF_ERROR(nvshmem_comm->Quiet(GpuCollectives::On(stream)));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/send_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

SendThunk::SendThunk(ThunkInfo thunk_info, const HloSendInstruction* instr,
                     int64_t replica_count, int64_t partition_count,
                     const Buffer& buffer)
    : SendThunk(std::move(thunk_info),
                GetP2PConfigForSendRecv(instr, instr->operand(0)->shape(),
                                        replica_count, partition_count),
                buffer, instr->name()) {}

SendThunk::SendThunk(ThunkInfo thunk_info, const P2PConfig& config,
                     const Buffer& buffer, absl::string_view instr_name)
    : CollectiveThunk(Thunk::kSend, thunk_info, {buffer}, CommunicationId(1)),
      config_(config),
      hlo_name_(instr_name) {}

absl::Status SendThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<SendThunk>> SendThunk::FromProto(
    ThunkInfo thunk_info, const SendThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  ASSIGN_OR_RETURN(CollectiveThunk::Buffer buffer,
                   CollectiveThunk::Buffer::FromProto(thunk_proto.buffer(),
                                                      buffer_allocations));

  CollectiveConfig config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  P2PConfig::IdToSourceTargetMap id_to_source_target;
  for (const SourceTarget& source_target : thunk_proto.source_target_pairs()) {
    id_to_source_target.insert({source_target.target(), {}})
        .first->second.source = source_target.source();
    id_to_source_target.insert({source_target.source(), {}})
        .first->second.target = source_target.target();
  }

  return std::make_unique<SendThunk>(
      std::move(thunk_info), P2PConfig{config, std::move(id_to_source_target)},
      buffer, thunk_proto.instruction_name());
}

absl::StatusOr<ThunkProto> SendThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  SendThunkProto* thunk_proto = proto.mutable_send_thunk();

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  ASSIGN_OR_RETURN(*thunk_proto->mutable_buffer(), buffer().ToProto());
  std::vector<SourceTarget> source_target_pairs;
  source_target_pairs.reserve(config_.id_to_source_target.size() / 2);
  for (const auto& [key_id, map_entry] : config_.id_to_source_target) {
    if (!map_entry.source.has_value()) {
      // Same pair is in the map with target/source switched.
      continue;
    }
    SourceTarget pair;
    pair.set_source(*map_entry.source);
    pair.set_target(key_id);
    source_target_pairs.push_back(pair);
  }
  thunk_proto->mutable_source_target_pairs()->Assign(
      source_target_pairs.begin(), source_target_pairs.end());

  thunk_proto->set_instruction_name(hlo_name_);
  return proto;
}

absl::Status RunSend(DeviceBufferPair& buffer, se::Stream& stream,
                     Communicator& comm, int64_t current_id, int64_t target_id,
                     absl::string_view device_string) {
  // Determine the target IDs for this instance. The target ID is the ID
  // to which this instance will copy its data.
  int device_ordinal = stream.parent()->device_ordinal();
  se::DeviceAddressBase src_addr = buffer.source_buffer;

  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "%s : id = %d, target_id = %d", device_string, current_id, target_id);

  // Send source buffer to target peer if needed.
  XLA_VLOG_DEVICE(3, device_ordinal)
      << absl::StreamFormat("target_id = %d, call comm.Send()", target_id);
  auto future = comm.Send(src_addr, buffer.element_type, buffer.element_count,
                          RankId(target_id), GpuCollectives::On(stream));
  TF_RETURN_IF_ERROR(future.Await());
  return absl::OkStatus();
}

absl::Status SendThunk::RunCollective(const ExecuteParams& params,
                                      const GpuCliqueKey&, se::Stream& stream,
                                      Communicator& comm) {
  auto send_buffer = buffers()[0];
  DeviceBufferPair device_buffer_pair{
      config_.config.operand_element_type[0],
      send_buffer.element_count,
      params.buffer_allocations->GetDeviceAddress(
          send_buffer.source_buffer.slice),
      params.buffer_allocations->GetDeviceAddress(
          send_buffer.destination_buffer.slice),
      send_buffer.source_memory_space,
      send_buffer.destination_memory_space};

  GlobalDeviceId global_device_id = params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID current_logical_id,
                      params.collective_params->device_assn->LogicalIdForDevice(
                          global_device_id));
  const int64_t current_id =
      config_.config.group_mode ==
              CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string = GetDeviceString(*params.collective_params);

  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);

  int device_ordinal = stream.parent()->device_ordinal();

  std::optional<int64_t> target_id = source_target.target;

  if (!target_id) {
    XLA_VLOG_DEVICE(3, device_ordinal)
        << absl::StreamFormat("%s : Skipping Send", device_string);
    return absl::OkStatus();
  }

  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Performing Send "
      << ", current_id: " << current_id << ", group mode: "
      << CollectiveOpGroupModeToString(config_.config.group_mode)
      << ", hlo_name=(" << hlo_name_ << ")";

  return RunSend(device_buffer_pair, stream, comm, current_id, *target_id,
                 device_string);
}

}  // namespace gpu
}  // namespace xla

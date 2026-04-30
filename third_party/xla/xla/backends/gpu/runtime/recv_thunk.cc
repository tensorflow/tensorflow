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

#include "xla/backends/gpu/runtime/recv_thunk.h"

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
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

RecvThunk::RecvThunk(ThunkInfo thunk_info, const HloRecvInstruction* instr,
                     int64_t replica_count, int64_t partition_count,
                     const Buffer& buffer)
    : RecvThunk(std::move(thunk_info),
                GetP2PConfigForSendRecv(instr, instr->shape().tuple_shapes(0),
                                        replica_count, partition_count),
                buffer, instr->name()) {}

RecvThunk::RecvThunk(ThunkInfo thunk_info, const P2PConfig& config,
                     const Buffer& buffer, absl::string_view instr_name)
    : CollectiveThunk(Thunk::kRecv, thunk_info, {buffer}, CommunicationId(1)),
      config_(config),
      hlo_name_(instr_name) {}

absl::Status RecvThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<RecvThunk>> RecvThunk::FromProto(
    ThunkInfo thunk_info, const RecvThunkProto& thunk_proto,
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

  return std::make_unique<RecvThunk>(
      std::move(thunk_info), P2PConfig{config, std::move(id_to_source_target)},
      buffer, thunk_proto.instruction_name());
}

absl::StatusOr<ThunkProto> RecvThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  RecvThunkProto* thunk_proto = proto.mutable_recv_thunk();

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

absl::Status RunRecv(DeviceBufferPair& buffer, se::Stream& stream,
                     Communicator& comm, int64_t current_id,
                     std::optional<int64_t> source_id,
                     absl::string_view device_string) {
  // Determine the target IDs for this instance. The target ID is the ID
  // to which this instance will copy its data.
  int device_ordinal = stream.parent()->device_ordinal();
  se::DeviceAddressBase dest_addr = buffer.destination_buffer;

  XLA_VLOG_DEVICE(3, device_ordinal)
      << absl::StreamFormat("%s : id = %d, source_id = %d", device_string,
                            current_id, source_id.value_or(-1));

  // Receive data from the source peer to the destination buffer.
  if (source_id) {
    XLA_VLOG_DEVICE(3, device_ordinal)
        << absl::StreamFormat("source_id: %d, call comm.Recv()", *source_id);
    auto future =
        comm.Recv(dest_addr, buffer.element_type, buffer.element_count,
                  RankId(*source_id), GpuCollectives::On(stream));
    TF_RETURN_IF_ERROR(future.Await());
  } else {
    // If there is no source peer, i.e. no sender to this instance, zero out
    // the destination buffer.
    XLA_VLOG_DEVICE(3, device_ordinal)
        << absl::StreamFormat("%s : Recv: Issuing MemZero", device_string);
    TF_RETURN_IF_ERROR(stream.MemZero(&dest_addr, dest_addr.size()));
  }

  return absl::OkStatus();
}

absl::Status RecvThunk::RunCollective(const ExecuteParams& params,
                                      const GpuCliqueKey& clique_key,
                                      se::Stream& stream, Communicator& comm) {
  auto recv_buffer = buffers()[0];
  DeviceBufferPair device_buffer_pair{
      config_.config.operand_element_type[0],
      recv_buffer.element_count,
      params.buffer_allocations->GetDeviceAddress(
          recv_buffer.source_buffer.slice),
      params.buffer_allocations->GetDeviceAddress(
          recv_buffer.destination_buffer.slice),
      recv_buffer.source_memory_space,
      recv_buffer.destination_memory_space};

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

  // Determine the source IDs for this instance. The source ID is the ID for
  // the peer that will copy its data to this instance. If there is no
  // source, just memzero() the destination buffer.
  int device_ordinal = stream.parent()->device_ordinal();

  const std::optional<int64_t> source_id = source_target.source;

  XLA_VLOG_DEVICE(3, device_ordinal) << absl::StreamFormat(
      "Performing Recv, current_id: %d, group mode: %s, hlo_name=(%s)",
      current_id, CollectiveOpGroupModeToString(config_.config.group_mode),
      hlo_name_);

  return RunRecv(device_buffer_pair, stream, comm, current_id, source_id,
                 device_string);
}

}  // namespace gpu
}  // namespace xla

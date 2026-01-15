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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
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
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

SendThunk::SendThunk(ThunkInfo thunk_info, const HloSendInstruction* instr,
                     int64_t replica_count, int64_t partition_count,
                     const Buffer& buffer)
    : SendThunk(std::move(thunk_info),
                GetP2PConfigForSendRecv(instr, instr->operand(0)->shape(),
                                        replica_count, partition_count),
                std::make_shared<CollectiveThunk::AsyncEvents>(), buffer,
                instr->name()) {}

SendThunk::SendThunk(ThunkInfo thunk_info, const P2PConfig& config,
                     std::shared_ptr<AsyncEvents> async_events,
                     const Buffer& buffer, absl::string_view instr_name)
    : CollectiveThunk(Thunk::kSend, thunk_info, async_events, true),
      config_(config),
      buffer_(buffer),
      execution_counters_(config_.validation_kind ==
                                  P2PConfig::ValidationKind::kConditional
                              ? std::make_shared<ExecutionCounters>()
                              : nullptr),
      hlo_name_(instr_name) {}

absl::Status SendThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  if (execution_counters_) {
    TF_RETURN_IF_ERROR(execution_counters_->Initialize(
        params.executor, params.collective_params->run_id));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> SendThunk::ConditionalShouldRun(
    const ExecuteParams& params, int64_t current_id, int64_t target_id) const {
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(int64_t* counter,
                      execution_counters_->GetCounter(
                          executor, params.collective_params->run_id));
  auto it = config_.source_target_to_bounds.find(
      std::make_pair(current_id, target_id));
  TF_RET_CHECK(it != config_.source_target_to_bounds.end())
      << "Missing bounds for conditional Send";
  bool should_run =
      !(*counter < it->second.first || *counter > it->second.second);
  VLOG(3) << "RunCollective counter " << *counter << " " << should_run;
  ++(*counter);
  return should_run;
}

absl::StatusOr<std::unique_ptr<SendThunk>> SendThunk::FromProto(
    ThunkInfo thunk_info, const SendThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    CollectiveThunk::AsyncEventsMap& async_events_map) {
  std::shared_ptr<CollectiveThunk::AsyncEvents>& async_events =
      async_events_map[AsyncEventsUniqueId{
          thunk_proto.async_events_unique_id()}];
  if (!async_events) {
    async_events = std::make_shared<CollectiveThunk::AsyncEvents>();
  }

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
      async_events, buffer, thunk_proto.instruction_name());
}

absl::StatusOr<ThunkProto> SendThunk::ToProto() const {
  CHECK_EQ(config_.validation_kind, P2PConfig::ValidationKind::kValid);
  CHECK(config_.source_target_to_bounds.empty());

  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  SendThunkProto* thunk_proto = proto.mutable_send_thunk();

  std::optional<AsyncEventsUniqueId> async_events_id = GetAsyncEventsUniqueId();
  CHECK(async_events_id.has_value());
  thunk_proto->set_async_events_unique_id(async_events_id->value());

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
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

  VLOG(3) << absl::StreamFormat("[%d] %s : id = %d, target_id = %d",
                                device_ordinal, device_string, current_id,
                                target_id);

  // Send source buffer to target peer if needed.
  VLOG(3) << "[" << device_ordinal << "] target_id: " << target_id
          << ", call comm.Send()";
  TF_RETURN_IF_ERROR(MaybeRegisterBuffers(stream.parent(), {buffer}, &comm));
  auto future = comm.Send(src_addr, buffer.element_type, buffer.element_count,
                          RankId(target_id), GpuCollectives::On(stream));
  TF_RETURN_IF_ERROR(future.Await());
  return absl::OkStatus();
}

absl::StatusOr<bool> SendThunk::RunCollective(const ExecuteParams& params,
                                              const GpuCliqueKey&,
                                              se::Stream& stream,
                                              Communicator& comm) {
  DeviceBufferPair device_buffer_pair{
      config_.config.operand_element_type[0],
      buffer_.element_count,
      params.buffer_allocations->GetDeviceAddress(buffer_.source_buffer.slice),
      params.buffer_allocations->GetDeviceAddress(
          buffer_.destination_buffer.slice),
      buffer_.source_memory_space,
      buffer_.destination_memory_space};

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
  bool should_run = false;

  switch (config_.validation_kind) {
    case P2PConfig::ValidationKind::kValid:
      should_run = true;
      break;
    case P2PConfig::ValidationKind::kInvalid:
      should_run = false;
      break;
    case P2PConfig::ValidationKind::kConditional:
      if (target_id) {
        TF_ASSIGN_OR_RETURN(
            should_run, ConditionalShouldRun(params, current_id, *target_id));
      }
      break;
  }

  if (!target_id || !should_run) {
    VLOG(3) << "[" << device_ordinal << "] Skipping Send";
    return false;
  }

  VLOG(3) << "[" << device_ordinal << "] Performing Send "
          << ", current_id: " << current_id << ", group mode: "
          << CollectiveOpGroupModeToString(config_.config.group_mode)
          << ", hlo_name=(" << hlo_name_ << ")";

  TF_RETURN_IF_ERROR(RunSend(device_buffer_pair, stream, comm, current_id,
                             *target_id, device_string));
  return false;
}

}  // namespace gpu
}  // namespace xla

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

#include "xla/backends/gpu/runtime/collective_thunk.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/rendezvous.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int64_t kCollectiveMemorySpaceColor = 1;

bool IsTypeSupportedBy(PrimitiveType element_type, Thunk::Kind reduction_op) {
  switch (element_type) {
    case S8:
    case PRED:
    case U8:
    case S32:
    case U32:
    case S64:
    case U64:
    case F8E5M2:
    case F8E4M3FN:
    case F16:
    case F32:
    case F64:
    case BF16:
    case C64:
    case C128:
      return true;
    case S16:
    case U16:
      // 16-bit integer reductions are not directly supported by NCCL and cannot
      // be implicitly converted into other 16-bit types like ncclFloat16 as
      // they involve actual computation and not just data movement.
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      return !IsReductionCollective(reduction_op);
    default:
      return false;
  }
}

}  // namespace

// Returns if the collective communication operation is degenerate because all
// the groups formed by the operation are singleton. A given op can be
// degenerate under several conditions, corresponding to the modes supported
// in GetParticipatingDevices().
//   1. no channel id, use_global_device_ids = false:
//         degenerate if replica_groups are singleton, or groups empty and
//         replica_count == 1.
//   2. channel_id is set, use_global_device_ids = false:
//         degenerate if replica_groups are singleton and num_partitions == 1,
//         or groups empty and num_replicas == 1 && num_partitions == 1.
//   3. channel_id is set, use_global_device_ids = true (flattened-ids):
//         degenerate if replica_groups are singleton (groups cannot be empty).
//   4. no channel_id, no use_global_device_ids:
//         identical to 1.
//   5. channel_id is set, no use_global_device_ids:
//         degenerate if replica_groups are singleton or group empty and
//         num_partitions == 1 (since replica groups contain partition ids).
//
bool CollectiveConfig::IsDegenerate(int64_t replica_count,
                                    int64_t partition_count) const {
  bool groups_empty = replica_groups.empty();

  // check if all replica_groups are singleton. If not, then the operation is
  // not degenerate.
  bool all_groups_singleton =
      !groups_empty &&
      absl::c_all_of(replica_groups, [](const ReplicaGroup& group) {
        return group.replica_ids_size() == 1;
      });

  switch (group_mode) {
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA:
      return all_groups_singleton || (groups_empty && replica_count == 1);
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION:
      return all_groups_singleton || (groups_empty && partition_count == 1);
    case CollectiveOpGroupMode::
        COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION:
      return (all_groups_singleton && partition_count == 1) ||
             (groups_empty && replica_count == 1 && partition_count == 1);
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID:
      CHECK(!groups_empty)
          << "replica groups cannot be empty if use_global_device_ids = true";
      return all_groups_singleton;
    default:
      CHECK(0) << "Invalid collective op mode";
      return false;
  }
}

CollectiveConfigProto CollectiveConfig::ToProto() const {
  CollectiveConfigProto proto;

  proto.mutable_operand_element_type()->Assign(operand_element_type.begin(),
                                               operand_element_type.end());
  proto.mutable_replica_groups()->Assign(replica_groups.begin(),
                                         replica_groups.end());

  proto.set_group_mode(group_mode);
  proto.set_use_symmetric_buffer(use_symmetric_buffer);

  return proto;
}

CollectiveConfig CollectiveConfig::FromProto(
    const CollectiveConfigProto& proto) {
  CollectiveConfig config;

  config.operand_element_type.reserve(proto.operand_element_type_size());
  for (int element_type : proto.operand_element_type()) {
    config.operand_element_type.push_back(
        static_cast<PrimitiveType>(element_type));
  }

  config.replica_groups.assign(proto.replica_groups().begin(),
                               proto.replica_groups().end());

  config.group_mode = proto.group_mode();
  config.use_symmetric_buffer = proto.use_symmetric_buffer();
  return config;
}

CollectiveConfig GetCollectiveConfig(
    const HloInstruction* hlo, std::optional<bool> use_global_device_ids) {
  CollectiveConfig config;
  config.operand_element_type.reserve(hlo->operands().size());
  for (const HloInstruction* operand : hlo->operands()) {
    config.operand_element_type.push_back(operand->shape().element_type());
  }
  config.replica_groups = hlo->replica_groups();

  config.group_mode = GetCollectiveOpGroupMode(hlo->channel_id().has_value(),
                                               use_global_device_ids)
                          .value();

  config.use_symmetric_buffer =
      hlo->GetModule() &&
      hlo->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_enable_nccl_symmetric_buffers();
  return config;
}

CollectiveThunk::CollectiveThunk(Kind kind, ThunkInfo thunk_info,
                                 std::vector<Buffer> buffers,
                                 CommunicationId communication_id,
                                 CollectivesMode collectives_mode)
    : Command(CommandType::kCollectiveCmd, kind, std::move(thunk_info)),
      buffers_(std::move(buffers)),
      communication_id_(communication_id),
      collectives_mode_(collectives_mode) {}

bool CollectiveThunk::use_private_memory() const {
  return collectives_mode_ == DebugOptions::COLLECTIVES_PRIVATE_MEMORY ||
         collectives_mode_ == DebugOptions::COLLECTIVES_MODE_INVALID;
}
bool CollectiveThunk::use_symmetric_memory() const {
  return collectives_mode_ == DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY;
}
bool CollectiveThunk::use_peer_memory() const {
  return collectives_mode_ == DebugOptions::COLLECTIVES_PEER_MEMORY;
}

absl::StatusOr<GpuCliqueKey> GetCollectiveGpuCliqueKey(
    const CollectiveParams& params, const CollectiveConfig& collective_config,
    CommunicationId communication_id) {
  return GetGpuCliqueKey(params, collective_config.replica_groups,
                         collective_config.group_mode, communication_id);
}

absl::StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const BufferAllocations* buffer_allocations,
    const std::vector<CollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types) {
  if (buffers.size() != element_types.size()) {
    return FailedPrecondition("Mismatch in operand buffer counts.");
  }

  std::vector<DeviceBufferPair> device_buffers;
  device_buffers.reserve(buffers.size());
  for (int i = 0; i < buffers.size(); ++i) {
    device_buffers.emplace_back(DeviceBufferPair{
        element_types[i], buffers[i].element_count,
        buffer_allocations->GetDeviceAddress(buffers[i].source_buffer.slice),
        buffer_allocations->GetDeviceAddress(
            buffers[i].destination_buffer.slice),
        buffers[i].source_memory_space, buffers[i].destination_memory_space});
  }
  return device_buffers;
}

absl::StatusOr<CollectiveBufferProto> CollectiveThunk::Buffer::ToProto() const {
  CollectiveBufferProto proto;
  proto.set_element_count(element_count);
  ASSIGN_OR_RETURN(*proto.mutable_source_buffer(), source_buffer.ToProto());
  ASSIGN_OR_RETURN(*proto.mutable_destination_buffer(),
                   destination_buffer.ToProto());
  proto.set_source_memory_space(source_memory_space);
  proto.set_destination_memory_space(destination_memory_space);
  return proto;
}

absl::StatusOr<CollectiveThunk::Buffer> CollectiveThunk::Buffer::FromProto(
    const CollectiveBufferProto& buffer_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  CollectiveThunk::Buffer res;
  res.element_count = buffer_proto.element_count();
  ASSIGN_OR_RETURN(
      res.source_buffer,
      ShapedSlice::FromProto(buffer_proto.source_buffer(), buffer_allocations));
  ASSIGN_OR_RETURN(res.destination_buffer,
                   ShapedSlice::FromProto(buffer_proto.destination_buffer(),
                                          buffer_allocations));
  res.source_memory_space = buffer_proto.source_memory_space();
  res.destination_memory_space = buffer_proto.destination_memory_space();
  return res;
}

absl::Status CollectiveThunk::Prepare(const PrepareParams& params) {
  TF_RET_CHECK(params.collective_params &&
               params.collective_params->device_assn)
      << "Collective parameters and device assignment are required for "
         "collective thunk execution";

  // Device groups depend only on device assignment and replica groups, both
  // constant for the thunk's lifetime (device assignment is owned by
  // PjRtExecutable and never changes between executions). Compute once.
  absl::call_once(device_groups_once_, [&] {
    device_groups_ = GetParticipatingDevicesGroups(
        *params.collective_params->device_assn, config().replica_groups,
        config().group_mode);
    if (device_groups_.ok()) {
      absl::c_for_each(*device_groups_, [](auto& g) { absl::c_sort(g); });
      absl::c_sort(*device_groups_);
    }
  });

  RETURN_IF_ERROR(device_groups_.status());

  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, communication_id_));

  RETURN_IF_ERROR(params.collective_clique_requests->RequestClique(
      clique_key, *device_groups_, GetCliqueRequirements(clique_key)));

  if (CanUseSymmetricBuffer() && config().use_symmetric_buffer) {
    for (const Buffer& buffer : buffers_) {
      if (buffer.source_memory_space == kCollectiveMemorySpaceColor) {
        TF_RETURN_IF_ERROR(
            params.collective_memory_requests->RequestSymmetricAllocation(
                clique_key, buffer.source_buffer.slice.index()));
      }

      if (buffer.destination_memory_space == kCollectiveMemorySpaceColor) {
        TF_RETURN_IF_ERROR(
            params.collective_memory_requests->RequestSymmetricAllocation(
                clique_key, buffer.destination_buffer.slice.index()));
      }
    }
  }

  return PrepareCollective(params, clique_key);
}

absl::Status CollectiveThunk::Initialize(const InitializeParams& params) {
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, communication_id_));
  return InitializeCollective(params, clique_key);
}

absl::Status CollectiveThunk::FirstCallRendezvous(
    const ExecuteParams& params, const GpuCliqueKey& clique_key,
    absl::string_view label, RendezvousFlag& flag) {
  if (!RequiresRendezvous() || flag.IsCompleted()) {
    return absl::OkStatus();
  }

  int32_t device_ordinal = params.stream->parent()->device_ordinal();
  size_t num_local_participants = clique_key.num_local_participants();
  auto global_device_id = params.collective_params->global_device_id;
  RankId rank = clique_key.rank(global_device_id).value_or(RankId(-1));

  XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
      "First call rendezvous %s %v; run_id=%d; num_local_participants=%d; "
      "rank=%d; clique_key=%s",
      label, kind(), params.collective_params->run_id.ToInt(),
      num_local_participants, rank.value(), clique_key.ToString());

  auto rendezvous_key = FirstCallRendezvousKey{clique_key};
  auto rendezvous_name = absl::StrFormat(
      "[%d] first call %s collective operation: kind=%v; run_id=%ld",
      device_ordinal, label, kind(), params.collective_params->run_id.ToInt());

  const xla::DebugOptions debug_options = xla::GetDebugOptionsFromFlags();

  return Rendezvous(
      flag, rendezvous_name, rendezvous_key, num_local_participants,
      /*warn_stuck_timeout=*/
      absl::Seconds(
          debug_options
              .xla_gpu_first_collective_call_warn_stuck_timeout_seconds()),
      /*terminate_timeout=*/
      absl::Seconds(
          debug_options
              .xla_gpu_first_collective_call_terminate_timeout_seconds()));
}

absl::Status CollectiveThunk::RunWithCommAndRendezvous(
    const ExecuteParams& params,
    absl::FunctionRef<absl::Status(const GpuCliqueKey&, Communicator&)> fn) {
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, communication_id_));
  ASSIGN_OR_RETURN(Communicator * comm,
                   params.collective_cliques->GetComm(
                       clique_key, params.collective_params->global_device_id));
  DCHECK(comm) << "Failed to get communicator for collective operation";

  std::pair<RendezvousFlag*, RendezvousFlag*> rend_flags;
  ASSIGN_OR_RETURN(rend_flags,
                   params.collective_cliques->GetCliqueFirstRendezvousFlags(
                       clique_key, params.module_name));

  RETURN_IF_ERROR(
      FirstCallRendezvous(params, clique_key, "before", *(rend_flags.first)));
  RETURN_IF_ERROR(fn(clique_key, *comm));
  RETURN_IF_ERROR(
      FirstCallRendezvous(params, clique_key, "after", *(rend_flags.second)));
  return absl::OkStatus();
}

absl::Status CollectiveThunk::ExecuteOnStream(const ExecuteParams& params) {
  XLA_VLOG_DEVICE(1, params.stream->parent()->device_ordinal())
      << absl::StreamFormat("Starting %v.", kind());

  return RunWithCommAndRendezvous(
      params, [&](const GpuCliqueKey& clique_key, Communicator& comm) {
        return RunCollective(params, clique_key, *params.stream, comm);
      });
}

absl::StatusOr<const se::CommandBuffer::Command*> CollectiveThunk::Record(
    const ExecuteParams& execute_params, const RecordParams& record_params,
    RecordAction record_action, se::CommandBuffer* command_buffer) {
  std::unique_ptr<se::CommandBuffer> nested_cmd;
  RETURN_IF_ERROR(RunWithCommAndRendezvous(
      execute_params,
      [&](const GpuCliqueKey& clique_key, Communicator& comm) -> absl::Status {
        ASSIGN_OR_RETURN(nested_cmd,
                         se::TraceCommandBufferFactory::Create(
                             execute_params.stream->parent(),
                             execute_params.command_buffer_trace_stream,
                             [&](se::Stream* stream) {
                               return RunCollective(execute_params, clique_key,
                                                    *stream, comm);
                             }));
        return absl::OkStatus();
      }));

  RETURN_IF_ERROR(nested_cmd->SetPriority(se::StreamPriority::Highest));

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateChildCommand(*nested_cmd,
                                              create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(
        command_buffer->UpdateChildCommand(update->command, *nested_cmd));
    return update->command;
  }
  return Internal("Invalid record action");
}

absl::StatusOr<std::vector<Communicator*>> CollectiveThunk::GetCommunicators(
    const ExecuteParams& params) const {
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, communication_id_));
  ASSIGN_OR_RETURN(Communicator * comm,
                   params.collective_cliques->GetComm(
                       clique_key, params.collective_params->global_device_id));
  return std::vector<Communicator*>{comm};
}

Thunk::BufferUses CollectiveThunk::buffer_uses() const {
  BufferUses uses;
  uses.reserve(buffers_.size() * 2);
  for (const Buffer& buffer : buffers_) {
    uses.push_back(BufferUse::Read(buffer.source_buffer.slice,
                                   buffer.source_buffer.shape));
    uses.push_back(BufferUse::Write(buffer.destination_buffer.slice,
                                    buffer.destination_buffer.shape));
  }
  return uses;
}

std::string CollectiveThunk::GetDeviceString(
    const CollectiveParams& collective_params) {
  GlobalDeviceId global_device_id = collective_params.global_device_id;
  DeviceAssignment::LogicalID logical_id =
      collective_params.device_assn->LogicalIdForDevice(global_device_id)
          .value();
  return absl::StrFormat("(r%d, p%d) : GlobalID %d, ord %d",
                         logical_id.replica_id, logical_id.computation_id,
                         global_device_id.value(),
                         collective_params.local_device_id.value());
}

absl::Status IsValidOperand(Shape shape, Thunk::Kind reduction_op) {
  if (!shape.IsArray()) {
    return absl::AbortedError(
        absl::StrFormat("input is not a dense array: %s",
                        shape.ToString(/*print_layout=*/true)));
  }
  if (!IsTypeSupportedBy(shape.element_type(), reduction_op)) {
    return absl::AbortedError(absl::StrFormat(
        "element type %s not supported by NCCL",
        primitive_util::LowercasePrimitiveTypeName(shape.element_type())));
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu

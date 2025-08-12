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

#include "xla/backends/gpu/runtime/nvshmem_collective_permute_thunk.h"

#include <unistd.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

absl::StatusOr<const int64_t> GetCurrentId(
    Thunk::CollectiveExecuteParams* collective_params,
    const P2PConfig& config) {
  GlobalDeviceId global_device_id = collective_params->global_device_id;
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID current_logical_id,
      collective_params->device_assn->LogicalIdForDevice(global_device_id));
  const int64_t current_id =
      config.config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  return current_id;
}

}  // namespace

NvshmemCollectivePermuteStartThunk::NvshmemCollectivePermuteStartThunk(
    ThunkInfo thunk_info, const HloCollectivePermuteInstruction* instr,
    int64_t replica_count, int64_t partition_count,
    const std::vector<CollectiveThunk::Buffer>& buffers,
    bool p2p_memcpy_enabled, AsyncStreamKind stream_kind)
    : NvshmemCollectiveThunk(Thunk::kNvshmemCollectivePermuteStart, thunk_info,
                             IsGPUSyncCollective(*instr)),
      config_(GetNvshmemP2PConfig(instr, replica_count, partition_count)),
      buffers_(buffers),
      p2p_memcpy_enabled_(p2p_memcpy_enabled) {}

/*static*/ P2PConfig NvshmemCollectivePermuteStartThunk::GetNvshmemP2PConfig(
    const HloCollectivePermuteInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  P2PConfig collective_permute_config;
  auto& config = collective_permute_config.config;

  config.operand_count = instr->operand_count();
  for (const HloInstruction* operand : instr->operands()) {
    config.operand_element_type.push_back(operand->shape().element_type());
  }
  config.SetCollectiveOpKindAndID(instr);
  config.group_mode = GetGroupMode(instr);

  // With a collective permute, all execution instances together form one
  // replica group.
  const int64_t num_participants =
      config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? replica_count
          : partition_count;
  config.replica_groups.emplace_back();
  ReplicaGroup& replica_group = config.replica_groups.front();
  for (int i = 0; i < num_participants; ++i) {
    replica_group.add_replica_ids(i);
  }

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs =
      instr->source_target_pairs();

  for (const std::pair<int64_t, int64_t>& source_target : source_target_pairs) {
    int64_t source = source_target.first;
    int64_t target = source_target.second;
    VLOG(1) << "CollectivePermute: Adding source-target pair: " << source
            << " -> " << target;
    collective_permute_config.id_to_source_target.insert({target, {}})
        .first->second.source = source;
    collective_permute_config.id_to_source_target.insert({source, {}})
        .first->second.target = target;
  }

  return collective_permute_config;
}

/*static*/ CollectiveOpGroupMode
NvshmemCollectivePermuteStartThunk::GetGroupMode(
    const HloCollectivePermuteInstruction* instr) {
  return GetCollectiveOpGroupMode(instr->channel_id().has_value(), std::nullopt)
      .value();
}

absl::Status NvshmemCollectivePermuteStartThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(NvshmemCollectiveThunk::Initialize(params));

  if (p2p_memcpy_enabled_) {
    return absl::InvalidArgumentError(
        "p2p_memcpy_enabled_ is not supported in NVSHMEM collective permute");
  }
  return absl::OkStatus();
}

absl::Status NvshmemCollectivePermuteStartThunk::RunNvshmemCollective(
    const ExecuteParams& params, se::Stream& stream) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params,
                             std::vector<CollectiveThunk::Buffer>(buffers_),
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(const int64_t current_id,
                      GetCurrentId(params.collective_params, config_));
  std::string device_string =
      CollectiveThunk::GetDeviceString(*params.collective_params);

  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);

  return ::xla::gpu::RunCollectivePermute(source_target, device_buffers, stream,
                                          device_string, current_id);
}

absl::Status RunCollectivePermute(P2PConfig::SourceTargetMapEntry source_target,
                                  std::vector<DeviceBufferPair>& buffers,
                                  se::Stream& stream,
                                  absl::string_view device_string,
                                  int64_t current_id) {
  TF_ASSIGN_OR_RETURN(auto* collectives, GetNvshmemCollectivesFromRegistry());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Communicator> nvshmem_comm,
                      collectives->CreateCommunicator());

  int device_ordinal = stream.parent()->device_ordinal();

  VLOG(3) << "Performing collective permute from device ordinal: "
          << device_ordinal << " current_id " << current_id;

  std::optional<int64_t> source_id = source_target.source;
  std::optional<int64_t> target_id = source_target.target;

  std::vector<se::DeviceMemoryBase> src_addrs, dest_addrs;
  absl::c_transform(
      buffers, std::back_inserter(src_addrs),
      [](const DeviceBufferPair& buffer) { return buffer.source_buffer; });
  absl::c_transform(
      buffers, std::back_inserter(dest_addrs),
      [](const DeviceBufferPair& buffer) { return buffer.destination_buffer; });

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d, target_id = %d",
                                device_string, current_id,
                                source_id.value_or(-1), target_id.value_or(-1));

  for (uint64_t idx = 0; idx < buffers.size(); ++idx) {
    const auto src_addr = src_addrs.at(idx);
    const auto dest_addr = dest_addrs.at(idx);
    const auto buffer = buffers.at(idx);

    if (target_id) {
      VLOG(1) << "CollectivePermute: rank " << device_ordinal
              << " sending data to target " << *target_id;

      auto send_event = nvshmem_comm->Send(
          dest_addr, src_addr, buffer.element_type, buffer.element_count,
          RankId(*target_id), GpuCollectives::On(stream));
      tsl::BlockUntilReady(send_event);
      if (send_event.IsError()) {
        return send_event.GetError();
      }
    }

    if (source_id) {
      // NVSHMEM put/get API is one-way communication - the sender initiates the
      // transfer and the receiver doesn't need to explicitly receive. We use a
      // barrier here to ensure all puts have completed before proceeding.
      VLOG(1) << "CollectivePermute: rank " << device_ordinal
              << " receiving data from source " << *source_id;

      TF_RETURN_IF_ERROR(nvshmem_comm->Barrier(GpuCollectives::On(stream)));
    }
  }

  if (!source_id) {
    // If there is no source peer, zero out dest buffer
    VLOG(3) << absl::StreamFormat("%s : collective-Permute: Issuing MemZero",
                                  device_string);
    for (DeviceBufferPair& buffer : buffers) {
      TF_RETURN_IF_ERROR(stream.MemZero(&buffer.destination_buffer,
                                        buffer.destination_buffer.size()));
    }
  }

  return absl::OkStatus();
}

/*static*/ absl::Status NvshmemCollectivePermuteStartThunk::CheckImplementable(
    const HloCollectivePermuteInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  // Check if the operation is degenerate (no communication needed)
  if (CollectivePermuteStartThunk::IsDegenerate(inst, replica_count,
                                                partition_count)) {
    return absl::OkStatus();
  }

  // Check if the operation is implementable with NVSHMEM
  for (const auto& operand : inst->operands()) {
    TF_RETURN_IF_ERROR(IsValidNvshmemOperand(
        operand->shape(), Thunk::kNvshmemCollectivePermuteStart));
  }

  // Check if all source-target pairs are valid
  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs =
      inst->source_target_pairs();
  const int64_t expected_size =
      inst->channel_id().has_value() ? partition_count : replica_count;
  if (source_target_pairs.empty()) {
    return absl::InvalidArgumentError("No source-target pairs specified");
  }
  if (source_target_pairs.size() > expected_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Too many source-target pairs: %d > %d",
                        source_target_pairs.size(), expected_size));
  }

  return absl::OkStatus();
}

NvshmemCollectivePermuteDoneThunk::NvshmemCollectivePermuteDoneThunk(
    ThunkInfo thunk_info,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events,
    AsyncStreamKind stream_kind)
    : NvshmemCollectiveDoneThunk(Thunk::kNvshmemCollectivePermuteDone,
                                 std::move(thunk_info), async_events,
                                 stream_kind) {}

absl::Status NvshmemCollectivePermuteDoneThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  TF_RETURN_IF_ERROR(NvshmemCollectiveDoneThunk::ExecuteOnStream(params));

  // Perform a fence operation to ensure all memory operations are completed
  TF_ASSIGN_OR_RETURN(auto* collectives, GetNvshmemCollectivesFromRegistry());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Communicator> nvshmem_comm,
                      collectives->CreateCommunicator());
  return nvshmem_comm->Fence();
}

}  // namespace gpu
}  // namespace xla

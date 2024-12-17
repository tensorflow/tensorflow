/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/nccl_collective_permute_thunk.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {
namespace {
absl::StatusOr<const int64_t> GetCurrentId(
    Thunk::CollectiveExecuteParams* collective_params,
    const NcclP2PConfig& config) {
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

bool IsLocalPeerTransfer(
    const NcclP2PConfig::SourceTargetMapEntry& source_target,
    const int64_t current_id, const int64_t device_count) {
  const std::optional<int64_t> source_id = source_target.source;
  const std::optional<int64_t> target_id = source_target.target;
  // Mixing nccl p2p with p2p memcopy will cause random deadlocks, namely
  // when calling nccl call and cuda memcpy p2p together(which both are
  // synchronizing devices), in this case if this rank is sending across host
  // using a nccl call but receiving from a local peer which is going through
  // cuda api, the deadlock could happen because nccl cannot ensure the
  // order of cuda api calls.
  // We determine if it's a local peer if the source/target id is within a node
  // if they are present.
  int64_t host_id = (current_id / device_count);
  if (source_id && host_id != *source_id / device_count) return false;
  if (target_id && host_id != *target_id / device_count) return false;
  return true;
}

}  // namespace

NcclCollectivePermuteStartThunk::NcclCollectivePermuteStartThunk(
    ThunkInfo thunk_info, const HloCollectivePermuteInstruction* instr,
    int64_t replica_count, int64_t partition_count, const Buffer& buffer,
    bool p2p_memcpy_enabled)
    : NcclCollectiveThunk(Thunk::kNcclCollectivePermuteStart, thunk_info,
                          IsSyncCollective(instr)),
      config_(GetNcclP2PConfig(instr, replica_count, partition_count)),
      buffer_(buffer),
      p2p_memcpy_enabled_(p2p_memcpy_enabled) {}

/*static*/ NcclP2PConfig NcclCollectivePermuteStartThunk::GetNcclP2PConfig(
    const HloCollectivePermuteInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  NcclP2PConfig collective_permute_config;
  auto& config = collective_permute_config.config;

  config.operand_count = 1;
  const Shape shape = instr->operand(0)->shape();
  config.operand_element_type.push_back(shape.element_type());
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

    collective_permute_config.id_to_source_target.insert({target, {}})
        .first->second.source = source;
    collective_permute_config.id_to_source_target.insert({source, {}})
        .first->second.target = target;
  }

  return collective_permute_config;
}

/*static*/ bool NcclCollectivePermuteStartThunk::IsDegenerate(
    const HloCollectivePermuteInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  // The collective permute is degenerate if all source-target pairs are
  // identity, and all the IDs appear in the list.
  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs =
      instr->source_target_pairs();
  // Each ID can appear only once as a source and as a target. So if all pairs
  // are identity, all IDs must appear in the list is the size == number of
  // replicas/partitions.
  const int64_t expected_size =
      instr->channel_id().has_value() ? partition_count : replica_count;
  return source_target_pairs.size() == expected_size &&
         absl::c_all_of(source_target_pairs,
                        [](const std::pair<int64_t, int64_t>& source_target) {
                          return source_target.first == source_target.second;
                        });
}

/*static*/ CollectiveOpGroupMode NcclCollectivePermuteStartThunk::GetGroupMode(
    const HloCollectivePermuteInstruction* instr) {
  return GetCollectiveOpGroupMode(instr->channel_id().has_value(), std::nullopt)
      .value();
}

absl::Status NcclCollectivePermuteStartThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::Initialize(params));
  device_count_ = params.local_device_count;
  CHECK_GT(device_count_, 0);
  VLOG(5) << "Local device count: " << device_count_;

  if (p2p_memcpy_enabled_) {
    TF_ASSIGN_OR_RETURN(const int64_t current_id,
                        GetCurrentId(params.collective_params, config_));
    absl::MutexLock lock(&barrier_mutex_);
    if (barrier_flags_.find(current_id) == barrier_flags_.end()) {
      if (!params.stream->parent()->HostMemoryRegister(
              &barrier_flags_[current_id], sizeof(uint8_t))) {
        LOG(ERROR) << "Registering barrier flag failed.";
      }
    }

    TF_ASSIGN_OR_RETURN(
        std::vector<DeviceBufferPair> device_buffers,
        ConvertToDeviceBuffers(params.buffer_allocations, {buffer_},
                               config_.config.operand_element_type));
    TF_RET_CHECK(device_buffers.size() == 1) << "Expected one buffer pair.";
    DeviceBufferPair& buffer = device_buffers[0];
    const NcclP2PConfig::SourceTargetMapEntry source_target =
        NcclP2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);

    const std::optional<int64_t> source_id = source_target.source;
    se::DeviceMemoryBase dest_addr = buffer.destination_buffer;

    TF_RETURN_IF_ERROR(recv_ptr_map_.InitializeId(current_id));

    if (source_id) {
      TF_RETURN_IF_ERROR(
          recv_ptr_map_.PutRecvPtr(current_id, dest_addr.opaque()));
    }
  }

  return absl::OkStatus();
}

absl::Status NcclCollectivePermuteStartThunk::Cleanup(
    const CleanupParams& params) {
  TF_ASSIGN_OR_RETURN(const int64_t current_id,
                      GetCurrentId(params.collective_params, config_));

  absl::MutexLock lock(&barrier_mutex_);
  if (!params.executor->HostMemoryUnregister(&barrier_flags_[current_id])) {
    LOG(ERROR) << "Unregistering barrier flag failed.";
  }
  return absl::OkStatus();
}

absl::Status NcclCollectivePermuteStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, {buffer_},
                             config_.config.operand_element_type));
  TF_RET_CHECK(device_buffers.size() == 1) << "Expected one buffer pair.";
  TF_ASSIGN_OR_RETURN(const int64_t current_id,
                      GetCurrentId(params.collective_params, config_));
  std::string device_string = GetDeviceString(*params.collective_params);

  const NcclP2PConfig::SourceTargetMapEntry source_target =
      NcclP2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);
  bool is_local_peer =
      IsLocalPeerTransfer(source_target, current_id, device_count_);
  VLOG(5) << "Is local peer : " << (is_local_peer ? "true" : "false");

  bool use_memcpy = is_local_peer && recv_ptr_map_.IsInitialized(current_id) &&
                    p2p_memcpy_enabled_;

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  if (use_memcpy) {
    se::DeviceMemoryBase sync_var_address =
        se::DeviceMemoryBase((void*)(&barrier_flags_[current_id]));
    TF_RETURN_IF_ERROR(comm_handle.comm->AllReduce(
        sync_var_address, sync_var_address, PrimitiveType::U8, 1,
        ReductionKind::MIN, GpuCollectives::On(stream)));
  }

  return ::xla::gpu::RunCollectivePermute(
      collectives, source_target, device_buffers[0], stream, comm_handle.comm,
      device_string, current_id, use_memcpy, recv_ptr_map_);
}

absl::Status RunCollectivePermute(
    GpuCollectives* collectives,
    NcclP2PConfig::SourceTargetMapEntry source_target, DeviceBufferPair& buffer,
    se::Stream& stream, Communicator* comm, absl::string_view device_string,
    int64_t current_id, bool use_memcpy,
    NcclCollectivePermuteStartThunk::RecvPtrMap& recv_ptr_map) {
  // Determine the source and target IDs for this instance. The source ID is the
  // ID which will copy its data to this instance. The destination ID is the ID
  // to which this instance will copy its data. Either are optional.
  //
  // No source and no dest:
  //  - this instance does not actually participate, no one send it any data and
  //    it does not have to send any data as well. Since there is no dest,
  //    just memzero() the dest buffer as required by the collective permute
  //    semantics.
  //
  // No source, dest present:
  //  - This instance has to send data to 'dest' Issue an send of the input.
  //    Since there is no source, memzero the dest buffer.
  //
  // Source present, no destination:
  //  - This instance received data from the source, does not have to send data
  //    to anyone, Issue a receive.
  //
  // Source and dest both present:
  //   - Issue a send of the input to dest, receive for the output from the
  //     src.
  //
  //

  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing collective permute from device ordinal: "
          << device_ordinal << " current_id " << current_id;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), {buffer}, comm));

  const std::optional<int64_t> source_id = source_target.source;
  const std::optional<int64_t> target_id = source_target.target;

  se::DeviceMemoryBase src_addr = buffer.source_buffer;
  se::DeviceMemoryBase dest_addr = buffer.destination_buffer;

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d, target_id = %d",
                                device_string, current_id,
                                source_id.value_or(-1), target_id.value_or(-1));

  if (!use_memcpy) {
    // GroupStart/End API is needed only if we will issue both send & recv
    // calls.
    const bool is_nccl_group_needed = (target_id && source_id);
    if (is_nccl_group_needed) {
      TF_RETURN_IF_ERROR(collectives->GroupStart());
    }
    // Send source buffer to target peer if needed.
    if (target_id) {
      TF_RETURN_IF_ERROR(comm->Send(src_addr, buffer.element_type,
                                    buffer.element_count, *target_id,
                                    GpuCollectives::On(stream)));
    }

    // Receive data from the source peer to the destination buffer.
    if (source_id) {
      TF_RETURN_IF_ERROR(comm->Recv(dest_addr, buffer.element_type,
                                    buffer.element_count, *source_id,
                                    GpuCollectives::On(stream)));
    }
    if (is_nccl_group_needed) {
      TF_RETURN_IF_ERROR(collectives->GroupEnd());
    }
  }

  if (!source_id) {
    // If there is no source peer, i.e. no one send us any data, zero out dest
    // buffer.
    VLOG(3) << absl::StreamFormat("%s : collective-Permute: Issuing MemZero",
                                  device_string);
    TF_RETURN_IF_ERROR(stream.MemZero(&dest_addr, dest_addr.size()));
  }
  if (use_memcpy && target_id) {
    TF_ASSIGN_OR_RETURN(auto recv_ptr, recv_ptr_map.GetRecvPtr(*target_id));

    VLOG(3) << "Using memcpy, received target pointer: " << recv_ptr.get()
            << " current_id " << current_id << " target_id: " << *target_id;

    VLOG(3) << current_id << " initiating memcpy to " << *target_id;
    se::DeviceMemoryBase dst_addr = se::DeviceMemoryBase(recv_ptr.get());
    TF_RETURN_IF_ERROR(stream.MemcpyD2D(&dst_addr, src_addr, src_addr.size()));
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

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

#include "xla/backends/gpu/runtime/collective_permute_thunk.h"

#include <algorithm>
#include <cstddef>
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
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/rendezvous.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
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

bool IsLocalPeerTransfer(const P2PConfig::SourceTargetMapEntry& source_target,
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

CollectivePermuteStartThunk::CollectivePermuteStartThunk(
    ThunkInfo thunk_info, const HloCollectivePermuteInstruction* instr,
    int64_t replica_count, int64_t partition_count,
    const std::vector<Buffer>& buffers, bool p2p_memcpy_enabled,
    AsyncStreamKind stream_kind)
    : CollectiveThunk(Thunk::kCollectivePermuteStart, thunk_info,
                      IsGPUSyncCollective(*instr), stream_kind),
      config_(GetP2PConfig(instr, replica_count, partition_count)),
      buffers_(buffers),
      p2p_memcpy_enabled_(p2p_memcpy_enabled) {}

/*static*/ P2PConfig CollectivePermuteStartThunk::GetP2PConfig(
    const HloCollectivePermuteInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  P2PConfig collective_permute_config;
  auto& config = collective_permute_config.config;

  config.operand_count = instr->operand_count();
  for (int i = 0; i < config.operand_count; ++i) {
    config.operand_element_type.push_back(
        instr->operand(i)->shape().element_type());
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

    collective_permute_config.id_to_source_target.insert({target, {}})
        .first->second.source = source;
    collective_permute_config.id_to_source_target.insert({source, {}})
        .first->second.target = target;
  }

  return collective_permute_config;
}

/*static*/ bool CollectivePermuteStartThunk::IsDegenerate(
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

/*static*/ CollectiveOpGroupMode CollectivePermuteStartThunk::GetGroupMode(
    const HloCollectivePermuteInstruction* instr) {
  return GetCollectiveOpGroupMode(instr->channel_id().has_value(), std::nullopt)
      .value();
}

absl::Status CollectivePermuteStartThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  device_count_ = params.local_device_count;
  CHECK_GT(device_count_, 0);
  VLOG(5) << "Local device count: " << device_count_;

  if (p2p_memcpy_enabled_) {
    TF_ASSIGN_OR_RETURN(const int64_t current_id,
                        GetCurrentId(params.collective_params, config_));
    {
      absl::MutexLock lock(&barrier_mutex_);
      if (receiver_barrier_events_.find(current_id) ==
          receiver_barrier_events_.end()) {
        TF_ASSIGN_OR_RETURN(auto receiver_event,
                            params.executor->CreateEvent());
        receiver_barrier_events_.emplace(current_id, std::move(receiver_event));
      }
      if (sender_barrier_events_.find(current_id) ==
          sender_barrier_events_.end()) {
        TF_ASSIGN_OR_RETURN(auto sender_event, params.executor->CreateEvent());
        sender_barrier_events_.emplace(current_id, std::move(sender_event));
      }
    }
    TF_ASSIGN_OR_RETURN(
        std::vector<DeviceBufferPair> device_buffers,
        ConvertToDeviceBuffers(params.buffer_allocations, {buffers_},
                               config_.config.operand_element_type));
    const P2PConfig::SourceTargetMapEntry source_target =
        P2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);

    const std::optional<int64_t> source_id = source_target.source;

    TF_RETURN_IF_ERROR(recv_ptr_map_.InitializeId(current_id));

    if (source_id) {
      std::vector<se::DeviceMemoryBase> dest_addrs;
      std::transform(device_buffers.begin(), device_buffers.end(),
                     std::back_inserter(dest_addrs),
                     [](const DeviceBufferPair& buffer) {
                       return buffer.destination_buffer;
                     });
      std::vector<void*> dest_opaques;
      std::transform(
          dest_addrs.begin(), dest_addrs.end(),
          std::back_inserter(dest_opaques),
          [](se::DeviceMemoryBase dest_addr) { return dest_addr.opaque(); });
      TF_RETURN_IF_ERROR(recv_ptr_map_.PutRecvPtr(current_id, dest_opaques));
    }
  }

  return absl::OkStatus();
}
struct CallRendezvousKey {
  RunId run_id;

  template <typename H>
  friend H AbslHashValue(H h, const CallRendezvousKey& key) {
    return H::combine(std::move(h), key.run_id);
  }
};

bool operator==(const CallRendezvousKey& a, const CallRendezvousKey& b) {
  return a.run_id == b.run_id;
}

absl::Status CollectivePermuteStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params,
                             std::vector<CollectiveThunk::Buffer>(buffers_),
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(const int64_t current_id,
                      GetCurrentId(params.collective_params, config_));
  std::string device_string = GetDeviceString(*params.collective_params);

  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);
  bool is_local_peer =
      IsLocalPeerTransfer(source_target, current_id, device_count_);
  VLOG(5) << "Is local peer : " << (is_local_peer ? "true" : "false");

  bool use_memcpy = is_local_peer && recv_ptr_map_.IsInitialized(current_id) &&
                    p2p_memcpy_enabled_;

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  if (use_memcpy) {
    std::optional<int64_t> source_id = source_target.source;
    std::optional<int64_t> target_id = source_target.target;
    // Due to the one-sided push mechanism of memcpy p2p, we need to make sure
    // the buffer on the receiving side is ready before sender pushes the data.
    // Receiving side will record an event and the sender will wait for the
    // event before proceeding.
    if (source_id) {
      absl::MutexLock lock(&barrier_mutex_);
      auto receiver_event = receiver_barrier_events_.find(current_id);
      TF_RETURN_IF_ERROR(stream.RecordEvent(receiver_event->second.get()));
    }
    TF_ASSIGN_OR_RETURN(
        size_t num_local_participants,
        GetNumLocalParticipants(*params.collective_params,
                                config().replica_groups, config().group_mode));

    auto rendezvous_name = absl::StrFormat(
        "rendezvous before calling collective-permute; run_id=%d; op id:%d; "
        "num_local_participants:%d",
        params.collective_params->run_id.ToInt(), config_.config.op_id,
        num_local_participants);
    auto rendezvous_key = CallRendezvousKey{params.collective_params->run_id};

    // Perform a rendezvous to make sure all receivers have their events
    // recorded.
    Rendezvous(rendezvous_name, rendezvous_key, num_local_participants,
               /*warn_stuck_timeout=*/absl::Seconds(20),
               /*terminate_timeout=*/absl::Seconds(40));

    // For sending side, wait for the recorded event from the receiving side.
    if (target_id) {
      absl::MutexLock lock(&barrier_mutex_);
      auto receiver_event = receiver_barrier_events_.find(*target_id);
      TF_RETURN_IF_ERROR(stream.WaitFor(receiver_event->second.get()));
    }
  }

  auto status = ::xla::gpu::RunCollectivePermute(
      collectives, source_target, device_buffers, stream, comm_handle.comm,
      device_string, current_id, use_memcpy, recv_ptr_map_);

  if (use_memcpy) {
    std::optional<int64_t> source_id = source_target.source;
    std::optional<int64_t> target_id = source_target.target;
    // After the memcpy p2p is dispatched, the receiver needs to
    // wait for the sender's event before proceeding to ensure
    // data has been copied.
    if (target_id) {
      absl::MutexLock lock(&barrier_mutex_);
      auto sender_event = sender_barrier_events_.find(current_id);
      TF_RETURN_IF_ERROR(stream.RecordEvent(sender_event->second.get()));
    }
    TF_ASSIGN_OR_RETURN(
        size_t num_local_participants,
        GetNumLocalParticipants(*params.collective_params,
                                config().replica_groups, config().group_mode));

    auto rendezvous_name = absl::StrFormat(
        "rendezvous after calling collective-permute; run_id=%d; op id:%d; "
        "num_local_participants:%d",
        params.collective_params->run_id.ToInt(), config_.config.op_id,
        num_local_participants);
    auto rendezvous_key = CallRendezvousKey{params.collective_params->run_id};

    // Perform a rendezvous to make sure all senders have their events
    // recorded.
    Rendezvous(rendezvous_name, rendezvous_key, num_local_participants,
               /*warn_stuck_timeout=*/absl::Seconds(20),
               /*terminate_timeout=*/absl::Seconds(40));

    // For receiving side, wait for the recorded event from the sending side.
    if (source_id) {
      absl::MutexLock lock(&barrier_mutex_);
      auto sender_event = sender_barrier_events_.find(*source_id);
      TF_RETURN_IF_ERROR(stream.WaitFor(sender_event->second.get()));
    }
  }

  return status;
}

absl::Status RunCollectivePermute(
    GpuCollectives* collectives, P2PConfig::SourceTargetMapEntry source_target,
    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
    Communicator* comm, absl::string_view device_string, int64_t current_id,
    bool use_memcpy, CollectivePermuteStartThunk::RecvPtrMap& recv_ptr_map) {
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
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  std::optional<int64_t> source_id = source_target.source;
  std::optional<int64_t> target_id = source_target.target;

  std::vector<se::DeviceMemoryBase> src_addrs, dest_addrs;
  std::transform(
      buffers.begin(), buffers.end(), std::back_inserter(src_addrs),
      [](const DeviceBufferPair& buffer) { return buffer.source_buffer; });
  std::transform(
      buffers.begin(), buffers.end(), std::back_inserter(dest_addrs),
      [](const DeviceBufferPair& buffer) { return buffer.destination_buffer; });

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d, target_id = %d",
                                device_string, current_id,
                                source_id.value_or(-1), target_id.value_or(-1));

  if (!use_memcpy) {
    // GroupStart/End API is needed if we need to dispatch multiple NCCL kernels
    // for multiple buffers
    const bool is_nccl_group_needed = (buffers.size() > 1);
    if (is_nccl_group_needed) {
      TF_RETURN_IF_ERROR(collectives->GroupStart());
    }

    std::optional<RankId> source_rank;
    std::vector<RankId> target_ranks;
    if (source_id) source_rank = RankId(*source_id);
    if (target_id) target_ranks.push_back(RankId(*target_id));

    for (uint64_t idx = 0; idx < buffers.size(); ++idx) {
      const auto src_addr = src_addrs.at(idx);
      const auto dest_addr = dest_addrs.at(idx);
      const auto buffer = buffers.at(idx);
      TF_RETURN_IF_ERROR(comm->CollectivePermute(
          src_addr, dest_addr, buffer.element_type, buffer.element_count,
          source_rank, target_ranks, GpuCollectives::On(stream)));
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
    for (se::DeviceMemoryBase& dest_addr : dest_addrs) {
      TF_RETURN_IF_ERROR(stream.MemZero(&dest_addr, dest_addr.size()));
    }
  }

  if (use_memcpy && target_id) {
    TF_ASSIGN_OR_RETURN(auto recv_ptrs, recv_ptr_map.GetRecvPtr(*target_id));

    VLOG(3) << "Using memcpy, received target pointers, current_id: "
            << current_id << " target_id: " << *target_id;

    VLOG(3) << current_id << " initiating memcpy to " << *target_id;
    for (uint64_t idx = 0; idx < buffers.size(); ++idx) {
      se::DeviceMemoryBase dst_addr =
          se::DeviceMemoryBase(recv_ptrs.get().at(idx));
      auto src_addr = src_addrs.at(idx);
      TF_RETURN_IF_ERROR(
          stream.MemcpyD2D(&dst_addr, src_addr, src_addr.size()));
    }
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

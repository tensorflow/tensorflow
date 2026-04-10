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
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_clique_rendezvous.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/event_pool.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/unique_any.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

// Data exchanged between ranks during RunCollective memcpy path via
// GpuCliqueRendezvous. Each rank creates one of these and all ranks
// receive the full set after the rendezvous completes.
struct Events {
  // Pool-borrowed event recorded on this rank's stream before the rendezvous.
  // Signals that all prior work on this rank's buffers is complete.
  EventPool::Event ready;

  // Future that resolves with a pool-borrowed event recorded AFTER memcpy.
  // The receiver awaits this future, then WaitFors the resolved event to
  // synchronize the memcpy completion onto its own stream.
  Future<EventPool::Event> done;
};

}  // namespace

static absl::Status RunP2PMemcpy(
    const P2PConfig::SourceTargetRanks& source_target,
    const std::vector<DeviceBufferPair>& device_buffers, se::Stream& stream,
    const GpuCliqueKey& clique_key, const Thunk::ExecuteParams& params,
    ThunkId thunk_id);

CollectivePermuteThunk::CollectivePermuteThunk(
    ThunkInfo thunk_info, const HloCollectivePermuteInstruction* instr,
    int64_t replica_count, int64_t partition_count,
    const std::vector<Buffer>& buffers, bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kCollectivePermute, std::move(thunk_info),
                      CommunicationId(1)),
      config_(GetP2PConfig(instr, replica_count, partition_count)),
      buffers_(buffers),
      p2p_memcpy_enabled_(p2p_memcpy_enabled),
      id_to_component_members_(
          InitConnectedComponents(config_, p2p_memcpy_enabled)) {}

CollectivePermuteThunk::CollectivePermuteThunk(
    ThunkInfo thunk_info, const P2PConfig& config,
    const std::vector<Buffer>& buffers, bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kCollectivePermute, std::move(thunk_info),
                      CommunicationId(1)),
      config_(config),
      buffers_(buffers),
      p2p_memcpy_enabled_(p2p_memcpy_enabled),
      id_to_component_members_(
          InitConnectedComponents(config_, p2p_memcpy_enabled)) {}

absl::flat_hash_map<int64_t, std::vector<int64_t>>
CollectivePermuteThunk::InitConnectedComponents(const P2PConfig& config,
                                                bool p2p_memcpy_enabled) {
  if (!p2p_memcpy_enabled || config.config.replica_groups.empty()) {
    return {};
  }

  // Extract source-target pairs from the config.
  std::vector<std::pair<int64_t, int64_t>> pairs;
  for (const auto& [id, entry] : config.id_to_source_target) {
    if (entry.source.has_value()) {
      pairs.push_back({*entry.source, id});
    }
  }

  int64_t num_participants =
      config.config.replica_groups.front().replica_ids_size();

  // Build a lookup from each logical ID to its component members.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> result;
  for (auto& [root, members] :
       SourceTargetConnectedComponents(num_participants, pairs)) {
    for (int64_t id : members) {
      result[id] = members;
    }
  }
  return result;
}

absl::StatusOr<GpuCliqueKey>
CollectivePermuteThunk::BuildCommunicatingCliqueKey(
    int64_t current_id, const CollectiveParams& params) const {
  auto it = id_to_component_members_.find(current_id);
  if (it == id_to_component_members_.end()) {
    return Internal("Logical ID %d not found in connected components",
                    current_id);
  }
  const std::vector<int64_t>& component = it->second;

  ASSIGN_OR_RETURN(auto logical_id, params.device_assn->LogicalIdForDevice(
                                        params.global_device_id));

  // Map component members from logical IDs to GlobalDeviceIds.
  auto id_to_global = [&](int64_t id) -> GlobalDeviceId {
    if (config_.config.group_mode ==
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA) {
      return GlobalDeviceId(
          (*params.device_assn)(id, logical_id.computation_id));
    }
    return GlobalDeviceId((*params.device_assn)(logical_id.replica_id, id));
  };

  std::vector<GlobalDeviceId> devices;
  devices.reserve(component.size());
  for (int64_t id : component) {
    devices.push_back(id_to_global(id));
  }

  // Count how many of those devices are local to this process.
  int64_t num_local = 0;
  for (const auto& gid : devices) {
    for (const auto& [local_id, global_id] : *params.global_device_id_map) {
      if (global_id == gid) {
        ++num_local;
        break;
      }
    }
  }

  return GpuCliqueKey(std::move(devices), num_local);
}

P2PConfig CollectivePermuteThunk::GetP2PConfig(
    const HloCollectivePermuteInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  P2PConfig collective_permute_config;
  auto& config = collective_permute_config.config;
  config.use_symmetric_buffer =
      instr->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_enable_nccl_symmetric_buffers();
  for (const HloInstruction* operand : instr->operands()) {
    config.operand_element_type.push_back(operand->shape().element_type());
  }
  config.group_mode = GetGroupMode(instr);

  // With a collective permute, all execution instances together form one
  // replica group.
  const int64_t num_participants =
      config.group_mode ==
              CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
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

bool CollectivePermuteThunk::IsDegenerate(
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

CollectiveOpGroupMode CollectivePermuteThunk::GetGroupMode(
    const HloCollectivePermuteInstruction* instr) {
  return GetCollectiveOpGroupMode(instr->channel_id().has_value(), std::nullopt)
      .value();
}

absl::Status CollectivePermuteThunk::InitializeCollective(
    const InitializeParams& params, const GpuCliqueKey& clique_key) {
  if (!p2p_memcpy_enabled_ || !params.execution_scoped_state) {
    return absl::OkStatus();
  }

  // Find the communicating component for the current device and build a
  // clique key covering only those devices.
  ASSIGN_OR_RETURN(int64_t current_id,
                   GetCollectiveCurrentId(params.collective_params, config_));
  ASSIGN_OR_RETURN(
      GpuCliqueKey communicating_clique,
      BuildCommunicatingCliqueKey(current_id, *params.collective_params));

  // Only use p2p memcpy if the entire communicating component is local.
  if (!communicating_clique.is_local()) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, {buffers_},
                                          config_.config.operand_element_type));

  GlobalDeviceId gid = params.collective_params->global_device_id;
  std::optional<RankId> rank = communicating_clique.rank(gid);
  if (!rank.has_value()) {
    return Internal("Device %v not found in communicating clique key %v", gid,
                    communicating_clique);
  }

  // Exchange device buffer pairs with other ranks in the communicating
  // component via rendezvous.
  ASSIGN_OR_RETURN(auto local_device_buffers,
                   GpuCliqueRendezvous::Join(communicating_clique, *rank,
                                             std::move(device_buffers)));

  // Collect device buffer pairs from all participating ranks.
  size_t num_local = communicating_clique.num_local_participants();
  LocalPermuteState state;

  for (auto peer = RankId(0); peer < RankId(num_local); ++peer) {
    ASSIGN_OR_RETURN(
        const std::vector<DeviceBufferPair>& peer_buffers,
        local_device_buffers->at<std::vector<DeviceBufferPair>>(peer));
    state.buffer_pairs[peer] = peer_buffers;
  }

  // Store the state in execution-scoped state so it lives for the duration
  // of the execution and is accessible from RunCollective.
  params.execution_scoped_state->try_emplace(
      thunk_info().thunk_id, std::in_place_type<LocalPermuteState>,
      std::move(state));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CollectivePermuteThunk>>
CollectivePermuteThunk::FromProto(
    ThunkInfo thunk_info, const CollectivePermuteStartThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  CollectiveConfig config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  P2PConfig::IdToSourceTargetMap id_to_source_target;
  for (const SourceTarget& source_target : thunk_proto.source_target_pairs()) {
    id_to_source_target.insert({source_target.target(), {}})
        .first->second.source = source_target.source();
    id_to_source_target.insert({source_target.source(), {}})
        .first->second.target = source_target.target();
  }

  return std::make_unique<CollectivePermuteThunk>(
      std::move(thunk_info), P2PConfig{config, std::move(id_to_source_target)},
      std::move(buffers), thunk_proto.p2p_memcpy_enabled());
}

absl::StatusOr<ThunkProto> CollectivePermuteThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CollectivePermuteStartThunkProto* thunk_proto =
      proto.mutable_collective_permute_start_thunk();

  for (const Buffer& buffer : buffers_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  thunk_proto->set_p2p_memcpy_enabled(p2p_memcpy_enabled_);

  std::vector<SourceTarget> source_target_pairs;
  source_target_pairs.reserve(config_.id_to_source_target.size() / 2);
  for (const auto& [key_id, map_entry] : config_.id_to_source_target) {
    SourceTarget pair;
    if (!map_entry.source.has_value()) {
      // Same pair is in the map with target/source switched.
      continue;
    }
    pair.set_source(*map_entry.source);
    pair.set_target(key_id);
    source_target_pairs.push_back(pair);
  }
  thunk_proto->mutable_source_target_pairs()->Assign(
      source_target_pairs.begin(), source_target_pairs.end());

  return proto;
}

CollectivePermuteThunk::BufferUses CollectivePermuteThunk::buffer_uses() const {
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

absl::Status CollectivePermuteThunk::RunCollective(
    const ExecuteParams& params, const GpuCliqueKey& clique_key,
    se::Stream& stream, Communicator& comm) {
  ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations,
                             std::vector<CollectiveThunk::Buffer>(buffers_),
                             config_.config.operand_element_type));
  ASSIGN_OR_RETURN(int64_t current_id,
                   GetCollectiveCurrentId(params.collective_params, config_));
  std::string device_string = GetDeviceString(*params.collective_params);

  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);

  // Remap source/target from logical IDs to communicator-local ranks.
  // The NCCL communicator covers ALL devices, so logical IDs directly map
  // to ranks in the full communicator.
  P2PConfig::SourceTargetRanks source_target_ranks;
  if (source_target.source) {
    source_target_ranks.source = RankId(*source_target.source);
  }
  if (source_target.target) {
    source_target_ranks.target = RankId(*source_target.target);
  }

  // Build the communicating clique key for this device's connected component
  // and check if it is fully local.
  std::optional<GpuCliqueKey> communicating_clique;
  bool use_p2p_memcpy = false;

  if (p2p_memcpy_enabled_) {
    ASSIGN_OR_RETURN(
        communicating_clique,
        BuildCommunicatingCliqueKey(current_id, *params.collective_params));
    if (communicating_clique->is_local()) {
      ASSIGN_OR_RETURN(
          use_p2p_memcpy,
          params.collective_cliques->peer_access_enabled(clique_key));
    }
  }

  if (!use_p2p_memcpy) {
    return ::xla::gpu::RunCollectivePermute(
        source_target_ranks, device_buffers, stream, comm, device_string,
        current_id, config_.config.use_symmetric_buffer);
  }

  // For the memcpy path, remap source/target to communicating clique ranks.
  ASSIGN_OR_RETURN(
      auto remapped_source_target,
      RemapSourceTargetToCliqueRanks(
          source_target, *communicating_clique,
          *params.collective_params->device_assn, config_.config.group_mode,
          params.collective_params->global_device_id));

  return RunP2PMemcpy(remapped_source_target, device_buffers, stream,
                      *communicating_clique, params, thunk_info().thunk_id);
}

absl::Status RunCollectivePermute(P2PConfig::SourceTargetRanks source_target,
                                  const std::vector<DeviceBufferPair>& buffers,
                                  se::Stream& stream, Communicator& comm,
                                  absl::string_view device_string,
                                  int64_t current_id,
                                  bool use_symmetric_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Performing collective permute, current_id " << current_id;

  std::vector<se::DeviceAddressBase> src_addrs, dest_addrs;
  absl::c_transform(
      buffers, std::back_inserter(src_addrs),
      [](const DeviceBufferPair& buffer) { return buffer.source_buffer; });
  absl::c_transform(
      buffers, std::back_inserter(dest_addrs),
      [](const DeviceBufferPair& buffer) { return buffer.destination_buffer; });

  VLOG(3) << absl::StreamFormat(
      "%s : id = %d, source_rank = %v, target_rank = %v", device_string,
      current_id, source_target.source ? *source_target.source : RankId{-1},
      source_target.target ? *source_target.target : RankId{-1});

  std::vector<RankId> target_ranks;
  if (source_target.target) {
    target_ranks.push_back(*source_target.target);
  }

  // GroupStart/End API is needed if we need to dispatch multiple NCCL kernels
  // for multiple buffers.
  if (buffers.size() <= 1) {
    for (uint64_t idx = 0; idx < buffers.size(); ++idx) {
      se::DeviceAddressBase src = src_addrs.at(idx);
      se::DeviceAddressBase dst = dest_addrs.at(idx);
      const DeviceBufferPair& buf = buffers.at(idx);
      auto future = comm.CollectivePermute(
          src, dst, buf.element_type, buf.element_count, source_target.source,
          target_ranks, GpuCollectives::On(stream));
      TF_RETURN_IF_ERROR(future.Await());
    }
  } else {
    TF_RETURN_IF_ERROR(MaybeRegisterBuffers(stream.parent(), buffers, &comm,
                                            use_symmetric_buffer));
    auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
    auto future = gpu_comm->GroupExecute(
        [&source_target, &buffers, &src_addrs, &dest_addrs, &target_ranks,
         &stream](GpuCommunicator* comm) -> absl::Status {
          for (uint64_t idx = 0; idx < buffers.size(); ++idx) {
            se::DeviceAddressBase src = src_addrs.at(idx);
            se::DeviceAddressBase dst = dest_addrs.at(idx);
            const DeviceBufferPair& buf = buffers.at(idx);
            TF_RETURN_IF_ERROR(comm->LaunchCollectivePermute(
                src, dst, buf.element_type, buf.element_count,
                source_target.source, target_ranks,
                GpuCollectives::On(stream)));
          }
          return absl::OkStatus();
        });
    TF_RETURN_IF_ERROR(future.Await());
  }

  if (!source_target.source) {
    // If there is no source peer, i.e. no one send us any data, zero out dest
    // buffer.
    VLOG(3) << absl::StreamFormat("%s : collective-Permute: Issuing MemZero",
                                  device_string);
    for (se::DeviceAddressBase& dest_addr : dest_addrs) {
      TF_RETURN_IF_ERROR(stream.MemZero(&dest_addr, dest_addr.size()));
    }
  }

  return absl::OkStatus();
}

// Performs a collective permute using direct D2D memcpy between local GPU
// peers, synchronized via GpuCliqueRendezvous and EventPool-borrowed events.
static absl::Status RunP2PMemcpy(
    const P2PConfig::SourceTargetRanks& source_target,
    const std::vector<DeviceBufferPair>& device_buffers, se::Stream& stream,
    const GpuCliqueKey& clique_key, const Thunk::ExecuteParams& params,
    ThunkId thunk_id) {
  GlobalDeviceId gid = params.collective_params->global_device_id;
  std::optional<RankId> rank = clique_key.rank(gid);
  if (!rank.has_value()) {
    return Internal("Device %v not found in clique key %v", gid, clique_key);
  }

  // Get EventPool from own StreamExecutor.
  auto* pool =
      stream.parent()->GetOrConstructResource<EventPool>(stream.parent());

  // Borrow a "ready" event and record it on our stream to signal that all
  // prior work on this rank's buffers is complete.
  ASSIGN_OR_RETURN(EventPool::Event ready, pool->GetOrCreateEvent());
  TF_RETURN_IF_ERROR(stream.RecordEvent(ready->get()));

  // Create promise/future pair for the "done" event that the sender will
  // set after completing the memcpy.
  auto [done_promise, done_future] = MakePromise<EventPool::Event>();

  // Join the rendezvous, exchanging Events with all other ranks.
  Events my_events{std::move(ready), std::move(done_future)};
  ASSIGN_OR_RETURN(
      auto rendezvous,
      GpuCliqueRendezvous::Join(clique_key, *rank, std::move(my_events)));

  // Sender: copy data to target's destination buffers.
  if (source_target.target) {
    RankId target = *source_target.target;

    // Wait for target's stream to be ready before writing to its buffers.
    ASSIGN_OR_RETURN(const Events& target_events,
                     rendezvous->at<Events>(target));
    TF_RETURN_IF_ERROR(stream.WaitFor(target_events.ready->get()));

    auto it_state = params.execution_scoped_state->find(thunk_id);
    if (it_state == params.execution_scoped_state->end()) {
      return Internal("LocalPermuteState not found for thunk %v", thunk_id);
    }

    auto* state = tsl::any_cast<CollectivePermuteThunk::LocalPermuteState>(
        &it_state->second);
    if (!state) {
      return Internal("LocalPermuteState type mismatch for thunk %v", thunk_id);
    }

    auto it = state->buffer_pairs.find(target);
    if (it == state->buffer_pairs.end()) {
      return Internal("Buffer pairs not found for target rank %d",
                      target.value());
    }
    const std::vector<DeviceBufferPair>& target_buffers = it->second;

    // Perform D2D copies from our source to target's destination.
    for (size_t i = 0; i < device_buffers.size(); ++i) {
      auto dst_addr = target_buffers[i].destination_buffer;
      auto src_addr = device_buffers[i].source_buffer;
      TF_RETURN_IF_ERROR(
          stream.MemcpyD2D(&dst_addr, src_addr, src_addr.size()));
    }

    // Record a "done" event and fulfill the promise so the target knows
    // the copy is complete.
    ASSIGN_OR_RETURN(EventPool::Event done, pool->GetOrCreateEvent());
    TF_RETURN_IF_ERROR(stream.RecordEvent(done->get()));
    done_promise.Set(std::move(done));
  } else {
    // Not a sender — fulfill promise with a dummy event.
    ASSIGN_OR_RETURN(EventPool::Event done, pool->GetOrCreateEvent());
    done_promise.Set(std::move(done));
  }

  // No source: zero out destination buffers.
  if (!source_target.source) {
    for (const auto& buf : device_buffers) {
      auto dest = buf.destination_buffer;
      TF_RETURN_IF_ERROR(stream.MemZero(&dest, dest.size()));
    }
  }

  // Receiver: wait for source's memcpy to complete.
  if (source_target.source) {
    RankId source = *source_target.source;
    ASSIGN_OR_RETURN(const Events& source_events,
                     rendezvous->at<Events>(source));

    // Await the source's done future (blocks until sender sets promise).
    const absl::StatusOr<EventPool::Event>& done_result =
        source_events.done.Await();
    if (!done_result.ok()) return done_result.status();
    TF_RETURN_IF_ERROR(stream.WaitFor((*done_result)->get()));
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Collective-permute communicating cliques helpers.
//===----------------------------------------------------------------------===//

absl::flat_hash_map<int64_t, std::vector<int64_t>>
SourceTargetConnectedComponents(
    int64_t num_participants,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs) {
  std::vector<int64_t> parent(num_participants);
  std::iota(parent.begin(), parent.end(), 0);

  // Find component root with path compression.
  auto find = [&](int64_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };

  for (const auto& [source, target] : source_target_pairs) {
    int64_t root_a = find(source);
    int64_t root_b = find(target);
    if (root_a != root_b) {
      parent[root_a] = root_b;
    }
  }

  // Group participants by component root; sort members for determinism.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> components;
  for (int64_t i = 0; i < num_participants; ++i) {
    components[find(i)].push_back(i);
  }
  for (auto& [root, members] : components) {
    absl::c_sort(members);
  }

  return components;
}

absl::StatusOr<P2PConfig::SourceTargetRanks> RemapSourceTargetToCliqueRanks(
    const P2PConfig::SourceTargetMapEntry& source_target,
    const GpuCliqueKey& clique_key, const DeviceAssignment& device_assn,
    CollectiveOpGroupMode group_mode, GlobalDeviceId global_device_id) {
  ASSIGN_OR_RETURN(auto logical_id,
                   device_assn.LogicalIdForDevice(global_device_id));

  // Map a logical ID (partition or replica) to its GlobalDeviceId.
  auto id_to_global = [&](int64_t id) -> GlobalDeviceId {
    if (group_mode ==
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA) {
      return GlobalDeviceId(device_assn(id, logical_id.computation_id));
    }
    return GlobalDeviceId(device_assn(logical_id.replica_id, id));
  };

  P2PConfig::SourceTargetRanks remapped;

  if (source_target.source) {
    GlobalDeviceId source_global = id_to_global(*source_target.source);
    std::optional<RankId> rank = clique_key.rank(source_global);
    if (!rank) {
      return Internal("Source device %v not found in clique key %v",
                      source_global, clique_key);
    }
    remapped.source = *rank;
  }

  if (source_target.target) {
    GlobalDeviceId target_global = id_to_global(*source_target.target);
    std::optional<RankId> rank = clique_key.rank(target_global);
    if (!rank) {
      return Internal("Target device %v not found in clique key %v",
                      target_global, clique_key);
    }
    remapped.target = *rank;
  }

  return remapped;
}

}  // namespace xla::gpu

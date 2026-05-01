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
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_clique_rendezvous.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/event_pool.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
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
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

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

static absl::Status RunPeerAccessPermute(
    const P2PConfig::SourceTargetRanks& source_target,
    const std::vector<DeviceBufferPair>& device_buffers, se::Stream& stream,
    const GpuCliqueKey& clique_key, const Thunk::ExecuteParams& params);

static absl::Status RunOneSidedPermute(
    const P2PConfig::SourceTargetRanks& source_target,
    const std::vector<DeviceBufferPair>& device_buffers, se::Stream& stream,
    const GpuCliqueKey& clique_key, const Thunk::ExecuteParams& params,
    Communicator& comm);

CollectivePermuteThunk::CollectivePermuteThunk(
    ThunkInfo thunk_info, const HloCollectivePermuteInstruction* instr,
    int64_t replica_count, int64_t partition_count,
    const std::vector<Buffer>& buffers, CollectivesMode collectives_mode,
    bool connected_components_enabled)
    : CollectiveThunk(Thunk::kCollectivePermute, std::move(thunk_info), buffers,
                      CommunicationId(1), collectives_mode),
      config_(GetP2PConfig(instr, replica_count, partition_count,
                           connected_components_enabled)),
      connected_components_enabled_(connected_components_enabled) {}

CollectivePermuteThunk::CollectivePermuteThunk(
    ThunkInfo thunk_info, const P2PConfig& config,
    const std::vector<Buffer>& buffers, CollectivesMode collectives_mode,
    bool connected_components_enabled)
    : CollectiveThunk(Thunk::kCollectivePermute, std::move(thunk_info), buffers,
                      CommunicationId(1), collectives_mode),
      config_(config),
      connected_components_enabled_(connected_components_enabled) {}

absl::Status CollectivePermuteThunk::PrepareCollective(
    const PrepareParams& params, const GpuCliqueKey& clique_key) {
  CollectiveMemoryRequests& mem_requests = *params.collective_memory_requests;

  if (use_symmetric_memory() && clique_key.is_local()) {
    // Request symmetric memory for both source and destination buffers.
    // One-sided Put requires both send and receive buffers to be registered
    // as symmetric memory windows.
    //
    // Use allocation indices directly rather than address-based lookup
    // (RequestSymmetricAddress) because padded collective memory allocations
    // can overlap at boundaries, causing FindAllocationIndex to return the
    // wrong allocation.
    for (const Buffer& buffer : buffers()) {
      RETURN_IF_ERROR(mem_requests.RequestSymmetricAllocation(
          clique_key, buffer.source_buffer.slice.index()));
      RETURN_IF_ERROR(mem_requests.RequestSymmetricAllocation(
          clique_key, buffer.destination_buffer.slice.index()));
    }
  }

  if (use_peer_memory() && clique_key.is_local()) {
    // Request peer memory exchange for destination buffers so that senders
    // can look up the target's destination address via FindPeerAddress.
    for (const Buffer& buffer : buffers()) {
      RETURN_IF_ERROR(mem_requests.RequestPeerAllocation(
          clique_key, buffer.destination_buffer.slice.index()));
    }
  }

  return absl::OkStatus();
}

P2PConfig CollectivePermuteThunk::GetP2PConfig(
    const HloCollectivePermuteInstruction* instr, int64_t replica_count,
    int64_t partition_count, bool connected_components_enabled) {
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

  const int64_t num_participants =
      config.group_mode ==
              CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          ? replica_count
          : partition_count;

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs =
      instr->source_target_pairs();

  if (connected_components_enabled) {
    // Build replica groups from connected components of the source-target pairs
    // graph. This ensures the GPU clique only includes devices that actually
    // communicate, rather than all devices in the computation.
    auto connected_components =
        SourceTargetConnectedComponents(num_participants, source_target_pairs);
    for (auto& [root, members] : connected_components) {
      ReplicaGroup& group = config.replica_groups.emplace_back();
      for (int64_t id : members) {
        group.add_replica_ids(id);
      }
    }
  } else {
    // Default: all execution instances together form one replica group.
    config.replica_groups.emplace_back();
    ReplicaGroup& replica_group = config.replica_groups.front();
    for (int i = 0; i < num_participants; ++i) {
      replica_group.add_replica_ids(i);
    }
  }

  for (const auto& [source, target] : source_target_pairs) {
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


absl::StatusOr<std::unique_ptr<CollectivePermuteThunk>>
CollectivePermuteThunk::FromProto(
    ThunkInfo thunk_info, const CollectivePermuteThunkProto& thunk_proto,
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
      std::move(buffers), thunk_proto.collectives_mode(),
      thunk_proto.connected_components_enabled());
}

absl::StatusOr<ThunkProto> CollectivePermuteThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CollectivePermuteThunkProto* thunk_proto =
      proto.mutable_collective_permute_thunk();

  for (const Buffer& buffer : buffers()) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  thunk_proto->set_collectives_mode(collectives_mode());
  thunk_proto->set_connected_components_enabled(connected_components_enabled_);

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

absl::Status CollectivePermuteThunk::RunCollective(
    const ExecuteParams& params, const GpuCliqueKey& clique_key,
    se::Stream& stream, Communicator& comm) {
  int device_ordinal = stream.parent()->device_ordinal();

  ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations,
                             std::vector<CollectiveThunk::Buffer>(buffers()),
                             config_.config.operand_element_type));
  ASSIGN_OR_RETURN(int64_t current_id,
                   GetCollectiveCurrentId(params.collective_params, config_));
  std::string device_string = GetDeviceString(*params.collective_params);

  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);

  // Remap source/target to clique-local ranks.
  ASSIGN_OR_RETURN(
      P2PConfig::SourceTargetRanks source_target_ranks,
      RemapSourceTargetToCliqueRanks(
          source_target, clique_key, *params.collective_params->device_assn,
          config_.config.group_mode,
          params.collective_params->global_device_id));

  // One-sided mode: use Put + Signal to write directly to peer symmetric
  // memory without host-side rendezvous or pointer exchange.
  // Only for local cliques — inter-node falls back to host-initiated.
  if (use_symmetric_memory() && clique_key.is_local()) {
    XLA_VLOG_DEVICE(3, device_ordinal)
        << "CollectivePermute: using one-sided mode (Put+Signal)";
    return RunOneSidedPermute(source_target_ranks, device_buffers, stream,
                              clique_key, params, comm);
  }

  // Peer-access mode: use D2D memcpy with event-based synchronization.
  if (use_peer_memory() && clique_key.is_local()) {
    ASSIGN_OR_RETURN(
        bool use_p2p_memcpy,
        params.collective_cliques->peer_access_enabled(clique_key));
    if (use_p2p_memcpy) {
      XLA_VLOG_DEVICE(3, device_ordinal)
          << "CollectivePermute: using peer-access mode (D2D memcpy)";
      return RunPeerAccessPermute(source_target_ranks, device_buffers, stream,
                                  clique_key, params);
    }
  }

  // Host-initiated mode: use standard CollectivePermute API.
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "CollectivePermute: using host-initiated mode";
  return ::xla::gpu::RunCollectivePermute(
      source_target_ranks, device_buffers, stream, comm, device_string,
      current_id, config_.config.use_symmetric_buffer);
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
// Peer destination addresses are resolved via FindPeerAddress (populated from
// RequestPeerAllocation in PrepareCollective).
static absl::Status RunPeerAccessPermute(
    const P2PConfig::SourceTargetRanks& source_target,
    const std::vector<DeviceBufferPair>& device_buffers, se::Stream& stream,
    const GpuCliqueKey& clique_key, const Thunk::ExecuteParams& params) {
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

    // Perform D2D copies from our source to target's destination.
    for (const auto& buf : device_buffers) {
      auto dst_addr = params.collective_memory->FindPeerAddress(
          clique_key, target, buf.destination_buffer);
      if (!dst_addr.has_value()) {
        return Internal("Peer address not found for target rank %d",
                        target.value());
      }
      TF_RETURN_IF_ERROR(stream.MemcpyD2D(&*dst_addr, buf.source_buffer,
                                          buf.source_buffer.size()));
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

// Performs a collective permute using one-sided Put + Signal operations.
// The sender writes directly into the receiver's symmetric memory buffer
// without any host-side rendezvous or pointer exchange.
//
// Synchronization protocol (analogous to ready/done events in peer-access):
//   1. Signal source "recv buffer ready" — prior compute has consumed the data.
//   2. Wait for target's "ready" signal, then Put data into target's buffer.
//   3. Wait for source's PutSignal — data has been written to our buffer.
//
// All signals share sig_idx=0. Per invocation each rank sends 1 Signal
// ("ready") to its source and N PutSignals (one per buffer) to its target.
// NCCL's cumulative signal counter ensures correct ordering across invocations.
static absl::Status RunOneSidedPermute(
    const P2PConfig::SourceTargetRanks& source_target,
    const std::vector<DeviceBufferPair>& device_buffers, se::Stream& stream,
    const GpuCliqueKey& clique_key, const Thunk::ExecuteParams& params,
    Communicator& comm) {
  int device_ordinal = stream.parent()->device_ordinal();

  GpuSignalDesc signal_desc(/*sig_idx=*/0, /*ctx=*/0);

  // Step 1: Signal source that our recv buffer is ready for writing. On the
  // first invocation this bootstraps the protocol; on subsequent invocation it
  // gates the source's next Put until we have consumed the previous data.
  if (source_target.source) {
    RankId source = *source_target.source;
    XLA_VLOG_DEVICE(3, device_ordinal)
        << "OneSidedPermute: Signal peer " << source << " recv buffer ready";
    RETURN_IF_ERROR(
        comm.Signal(source, signal_desc, GpuCollectives::On(stream)).Await());
  }

  // Step 2: Wait for target's "ready" signal, then Put data.
  if (source_target.target) {
    RankId target = *source_target.target;

    XLA_VLOG_DEVICE(3, device_ordinal)
        << "OneSidedPermute: WaitSignal from peer " << target
        << " (recv buffer ready)";
    RETURN_IF_ERROR(comm.WaitSignal(target, /*op_cnt=*/1, signal_desc,
                                    GpuCollectives::On(stream))
                        .Await());

    // Fuse multiple Puts into a single NCCL group to avoid per-buffer
    // kernel launch overhead.
    auto put_all = [&](GpuCommunicator* c) -> absl::Status {
      for (size_t i = 0; i < device_buffers.size(); ++i) {
        const auto& buf = device_buffers[i];
        auto [sym_mem, offset] = params.collective_memory->FindSymmetricMemory(
            clique_key, buf.destination_buffer);

        if (sym_mem == nullptr) {
          return Internal(
              "Symmetric memory not found for destination "
              "buffer[%d] (address=%p, size=%d) in clique %v",
              i, buf.destination_buffer.opaque(), buf.destination_buffer.size(),
              clique_key);
        }

        XLA_VLOG_DEVICE(3, device_ordinal)
            << "OneSidedPermute: Put " << buf.source_buffer.size()
            << " bytes to peer " << target << " at offset " << offset;

        RETURN_IF_ERROR(c->LaunchPut(buf.source_buffer, sym_mem, offset,
                                     buf.source_buffer.size(), target,
                                     GpuCollectives::On(stream)));
      }
      return absl::OkStatus();
    };

    auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
    RETURN_IF_ERROR(gpu_comm->GroupExecute(put_all).Await());
  }

  // No source: zero out destination buffers.
  if (!source_target.source) {
    XLA_VLOG_DEVICE(3, device_ordinal)
        << "OneSidedPermute: no source, zeroing destination buffers";
    for (const auto& buf : device_buffers) {
      auto dest = buf.destination_buffer;
      RETURN_IF_ERROR(stream.MemZero(&dest, dest.size()));
    }
  }

  // Step 3: Wait for source's PutSignal(s) indicating data has been written.
  if (source_target.source) {
    RankId source = *source_target.source;

    XLA_VLOG_DEVICE(3, device_ordinal)
        << "OneSidedPermute: WaitSignal from peer " << source
        << " op_cnt=" << device_buffers.size() << " (data written)";
    RETURN_IF_ERROR(comm.WaitSignal(source, /*op_cnt=*/device_buffers.size(),
                                    signal_desc, GpuCollectives::On(stream))
                        .Await());
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

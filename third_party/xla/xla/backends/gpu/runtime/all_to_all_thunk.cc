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

#include "xla/backends/gpu/runtime/all_to_all_thunk.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/rendezvous.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

AllToAllConfig GetAllToAllConfig(const HloAllToAllInstruction* instr) {
  AllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetCollectiveConfig(instr, std::nullopt);
  config.has_split_dimension = instr->split_dimension().has_value();
  return config;
}

// Contains the values that are passed between host threads with rendezvous.
struct BufferRendezvousValue {
  uint16_t rank;
  uint64_t buffer;
};

}  // namespace

AllToAllStartThunk::AllToAllStartThunk(
    ThunkInfo thunk_info, const HloAllToAllInstruction* instr,
    std::vector<CollectiveThunk::Buffer> buffers, bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kAllToAllStart, thunk_info,
                      IsGPUSyncCollective(*instr),
                      AsyncStreamKind::kCollective),
      config_(GetAllToAllConfig(instr)),
      buffers_(std::move(buffers)),
      p2p_memcpy_enabled_(p2p_memcpy_enabled) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status AllToAllStartThunk::CheckImplementable(
    const HloAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    std::optional<uint64_t> split_dim = instr->split_dimension();
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kAllToAll));
      if (split_dim &&
          !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
        return absl::UnimplementedError(absl::Substitute(
            "all-to-all split dim $0 is not the most major in input shape $1",
            *split_dim, shape.ToString(/*print_layout=*/true)));
      }
    }
    return absl::OkStatus();
  };
  return AddOpDescription<AllToAllStartThunk>(status(), instr, replica_count,
                                              partition_count);
}

/*static*/ CollectiveOpGroupMode AllToAllStartThunk::GetGroupMode(
    const HloAllToAllInstruction* instr) {
  return GetAllToAllConfig(instr).config.group_mode;
}

absl::Status AllToAllStartThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  device_count_ = params.local_device_count;
  CHECK_GT(device_count_, 0);
  VLOG(5) << "Local device count: " << device_count_;

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));

  if (is_local() && p2p_memcpy_enabled_) {
    AsyncStreamKind stream_kind = GetAsyncStreamKind();
    TF_ASSIGN_OR_RETURN(
        CommunicatorHandle comm_handle,
        GetComm(collectives, *params.collective_params,
                *params.collective_cliques, config().replica_groups,
                config().group_mode, stream_kind));
    TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm_handle.comm->NumRanks());
    se::StreamExecutor* executor = params.executor;
    {
      absl::MutexLock lock(&pointer_maps_mutex_);
      if (!receive_pointer_maps_.count(executor)) {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<se::MemoryAllocation> alloc,
            executor->HostMemoryAllocate(num_ranks * sizeof(uint64_t)));
        bool inserted =
            receive_pointer_maps_.insert({executor, std::move(alloc)}).second;
        CHECK(inserted);
      }
    }
    {
      absl::MutexLock lock(&events_mutex_);
      if (!events_.count(executor)) {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Event> event,
                            executor->CreateEvent());
        events_.insert({executor, std::move(event)});
      }
    }
    std::optional<RankId> rank =
        comm_handle.clique_key.rank(params.collective_params->global_device_id);
    TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                        GetGpuCollectives(params));
    size_t chunk_element_count = buffers_[0].element_count / num_ranks;
    TF_ASSIGN_OR_RETURN(
        std::vector<DeviceBufferPair> device_buffers,
        ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                               config_.config.operand_element_type));
    if (config_.has_split_dimension) {
      CHECK_EQ(device_buffers.size(), 1);
    }
    for (int peer = 0; peer < num_ranks; ++peer) {
      BufferRendezvousValue buffer_rendezvous_value;
      buffer_rendezvous_value.rank = rank.value().value();
      int buffer_idx = (rank.value().value() + peer) % num_ranks;
      if (config_.has_split_dimension) {
        buffer_rendezvous_value.buffer = reinterpret_cast<uint64_t>(
            collectives
                ->Slice(
                    se::DeviceMemoryBase(device_buffers[0].destination_buffer),
                    device_buffers[0].element_type,
                    buffer_idx * chunk_element_count, chunk_element_count)
                .opaque());
      } else {
        buffer_rendezvous_value.buffer = reinterpret_cast<uint64_t>(
            device_buffers[buffer_idx].destination_buffer.opaque());
      }
      TF_ASSIGN_OR_RETURN(
          std::shared_ptr<std::vector<BufferRendezvousValue>>
              rendezvous_results,
          Rendezvous<std::vector<BufferRendezvousValue>>(
              /*name=*/"memcpy all-to-all address population",
              /*key=*/comm_handle.clique_key,
              /*value=*/buffer_rendezvous_value,
              /*num_threads=*/num_ranks,
              [num_ranks](
                  absl::Span<const BufferRendezvousValue* const> values) {
                std::vector<BufferRendezvousValue> values_copy(num_ranks);
                for (const auto& value : values) {
                  values_copy.at(value->rank) = *value;
                }
                return values_copy;
              }));
      int peer_buffer_idx =
          (rank.value().value() - peer + num_ranks) % num_ranks;
      uint64_t* recv_ptr;
      {
        absl::MutexLock lock(&pointer_maps_mutex_);
        recv_ptr = reinterpret_cast<uint64_t*>(
            receive_pointer_maps_[executor]->opaque());
      }
      recv_ptr[config_.has_split_dimension ? peer_buffer_idx : peer] =
          (*rendezvous_results)[peer_buffer_idx].buffer;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> AllToAllStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));

  if (is_local() && p2p_memcpy_enabled_) {
    uint64_t* receive_pointer_map = nullptr;
    {
      absl::MutexLock lock(&pointer_maps_mutex_);
      receive_pointer_map = reinterpret_cast<uint64_t*>(
          receive_pointer_maps_[stream.parent()]->opaque());
    }
    std::optional<RankId> rank =
        comm_handle.clique_key.rank(params.collective_params->global_device_id);
    se::Event* event = nullptr;
    {
      absl::MutexLock lock(&events_mutex_);
      event = events_[stream.parent()].get();
    }
    std::vector<se::Event*> events;
    {
      absl::MutexLock lock(&events_mutex_);
      absl::c_transform(events_, std::back_inserter(events),
                        [](const auto& pair) { return pair.second.get(); });
    }
    TF_RETURN_IF_ERROR(xla::gpu::RunMemCpyAllToAll(
        collectives, config_.has_split_dimension, device_buffers, stream,
        comm_handle.comm, receive_pointer_map, comm_handle.clique_key, *rank,
        event, events));
    return false;
  }
  TF_RETURN_IF_ERROR(
      xla::gpu::RunAllToAll(collectives, config_.has_split_dimension,
                            device_buffers, stream, comm_handle.comm));
  return true;
}

AsyncStreamKind AllToAllStartThunk::GetAsyncStreamKind() const {
  if (is_local() && p2p_memcpy_enabled_) {
    return AsyncStreamKind::kMemCpyP2P;
  }
  return CollectiveThunk::GetAsyncStreamKind();
}

bool AllToAllStartThunk::is_local() const {
  for (const auto& replica_group : config_.config.replica_groups) {
    const int64_t node_id = replica_group.replica_ids().at(0) / device_count_;
    if (!absl::c_all_of(replica_group.replica_ids(),
                        [this, node_id](const int64_t rank) {
                          return rank / device_count_ == node_id;
                        })) {
      return false;
    }
  }
  return true;
}

absl::Status RunAllToAll(GpuCollectives* collectives, bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-to-all from device ordinal: " << device_ordinal
          << ", has_split_dimension: " << has_split_dimension;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  PrimitiveType element_type = buffers[0].element_type;
  int64_t element_count = buffers[0].element_count;

  // All buffers must have the same element type and count.
  bool all_buffers_match = absl::c_all_of(buffers, [&](const auto& buffer) {
    return buffer.element_type == element_type &&
           buffer.element_count == element_count;
  });

  if (!all_buffers_match) {
    return InvalidArgument(
        "All buffers must have the same element type and count");
  }

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers;
  absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers;

  if (has_split_dimension) {
    TF_RET_CHECK(element_count % num_ranks == 0)
        << "Buffer element count must be an exact multiple of the number of "
           "participants";
    size_t chunk_element_count = element_count / num_ranks;

    for (const DeviceBufferPair& buffer : buffers) {
      for (int peer = 0; peer < num_ranks; ++peer) {
        send_buffers.push_back(collectives->Slice(
            buffer.source_buffer, element_type, peer * chunk_element_count,
            chunk_element_count));
        recv_buffers.push_back(collectives->Slice(
            buffer.destination_buffer, element_type, peer * chunk_element_count,
            chunk_element_count));
      }
    }

    auto event = comm->AllToAll(
        std::move(send_buffers), std::move(recv_buffers), element_type,
        chunk_element_count, GpuCollectives::On(stream));

    tsl::BlockUntilReady(event);
    if (event.IsError()) {
      return event.GetError();
    }
  } else {
    for (const DeviceBufferPair& buffer : buffers) {
      send_buffers.push_back(buffer.source_buffer);
      recv_buffers.push_back(buffer.destination_buffer);
    }

    auto event =
        comm->AllToAll(std::move(send_buffers), std::move(recv_buffers),
                       element_type, element_count, GpuCollectives::On(stream));

    tsl::BlockUntilReady(event);
    if (event.IsError()) {
      return event.GetError();
    }
  }

  return absl::OkStatus();
}

static absl::Status SendPtrToPeer(void* ptr, RankId peer, GpuCommunicator* comm,
                                  se::Stream& stream) {
  VLOG(3) << absl::StreamFormat(
      "RecvPtrFromPeer on device #%d; peer=%d; comm=%p; stream=%p",
      stream.parent()->device_ordinal(), peer.value(), comm, &stream);

  return comm->LaunchSend(se::DeviceMemoryBase(ptr, sizeof(void*)), U64, 1,
                          peer, GpuCollectives::On(stream));
}

static absl::Status RecvPtrFromPeer(void* ptr, RankId peer,
                                    GpuCommunicator* comm, se::Stream& stream) {
  VLOG(3) << absl::StreamFormat(
      "RecvPtrFromPeer on device #%d; peer=%d; comm=%p; stream=%p",
      stream.parent()->device_ordinal(), peer.value(), comm, &stream);

  return comm->LaunchRecv(se::DeviceMemoryBase(ptr, sizeof(void*)), U64, 1,
                          peer, GpuCollectives::On(stream));
}

// Syncs the execution progress across all devices.
absl::Status SyncProgress(absl::string_view name,
                          const GpuCliqueKey& clique_key, RankId rank,
                          int64_t num_ranks, se::Stream& stream,
                          se::Event* event, std::vector<se::Event*>& events) {
  // Record event for this device.
  TF_RETURN_IF_ERROR(stream.RecordEvent(event));

  // Rendezvous to make sure that all devices have called RecordEvent before any
  // device calls WaitFor on another stream.
  std::string finish_rendezvous_key =
      absl::StrFormat("finish %s for rank %d, clique %s", name, rank.value(),
                      clique_key.ToString());
  TF_RETURN_IF_ERROR(Rendezvous(/*name=*/finish_rendezvous_key,
                                /*key=*/clique_key,
                                /*num_threads=*/num_ranks));

  // Wait for all devices to reach the corresponding events.
  for (se::Event* e : events) {
    TF_RETURN_IF_ERROR(stream.WaitFor(e));
  }
  return absl::OkStatus();
}

// TODO(b/380457503): Memcpy AllToAll implementation must be moved to
// NcclCommunicator implementation.
absl::Status RunMemCpyAllToAll(GpuCollectives* collectives,
                               bool has_split_dimension,
                               std::vector<DeviceBufferPair>& buffers,
                               se::Stream& stream, Communicator* comm,
                               uint64_t receive_pointer_map[],
                               const GpuCliqueKey& clique_key, RankId rank,
                               se::Event* event,
                               std::vector<se::Event*>& events) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing mem-copy-all-to-all from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));
  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());
  TF_RETURN_IF_ERROR(SyncProgress("before memcpy all-to-all", clique_key, rank,
                                  num_ranks, stream, event, events));

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (DeviceBufferPair& buffer : buffers) {
      TF_RET_CHECK(buffer.element_count % num_ranks == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_element_count = buffer.element_count / num_ranks;

      for (int peer = 0; peer < num_ranks; ++peer) {
        se::DeviceMemoryBase send_slice =
            collectives->Slice(buffer.source_buffer, buffer.element_type,
                               peer * chunk_element_count, chunk_element_count);
        se::DeviceMemoryBase dst_addr =
            se::DeviceMemoryBase((void*)receive_pointer_map[peer]);
        TF_RETURN_IF_ERROR(
            stream.MemcpyD2D(&dst_addr, send_slice, send_slice.size()));
      }
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_ranks)
        << "Number of inputs didn't match the number of participants.";

    for (int peer = 0; peer < num_ranks; ++peer) {
      auto buffer_idx = (rank.value() + peer) % num_ranks;
      // double buffer, exchange data with peer
      se::DeviceMemoryBase dst_addr =
          se::DeviceMemoryBase((void*)receive_pointer_map[peer]);
      TF_RETURN_IF_ERROR(
          stream.MemcpyD2D(&dst_addr, buffers[buffer_idx].source_buffer,
                           buffers[buffer_idx].source_buffer.size()));
    }
  }

  TF_RETURN_IF_ERROR(SyncProgress("after memcpy all-to-all", clique_key, rank,
                                  num_ranks, stream, event, events));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

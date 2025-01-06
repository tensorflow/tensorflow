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

#include "xla/service/gpu/runtime/nccl_all_to_all_thunk.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

NcclAllToAllConfig GetNcclAllToAllConfig(const HloAllToAllInstruction* instr) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfig(instr, std::nullopt);
  config.has_split_dimension = instr->split_dimension().has_value();
  return config;
}

}  // namespace

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, const HloAllToAllInstruction* instr,
    std::vector<NcclCollectiveThunk::Buffer> buffers, bool p2p_memcpy_enabled)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info,
                          IsSyncCollective(instr)),
      config_(GetNcclAllToAllConfig(instr)),
      buffers_(std::move(buffers)),
      p2p_memcpy_enabled_(p2p_memcpy_enabled) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status NcclAllToAllStartThunk::CheckImplementable(
    const HloAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    std::optional<uint64_t> split_dim = instr->split_dimension();
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kNcclAllToAll));
      if (split_dim &&
          !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
        return absl::UnimplementedError(absl::Substitute(
            "all-to-all split dim $0 is not the most major in input shape $1",
            *split_dim, shape.ToString(/*print_layout=*/true)));
      }
    }
    return absl::OkStatus();
  };
  return AddOpDescription<NcclAllToAllStartThunk>(
      status(), instr, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    const HloAllToAllInstruction* instr) {
  return GetNcclAllToAllConfig(instr).config.group_mode;
}

absl::Status NcclAllToAllStartThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::Initialize(params));
  device_count_ = params.local_device_count;
  CHECK_GT(device_count_, 0);
  VLOG(5) << "Local device count: " << device_count_;

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));

  if (is_local() && p2p_memcpy_enabled_) {
    const CollectiveStreamId stream_id = nccl_stream_id();
    AsyncStreamKind stream_kind = GetAsyncStreamKind();
    TF_ASSIGN_OR_RETURN(
        CommunicatorHandle comm_handle,
        GetNcclComm(collectives, *params.collective_params,
                    *params.collective_cliques, config().replica_groups,
                    config().group_mode, stream_id, stream_kind));
    TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm_handle.comm->NumRanks());
    int local_id = params.stream->parent()->device_ordinal() % num_ranks;
    {
      absl::MutexLock lock(&pointer_maps_mutex_);
      if (!send_pointer_maps_.count(local_id)) {
        for (int i = 0; i < num_ranks; ++i) {
          if (!params.stream->parent()->HostMemoryRegister(
                  &send_pointer_maps_[local_id][i], sizeof(void*))) {
            VLOG(5) << "Registering host send pointer for memcpy failed.";
          }
          if (!params.stream->parent()->HostMemoryRegister(
                  &receive_pointer_maps_[local_id][i], sizeof(void*))) {
            VLOG(5) << "Registering host recv pointer for memcpy failed.";
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status NcclAllToAllStartThunk::Cleanup(const CleanupParams& params) {
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));

  if (p2p_memcpy_enabled_) {
    const CollectiveStreamId stream_id = nccl_stream_id();
    AsyncStreamKind stream_kind = GetAsyncStreamKind();
    TF_ASSIGN_OR_RETURN(
        CommunicatorHandle comm_handle,
        GetNcclComm(collectives, *params.collective_params,
                    *params.collective_cliques, config().replica_groups,
                    config().group_mode, stream_id, stream_kind));
    TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm_handle.comm->NumRanks());

    int local_id = params.executor->device_ordinal() % num_ranks;
    {
      absl::MutexLock lock(&pointer_maps_mutex_);
      if (send_pointer_maps_.count(local_id)) {
        for (auto& [id, value] : send_pointer_maps_[local_id]) {
          if (!params.executor->HostMemoryUnregister((void*)value)) {
            VLOG(5) << "Unregistering host send pointer for memcpy failed.";
          }
        }
      }
      if (receive_pointer_maps_.count(local_id)) {
        for (auto& [id, value] : receive_pointer_maps_[local_id]) {
          if (!params.executor->HostMemoryUnregister((void*)value)) {
            VLOG(5) << "Unregistering host recv pointer for memcpy failed.";
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status NcclAllToAllStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm_handle.comm->NumRanks());

  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));

  if (is_local() && p2p_memcpy_enabled_) {
    int local_id = stream.parent()->device_ordinal() % num_ranks;
    absl::flat_hash_map<int64_t, uint64_t>* send_pointer_map = nullptr;
    absl::flat_hash_map<int64_t, uint64_t>* receive_pointer_map = nullptr;
    {
      absl::MutexLock lock(&pointer_maps_mutex_);
      send_pointer_map = &send_pointer_maps_[local_id];
      receive_pointer_map = &receive_pointer_maps_[local_id];
    }
    return xla::gpu::RunMemCpyAllToAll(collectives, config_.has_split_dimension,
                                       device_buffers, stream, comm_handle.comm,
                                       *send_pointer_map, *receive_pointer_map);
  }
  return xla::gpu::RunAllToAll(collectives, config_.has_split_dimension,
                               device_buffers, stream, comm_handle.comm);
}

AsyncStreamKind NcclAllToAllStartThunk::GetAsyncStreamKind() const {
  return (is_local() && p2p_memcpy_enabled_) ? AsyncStreamKind::kMemCpyP2P
                                             : AsyncStreamKind::kCollective;
}

bool NcclAllToAllStartThunk::is_local() const {
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
  int32_t element_count = buffers[0].element_count;

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

    return comm->AllToAll(send_buffers, recv_buffers, element_type,
                          chunk_element_count, GpuCollectives::On(stream));

  } else {
    for (const DeviceBufferPair& buffer : buffers) {
      send_buffers.push_back(buffer.source_buffer);
      recv_buffers.push_back(buffer.destination_buffer);
    }

    return comm->AllToAll(send_buffers, recv_buffers, element_type,
                          element_count, GpuCollectives::On(stream));
  }
}

static absl::Status SendPtrToPeer(void* ptr, RankId peer, Communicator* comm,
                                  se::Stream& stream) {
  VLOG(3) << absl::StreamFormat(
      "RecvPtrFromPeer on device #%d; peer=%d; comm=%p; stream=%p",
      stream.parent()->device_ordinal(), peer.value(), comm, &stream);

  return comm->Send(se::DeviceMemoryBase(ptr, sizeof(void*)), U64, 1, peer,
                    GpuCollectives::On(stream));
}

static absl::Status RecvPtrFromPeer(void* ptr, RankId peer, Communicator* comm,
                                    se::Stream& stream) {
  VLOG(3) << absl::StreamFormat(
      "RecvPtrFromPeer on device #%d; peer=%d; comm=%p; stream=%p",
      stream.parent()->device_ordinal(), peer.value(), comm, &stream);

  return comm->Recv(se::DeviceMemoryBase(ptr, sizeof(void*)), U64, 1, peer,
                    GpuCollectives::On(stream));
}

// TODO(b/380457503): Memcpy AllToAll implementation must be moved to
// NcclCommunicator implementation.
absl::Status RunMemCpyAllToAll(
    GpuCollectives* collectives, bool has_split_dimension,
    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
    Communicator* comm,
    absl::flat_hash_map<int64_t, uint64_t>& send_pointer_map,
    absl::flat_hash_map<int64_t, uint64_t>& receive_pointer_map) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing mem-copy-all-to-all from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (DeviceBufferPair& buffer : buffers) {
      TF_RET_CHECK(buffer.element_count % num_ranks == 0)
          << "Buffer was not an exact multiple of the number of participants.";

      size_t chunk_element_count = buffer.element_count / num_ranks;

      TF_RETURN_IF_ERROR(collectives->GroupStart());
      for (int peer = 0; peer < num_ranks; ++peer) {
        se::DeviceMemoryBase recv_slice =
            collectives->Slice(buffer.destination_buffer, buffer.element_type,
                               peer * chunk_element_count, chunk_element_count);
        send_pointer_map[peer] = (uint64_t)recv_slice.opaque();

        TF_RETURN_IF_ERROR(
            SendPtrToPeer(&send_pointer_map[peer], RankId(peer), comm, stream));
        TF_RETURN_IF_ERROR(RecvPtrFromPeer(&receive_pointer_map[peer],
                                           RankId(peer), comm, stream));
      }
      TF_RETURN_IF_ERROR(collectives->GroupEnd());
      TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

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

    TF_RETURN_IF_ERROR(collectives->GroupStart());
    for (int peer = 0; peer < num_ranks; ++peer) {
      send_pointer_map[peer] =
          (uint64_t)buffers[peer].destination_buffer.opaque();

      TF_RETURN_IF_ERROR(
          SendPtrToPeer(&send_pointer_map[peer], RankId(peer), comm, stream));
      TF_RETURN_IF_ERROR(RecvPtrFromPeer(&receive_pointer_map[peer],
                                         RankId(peer), comm, stream));
    }
    TF_RETURN_IF_ERROR(collectives->GroupEnd());
    TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

    for (int peer = 0; peer < num_ranks; ++peer) {
      // double buffer, exchange data with peer
      se::DeviceMemoryBase dst_addr =
          se::DeviceMemoryBase((void*)receive_pointer_map[peer]);
      TF_RETURN_IF_ERROR(stream.MemcpyD2D(&dst_addr,
                                          buffers[peer].source_buffer,
                                          buffers[peer].source_buffer.size()));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

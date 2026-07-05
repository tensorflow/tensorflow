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

#include "xla/backends/gpu/runtime/all_gather_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {
AllGatherConfig GetAllGatherConfig(const HloAllGatherInstruction* inst) {
  AllGatherConfig config;
  config.config = GetCollectiveConfig(inst, inst->use_global_device_ids());
  return config;
}

CollectiveThunk::CollectivesMode GetCollectivesMode(
    const HloAllGatherInstruction* inst) {
  auto config = inst->backend_config<GpuBackendConfig>();
  if (config.ok()) {
    return config->collective_backend_config().collectives_mode();
  }
  return DebugOptions::COLLECTIVES_PRIVATE_MEMORY;
}

absl::Status CheckImplementableInst(const HloAllGatherInstruction* inst) {
  for (HloInstruction* operand : inst->operands()) {
    const Shape& shape = operand->shape();

    RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kAllGather));

    if (!ShapeUtil::IsEffectivelyMostMajorDimension(
            shape, inst->all_gather_dimension())) {
      return absl::AbortedError(absl::StrFormat(
          "all-gather dim %u is not the most major in input shape %s",
          inst->all_gather_dimension(), shape.ToString(/*print_layout=*/true)));
    }
  }

  return absl::OkStatus();
}
}  // namespace

static absl::Status RunOneSidedAllGather(
    const std::vector<DeviceBufferPair>& device_buffers, se::Stream& stream,
    const GpuCliqueKey& clique_key, const Thunk::ExecuteParams& params,
    Communicator& comm);

AllGatherThunk::AllGatherThunk(ThunkInfo thunk_info,
                               const HloAllGatherInstruction* inst,
                               std::vector<Buffer> buffers)
    : CollectiveThunk(Thunk::kAllGather, thunk_info, std::move(buffers),
                      CommunicationId(0), GetCollectivesMode(inst)),
      config_(GetAllGatherConfig(inst)) {
  CHECK_EQ(config_.config.operand_element_type.size(), this->buffers().size());
}

AllGatherThunk::AllGatherThunk(ThunkInfo thunk_info, CollectiveConfig config,
                               std::vector<Buffer> buffers,
                               CollectivesMode collectives_mode)
    : CollectiveThunk(Thunk::kAllGather, thunk_info, std::move(buffers),
                      CommunicationId(0), collectives_mode),
      config_(AllGatherConfig{std::move(config)}) {}

absl::Status AllGatherThunk::CheckImplementable(
    const HloAllGatherInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<AllGatherThunk>(CheckImplementableInst(inst), inst,
                                          replica_count, partition_count);
}

CollectiveOpGroupMode AllGatherThunk::GetGroupMode(
    const HloAllGatherInstruction* inst) {
  return GetAllGatherConfig(inst).config.group_mode;
}

absl::Status AllGatherThunk::PrepareCollective(const PrepareParams& params,
                                               const GpuCliqueKey& clique_key) {
  if (use_symmetric_memory() && clique_key.is_local()) {
    CollectiveMemoryRequests& mem_requests = *params.collective_memory_requests;
    for (const Buffer& buffer : buffers()) {
      RETURN_IF_ERROR(mem_requests.RequestSymmetricAllocationSlice(
          clique_key, buffer.source_buffer.slice));
      RETURN_IF_ERROR(mem_requests.RequestSymmetricAllocationSlice(
          clique_key, buffer.destination_buffer.slice));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AllGatherThunk>> AllGatherThunk::FromProto(
    ThunkInfo thunk_info, const AllGatherThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  return std::make_unique<AllGatherThunk>(
      std::move(thunk_info),
      CollectiveConfig::FromProto(thunk_proto.collective_config()),
      std::move(buffers), thunk_proto.collectives_mode());
}

absl::StatusOr<ThunkProto> AllGatherThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  AllGatherThunkProto* thunk_proto = proto.mutable_all_gather_thunk();

  for (const Buffer& buffer : buffers()) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }
  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  thunk_proto->set_collectives_mode(collectives_mode());
  return proto;
}

absl::Status AllGatherThunk::RunCollective(const ExecuteParams& params,
                                           const GpuCliqueKey& clique_key,
                                           se::Stream& stream,
                                           Communicator& comm) {
  int device_ordinal = stream.parent()->device_ordinal();

  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
                                          config_.config.operand_element_type));

  if (use_symmetric_memory() && clique_key.is_local()) {
    XLA_VLOG_DEVICE(3, device_ordinal)
        << "AllGather: using one-sided mode (Put+Signal)";
    return RunOneSidedAllGather(device_buffers, stream, clique_key, params,
                                comm);
  }

  XLA_VLOG_DEVICE(3, device_ordinal) << "AllGather: using host-initiated mode";
  return xla::gpu::RunAllGather(device_buffers, stream, comm,
                                config_.config.use_symmetric_buffer);
}

absl::Status RunAllGather(std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator& comm,
                          bool use_symmetric_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal) << "Performing all-gather";
  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(&comm);
  Future<> future = gpu_comm->GroupExecute([&]() -> absl::Status {
    for (DeviceBufferPair& buffer : buffers) {
      RETURN_IF_ERROR(gpu_comm->LaunchAllGather(
          buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
          buffer.element_count, GpuCollectives::On(stream)));
    }
    return absl::OkStatus();
  });
  RETURN_IF_ERROR(future.Await());
  XLA_VLOG_DEVICE(3, device_ordinal) << "Done performing all-gather";
  return absl::OkStatus();
}

// Performs a one-sided all-gather using Put + Signal operations. Each rank
// writes its source chunk into every peer's destination buffer at the correct
// offset using symmetric memory.
//
// Synchronization protocol:
//   1. Signal all peers "recv buffer ready" (gates peers' Puts until we have
//      consumed the previous data).
//   2. WaitSignal from all peers (all destination buffers are ready).
//   3. Put our source chunk into each peer's destination buffer.
//   4. WaitSignal from all peers for their PutSignals (data has arrived).
static absl::Status RunOneSidedAllGather(
    const std::vector<DeviceBufferPair>& device_buffers, se::Stream& stream,
    const GpuCliqueKey& clique_key, const Thunk::ExecuteParams& params,
    Communicator& comm) {
  int device_ordinal = stream.parent()->device_ordinal();

  auto global_device_id = params.collective_params->global_device_id;
  auto rank = clique_key.rank(global_device_id);
  if (!rank.has_value()) {
    return Internal("Device %d not found in clique %v",
                    global_device_id.value(), clique_key);
  }

  size_t num_ranks = clique_key.devices().size();
  size_t current_rank = rank->value();

  GpuSignalDesc signal_desc(/*sig_idx=*/0, /*ctx=*/0);

  // Step 1: Signal all peers that our recv buffer is ready for writing.
  for (size_t peer = 0; peer < num_ranks; ++peer) {
    if (peer == current_rank) {
      continue;
    }
    RankId peer_rank(peer);
    XLA_VLOG_DEVICE(3, device_ordinal) << "OneSidedAllGather: Signal peer "
                                       << peer_rank << " recv buffer ready";
    RETURN_IF_ERROR(
        comm.Signal(peer_rank, signal_desc, GpuCollectives::On(stream))
            .Await());
  }

  // Step 2: Wait for all peers to signal that their recv buffers are ready.
  for (size_t peer = 0; peer < num_ranks; ++peer) {
    if (peer == current_rank) {
      continue;
    }
    RankId peer_rank(peer);
    XLA_VLOG_DEVICE(3, device_ordinal)
        << "OneSidedAllGather: WaitSignal from peer " << peer_rank
        << " (recv buffer ready)";
    RETURN_IF_ERROR(comm.WaitSignal(peer_rank, /*op_cnt=*/1, signal_desc,
                                    GpuCollectives::On(stream))
                        .Await());
  }

  // Step 3: Put our source chunk into each peer's destination buffer.
  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(&comm);
  auto put_all = [&]() -> absl::Status {
    for (size_t i = 0; i < device_buffers.size(); ++i) {
      const auto& buf = device_buffers[i];
      size_t chunk_size = buf.source_buffer.size();

      for (size_t peer = 0; peer < num_ranks; ++peer) {
        if (peer == current_rank) {
          continue;
        }
        RankId peer_rank(peer);

        auto [sym_mem, dest_offset] =
            params.collective_memory->FindSymmetricMemory(
                clique_key, buf.destination_buffer);

        if (sym_mem == nullptr) {
          return Internal(
              "Symmetric memory not found for destination "
              "buffer[%d] (address=%p, size=%d) in clique %v",
              i, buf.destination_buffer.opaque(), buf.destination_buffer.size(),
              clique_key);
        }

        size_t offset = dest_offset + current_rank * chunk_size;

        XLA_VLOG_DEVICE(3, device_ordinal)
            << "OneSidedAllGather: Put " << chunk_size << " bytes to peer "
            << peer_rank << " at offset " << offset;

        RETURN_IF_ERROR(gpu_comm->LaunchPut(buf.source_buffer, sym_mem, offset,
                                            chunk_size, peer_rank,
                                            GpuCollectives::On(stream)));
      }
    }
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(gpu_comm->GroupExecute(put_all).Await());

  // Copy our own source chunk into our destination buffer (local copy).
  for (const auto& buf : device_buffers) {
    size_t chunk_size = buf.source_buffer.size();
    size_t local_offset = current_rank * chunk_size;
    auto dest = buf.destination_buffer;
    auto local_dest = se::DeviceAddressBase(
        static_cast<char*>(dest.opaque()) + local_offset, chunk_size);
    RETURN_IF_ERROR(stream.Memcpy(&local_dest, buf.source_buffer, chunk_size));
  }

  // Step 4: Wait for all peers' PutSignals indicating data has been written
  // to our destination buffer. Each peer sends one PutSignal per buffer.
  for (size_t peer = 0; peer < num_ranks; ++peer) {
    if (peer == current_rank) {
      continue;
    }
    RankId peer_rank(peer);

    XLA_VLOG_DEVICE(3, device_ordinal)
        << "OneSidedAllGather: WaitSignal from peer " << peer_rank
        << " op_cnt=" << device_buffers.size() << " (data written)";
    RETURN_IF_ERROR(comm.WaitSignal(peer_rank, /*op_cnt=*/device_buffers.size(),
                                    signal_desc, GpuCollectives::On(stream))
                        .Await());
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu

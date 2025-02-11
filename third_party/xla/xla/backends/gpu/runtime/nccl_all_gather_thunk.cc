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

#include "xla/backends/gpu/runtime/nccl_all_gather_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/nccl_collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace impl {
NcclAllGatherConfig GetNcclAllGatherConfig(
    const HloAllGatherInstruction* inst) {
  NcclAllGatherConfig config;
  config.config = GetNcclCollectiveConfig(inst, inst->use_global_device_ids());
  return config;
}

absl::Status CheckImplementableInst(const HloAllGatherInstruction* inst) {
  for (HloInstruction* operand : inst->operands()) {
    const Shape& shape = operand->shape();

    TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kNcclAllGather));

    if (!ShapeUtil::IsEffectivelyMostMajorDimension(
            shape, inst->all_gather_dimension())) {
      return absl::AbortedError(absl::StrFormat(
          "all-gather dim %u is not the most major in input shape %s",
          inst->all_gather_dimension(), shape.ToString(/*print_layout=*/true)));
    }
  }

  return absl::OkStatus();
}
}  // namespace impl

NcclAllGatherStartThunk::NcclAllGatherStartThunk(
    ThunkInfo thunk_info, const HloAllGatherInstruction* inst,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : NcclCollectiveThunk(Thunk::kNcclAllGatherStart, thunk_info,
                          IsGPUSyncCollective(*inst)),
      config_(impl::GetNcclAllGatherConfig(inst)),
      buffers_(std::move(buffers)),
      p2p_memcpy_enabled_(p2p_memcpy_enabled) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status NcclAllGatherStartThunk::CheckImplementable(
    const HloAllGatherInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<NcclAllGatherStartThunk>(
      impl::CheckImplementableInst(inst), inst, replica_count, partition_count);
}

absl::Status NcclAllGatherStartThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::Initialize(params));
  device_count_ = params.local_device_count;
  if (should_use_memcpy()) {
    TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                        GetGpuCollectives(params));
    const CollectiveStreamId stream_id = nccl_stream_id();
    AsyncStreamKind stream_kind = GetAsyncStreamKind();
    TF_ASSIGN_OR_RETURN(
        CommunicatorHandle comm_handle,
        GetNcclComm(collectives, *params.collective_params,
                    *params.collective_cliques, config().replica_groups,
                    config().group_mode, stream_id, stream_kind));
    TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm_handle.comm->NumRanks());
    se::StreamExecutor* executor = params.executor;
    {
      absl::MutexLock lock(&pointers_mutex_);
      if (!send_pointers_.count(executor)) {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<se::MemoryAllocation> alloc,
            executor->HostMemoryAllocate(buffers_.size() * sizeof(uint64_t)));
        bool inserted =
            send_pointers_.insert({executor, std::move(alloc)}).second;
        CHECK(inserted);
        TF_ASSIGN_OR_RETURN(
            alloc, executor->HostMemoryAllocate(buffers_.size() * num_ranks *
                                                sizeof(uint64_t)));
        inserted =
            receive_pointer_maps_.insert({executor, std::move(alloc)}).second;
        CHECK(inserted);
      }
    }
  }
  return absl::OkStatus();
}

/*static*/ CollectiveOpGroupMode NcclAllGatherStartThunk::GetGroupMode(
    const HloAllGatherInstruction* inst) {
  return impl::GetNcclAllGatherConfig(inst).config.group_mode;
}

bool NcclAllGatherStartThunk::is_local() const {
  CHECK_NE(device_count_, -1);
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

absl::Status NcclAllGatherStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  if (should_use_memcpy()) {
    uint64_t* send_pointer = nullptr;
    uint64_t* receive_pointer_map = nullptr;
    {
      TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm_handle.comm->NumRanks());

      absl::MutexLock lock(&pointers_mutex_);
      CHECK_EQ(send_pointers_[stream.parent()]->size(),
               sizeof(*send_pointer) * device_buffers.size());
      send_pointer = reinterpret_cast<uint64_t*>(
          send_pointers_[stream.parent()]->opaque());
      CHECK_EQ(
          receive_pointer_maps_[stream.parent()]->size(),
          sizeof(*receive_pointer_map) * device_buffers.size() * num_ranks);
      receive_pointer_map = reinterpret_cast<uint64_t*>(
          receive_pointer_maps_[stream.parent()]->opaque());
    }
    TF_ASSIGN_OR_RETURN(
        GpuCliqueKey clique_key,
        GetGpuCliqueKey(collectives, *params.collective_params,
                        config().replica_groups, config().group_mode,
                        nccl_stream_id(), GetAsyncStreamKind()));

    std::optional<RankId> rank =
        clique_key.rank(params.collective_params->global_device_id);
    CHECK(rank.has_value());
    return xla::gpu::RunMemCpyAllGather(collectives, device_buffers, stream,
                                        comm_handle.comm, *rank, send_pointer,
                                        receive_pointer_map);
  }
  return xla::gpu::RunAllGather(collectives, device_buffers, stream,
                                comm_handle.comm);
}

absl::Status RunAllGather(GpuCollectives* collectives,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-gather from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_RETURN_IF_ERROR(collectives->GroupStart());

  for (DeviceBufferPair& buffer : buffers) {
    TF_RETURN_IF_ERROR(comm->AllGather(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count, GpuCollectives::On(stream)));
  }

  TF_RETURN_IF_ERROR(collectives->GroupEnd());

  VLOG(3) << "Done performing all-gather for ordinal: " << device_ordinal;
  return absl::OkStatus();
}

absl::Status RunMemCpyAllGather(GpuCollectives* collectives,
                                std::vector<DeviceBufferPair>& buffers,
                                se::Stream& stream, Communicator* comm,
                                RankId rank, uint64_t* send_pointer,
                                uint64_t receive_pointer_map[]) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing mem-cpy-all-gather from device ordinal: "
          << device_ordinal << ", rank: " << rank.value();
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  // Send the current device's output buffer pointers to all other devices.
  for (size_t i = 0; i < buffers.size(); ++i) {
    DeviceBufferPair& buffer = buffers[i];
    static_assert(sizeof(uint64_t) ==
                  sizeof(buffer.destination_buffer.opaque()));
    send_pointer[i] =
        reinterpret_cast<uint64_t>(buffer.destination_buffer.opaque());
  }
  se::DeviceMemoryBase send_host_buffer(send_pointer,
                                        sizeof(uint64_t) * buffers.size());
  se::DeviceMemoryBase recv_host_buffer(
      receive_pointer_map, sizeof(uint64_t) * buffers.size() * num_ranks);
  TF_RETURN_IF_ERROR(comm->AllGather(send_host_buffer, recv_host_buffer, U64,
                                     buffers.size(),
                                     GpuCollectives::On(stream)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

  // Transfer data to each peer's output buffer.
  for (int dest_rank = 0; dest_rank < num_ranks; ++dest_rank) {
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      se::DeviceMemoryBase recv_buffer_base(
          reinterpret_cast<void*>(
              receive_pointer_map[dest_rank * buffers.size() + i]),
          buffer.destination_buffer.size());
      se::DeviceMemoryBase recv_buffer = collectives->Slice(
          recv_buffer_base, buffer.element_type,
          rank.value() * buffer.element_count, buffer.element_count);
      VLOG(5) << "Sending buffer " << buffer.source_buffer.opaque()
              << " on rank " << rank.value() << " to buffer "
              << recv_buffer.opaque() << " on rank " << dest_rank;
      TF_RETURN_IF_ERROR(stream.MemcpyD2D(&recv_buffer, buffer.source_buffer,
                                          buffer.source_buffer.size()));
    }
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

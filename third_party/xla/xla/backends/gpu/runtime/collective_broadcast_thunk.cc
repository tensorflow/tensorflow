/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
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
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

CollectiveBroadcastThunk::CollectiveBroadcastThunk(ThunkInfo thunk_info,
                                                   CollectiveConfig config,
                                                   std::vector<Buffer> buffers)
    : CollectiveThunk(Thunk::kCollectiveBroadcast, thunk_info,
                      std::move(buffers)),
      config_(config) {}

CollectiveBroadcastThunk::CollectiveBroadcastThunk(
    ThunkInfo thunk_info, const HloCollectiveBroadcastInstruction* instr,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled, bool has_dynamic_root)
    : CollectiveThunk(Thunk::kCollectiveBroadcast, thunk_info,
                      std::move(buffers)),
      config_(GetCollectiveConfig(instr, std::nullopt)),
      has_dynamic_root_(has_dynamic_root) {}

/*static*/ absl::Status CollectiveBroadcastThunk::CheckImplementable(
    const HloInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  return absl::OkStatus();
}

/*static*/ CollectiveOpGroupMode CollectiveBroadcastThunk::GetGroupMode(
    const HloCollectiveBroadcastInstruction* inst) {
  return GetCollectiveConfig(inst, std::nullopt).group_mode;
}

absl::Status CollectiveBroadcastThunk::Initialize(
    const InitializeParams& params) {
  se::StreamExecutor* executor = params.executor;
  CollectiveBroadcastMetadata* cb_metadata = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    cb_metadata = per_executor_cb_metadata_[executor].get();
  }
  if (has_dynamic_root_ && cb_metadata->bcast_roots == nullptr) {
    // Last operand is the dynamic root buffer which contains actual root ranks
    cb_metadata->num_roots = buffers().size() - 1;
    ASSIGN_OR_RETURN(
        std::unique_ptr<se::MemoryAllocation> alloc,
        executor->HostMemoryAllocate(cb_metadata->num_roots * sizeof(int32_t)));
    cb_metadata->bcast_roots = std::move(alloc);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CollectiveBroadcastThunk>>
CollectiveBroadcastThunk::FromProto(
    ThunkInfo thunk_info, const CollectiveBroadcastThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  CollectiveConfig config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  return std::make_unique<CollectiveBroadcastThunk>(std::move(thunk_info),
                                                    config, std::move(buffers));
}

absl::StatusOr<ThunkProto> CollectiveBroadcastThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CollectiveBroadcastThunkProto* thunk_proto =
      proto.mutable_collective_broadcast_thunk();

  for (const Buffer& buffer : buffers()) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.ToProto();

  return proto;
}

absl::Status CollectiveBroadcastThunk::RunCollective(
    const ExecuteParams& params, const GpuCliqueKey& clique_key,
    se::Stream& stream, Communicator& comm) {
  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
                                          config_.operand_element_type));
  CollectiveBroadcastMetadata* cb_metadata = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    cb_metadata = per_executor_cb_metadata_[stream.parent()].get();
  }

  return ::xla::gpu::RunCollectiveBroadcast(device_buffers, stream, comm,
                                            cb_metadata);
}

absl::Status RunCollectiveBroadcast(std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream, Communicator& comm,
                                    CollectiveBroadcastMetadata* cb_metadata,
                                    bool has_dynamic_root) {
  if (has_dynamic_root && cb_metadata) {
    DeviceBufferPair& roots_device_buffer = buffers.back();
    CHECK(cb_metadata->bcast_roots != nullptr);
    RETURN_IF_ERROR(stream.Memcpy(cb_metadata->bcast_roots->address().opaque(),
                                  roots_device_buffer.source_buffer,
                                  roots_device_buffer.source_buffer.size()));
    // Wait for the copies to complete.
    if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
      return absl::InternalError(
          absl::StrFormat("Failed to copy dynamic roots on stream %p: %s",
                          &stream, blocked.message()));
    }
  }
  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(&comm);
  Future<> future = gpu_comm->GroupExecute([&]() -> absl::Status {
    RankId root = RankId(0);
    for (int64_t i = 0; i < buffers.size(); ++i) {
      const DeviceBufferPair& buffer = buffers[i];
      if (has_dynamic_root && cb_metadata) {
        // If dynamic root is enabled, the actual root rank is read from the
        // last buffer and can be different for each broadcast.
        int32_t* roots_ptr = reinterpret_cast<int32_t*>(
            cb_metadata->bcast_roots->address().opaque());
        root = RankId(roots_ptr[i]);
      }
      se::DeviceAddressBase src_addr = buffer.source_buffer;
      se::DeviceAddressBase dest_addr = buffer.destination_buffer;
      RETURN_IF_ERROR(gpu_comm->LaunchBroadcast(
          // Always use rank 0 since we always broadcast from the first id
          // in replica_groups
          src_addr, dest_addr, buffer.element_type, buffer.element_count, root,
          GpuCollectives::On(stream)));
    }
    return absl::OkStatus();
  });
  return future.Await();
}

}  // namespace xla::gpu

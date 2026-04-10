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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
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
#include "tsl/platform/casts.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

CollectiveBroadcastThunk::CollectiveBroadcastThunk(ThunkInfo thunk_info,
                                                   CollectiveConfig config,
                                                   std::vector<Buffer> buffers)
    : CollectiveThunk(Thunk::kCollectiveBroadcast, thunk_info),
      config_(config),
      buffers_(std::move(buffers)) {}

CollectiveBroadcastThunk::CollectiveBroadcastThunk(
    ThunkInfo thunk_info, const HloCollectiveBroadcastInstruction* instr,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kCollectiveBroadcast, thunk_info),
      config_(GetCollectiveConfig(instr, std::nullopt)),
      buffers_(std::move(buffers)) {}

/*static*/ absl::Status CollectiveBroadcastThunk::CheckImplementable(
    const HloInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  return absl::OkStatus();
}

/*static*/ CollectiveOpGroupMode CollectiveBroadcastThunk::GetGroupMode(
    const HloCollectiveBroadcastInstruction* inst) {
  return GetCollectiveConfig(inst, std::nullopt).group_mode;
}

absl::StatusOr<std::unique_ptr<CollectiveBroadcastThunk>>
CollectiveBroadcastThunk::FromProto(
    ThunkInfo thunk_info, const CollectiveBroadcastStartThunkProto& thunk_proto,
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

  CollectiveBroadcastStartThunkProto* thunk_proto =
      proto.mutable_collective_broadcast_start_thunk();

  for (const Buffer& buffer : buffers_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.ToProto();

  return proto;
}

absl::Status CollectiveBroadcastThunk::RunCollective(
    const ExecuteParams& params, const GpuCliqueKey& clique_key,
    se::Stream& stream, Communicator& comm) {
  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                                          config_.operand_element_type));
  return ::xla::gpu::RunCollectiveBroadcast(device_buffers, stream, comm);
}

absl::Status RunCollectiveBroadcast(std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream, Communicator& comm) {
  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
  Future<> future = gpu_comm->GroupExecute(
      [&buffers, &stream](GpuCommunicator* comm) -> absl::Status {
        for (auto buffer : buffers) {
          se::DeviceAddressBase src_addr = buffer.source_buffer;
          se::DeviceAddressBase dest_addr = buffer.destination_buffer;
          RETURN_IF_ERROR(comm->LaunchBroadcast(
              // Always use rank 0 since we always broadcast from the first id
              // in replica_groups
              src_addr, dest_addr, buffer.element_type, buffer.element_count,
              RankId(0), GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });
  return future.Await();
}

}  // namespace xla::gpu

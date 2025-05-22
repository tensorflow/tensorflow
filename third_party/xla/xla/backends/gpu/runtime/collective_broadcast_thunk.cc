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
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

CollectiveBroadcastStartThunk::CollectiveBroadcastStartThunk(
    ThunkInfo thunk_info, const HloCollectiveBroadcastInstruction* instr,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kCollectiveBroadcastStart, thunk_info,
                      IsGPUSyncCollective(*instr),
                      AsyncStreamKind::kCollective),
      config_(GetCollectiveConfig(instr, std::nullopt)),
      buffers_(std::move(buffers)) {}

/*static*/ absl::Status CollectiveBroadcastStartThunk::CheckImplementable(
    const HloInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  return absl::OkStatus();
}

/*static*/ CollectiveOpGroupMode CollectiveBroadcastStartThunk::GetGroupMode(
    const HloCollectiveBroadcastInstruction* inst) {
  return GetCollectiveConfig(inst, std::nullopt).group_mode;
}

absl::StatusOr<bool> CollectiveBroadcastStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_, config_.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  TF_RETURN_IF_ERROR(::xla::gpu::RunCollectiveBroadcast(
      device_buffers, stream, comm_handle.comm, collectives));
  return true;
}

absl::Status RunCollectiveBroadcast(std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream, Communicator* comm,
                                    GpuCollectives* collectives) {
  TF_ASSIGN_OR_RETURN(GpuCommunicator * gpu_comm, collectives->TryCast(comm));
  tsl::AsyncValueRef<Communicator::Event> event = gpu_comm->GroupExecute(
      [&buffers, &stream](GpuCommunicator* comm) -> absl::Status {
        for (auto buffer : buffers) {
          se::DeviceMemoryBase src_addr = buffer.source_buffer;
          se::DeviceMemoryBase dest_addr = buffer.destination_buffer;
          TF_RETURN_IF_ERROR(comm->LaunchBroadcast(
              // Always use rank 0 since we always broadcast from the first id
              // in replica_groups
              src_addr, dest_addr, buffer.element_type, buffer.element_count,
              RankId(0), GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });
  tsl::BlockUntilReady(event);
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu

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

#include "xla/service/gpu/runtime/nccl_collective_broadcast_thunk.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

NcclCollectiveBroadcastStartThunk::NcclCollectiveBroadcastStartThunk(
    ThunkInfo thunk_info, const HloCollectiveBroadcastInstruction* instr,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : NcclCollectiveThunk(Thunk::kNcclCollectiveBroadcastStart, thunk_info,
                          IsSyncCollective(instr)),
      config_(GetNcclCollectiveConfig(instr, std::nullopt)),
      buffers_(std::move(buffers)) {}

/*static*/ absl::Status NcclCollectiveBroadcastStartThunk::CheckImplementable(
    const HloInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  return absl::OkStatus();
}

/*static*/ CollectiveOpGroupMode
NcclCollectiveBroadcastStartThunk::GetGroupMode(
    const HloCollectiveBroadcastInstruction* inst) {
  return GetNcclCollectiveConfig(inst, std::nullopt).group_mode;
}

absl::Status NcclCollectiveBroadcastStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_, config_.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  return ::xla::gpu::RunCollectiveBroadcast(device_buffers, stream,
                                            comm_handle.comm, collectives);
}

absl::Status RunCollectiveBroadcast(std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream, Communicator* comm,
                                    GpuCollectives* collectives) {
  TF_RETURN_IF_ERROR(collectives->GroupStart());
  for (auto buffer : buffers) {
    se::DeviceMemoryBase src_addr = buffer.source_buffer;
    se::DeviceMemoryBase dest_addr = buffer.destination_buffer;
    TF_RETURN_IF_ERROR(comm->Broadcast(
        // Always use rank 0 since we always broadcast from the first id in
        // replica_groups
        src_addr, dest_addr, buffer.element_type, buffer.element_count, 0,
        GpuCollectives::On(stream)));
  }
  return collectives->GroupEnd();
}

}  // namespace xla::gpu

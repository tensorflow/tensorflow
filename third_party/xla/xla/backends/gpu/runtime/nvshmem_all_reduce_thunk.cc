/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/nvshmem_all_reduce_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

absl::Status RunNvshmemAllReduce(ReductionKind reduction_kind,
                                 std::vector<DeviceBufferPair>& buffers,
                                 se::Stream& stream) {
  TF_ASSIGN_OR_RETURN(auto* collectives, GetNvshmemCollectivesFromRegistry());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Communicator> nvshmem_comm,
                      collectives->CreateCommunicator());

  VLOG(3) << "Performing nvshmem all-reduce from device ordinal: "
          << *nvshmem_comm->CurrentRank();
  for (DeviceBufferPair& buffer : buffers) {
    auto future = nvshmem_comm->AllReduce(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count, reduction_kind, GpuCollectives::On(stream));
    TF_RETURN_IF_ERROR(future.Await());
  }

  return absl::OkStatus();
}

namespace impl {

absl::Status CheckNvshmemImplementableInst(const HloInstruction* inst,
                                           Thunk::Kind reduction_op) {
  for (HloInstruction* operand : inst->operands()) {
    TF_RETURN_IF_ERROR(IsValidNvshmemOperand(operand->shape(), reduction_op));
  }

  if (!MatchReductionComputation(inst->called_computations().front())
           .has_value()) {
    return absl::UnimplementedError("Unrecognized reduction computation");
  }

  return absl::OkStatus();
}

template <typename HloInstType>
CollectiveOpGroupMode GetGroupModeInst(HloInstType* inst) {
  return GetAllReduceConfigInst(inst).config.group_mode;
}

}  // namespace impl

NvshmemAllReduceReduceScatterThunkBase::NvshmemAllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, AllReduceConfig config,
    std::vector<CollectiveThunk::Buffer> buffers, bool is_sync)
    : NvshmemCollectiveThunk(kind, thunk_info, is_sync),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_element_type.size(), buffers_.size());
}

NvshmemAllReduceStartThunk::NvshmemAllReduceStartThunk(
    ThunkInfo thunk_info, const HloAllReduceInstruction* inst,
    std::vector<CollectiveThunk::Buffer> buffers, bool p2p_memcpy_enabled)
    : NvshmemAllReduceReduceScatterThunkBase(
          Thunk::kNvshmemAllReduceStart, thunk_info,
          GetAllReduceConfigInst(inst), std::move(buffers),
          IsGPUSyncCollective(*inst)) {}

absl::Status NvshmemAllReduceStartThunk::CheckImplementable(
    const HloAllReduceInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<NvshmemAllReduceStartThunk>(
      impl::CheckNvshmemImplementableInst(inst, Thunk::kNvshmemAllReduceStart),
      inst, replica_count, partition_count);
}

CollectiveOpGroupMode NvshmemAllReduceStartThunk::GetGroupMode(
    const HloAllReduceInstruction* inst) {
  return impl::GetGroupModeInst(inst);
}

absl::Status NvshmemAllReduceStartThunk::RunNvshmemCollective(
    const ExecuteParams& params, se::Stream& stream) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunNvshmemAllReduce(config_.reduction_kind, device_buffers,
                                         stream);
}

}  // namespace gpu
}  // namespace xla

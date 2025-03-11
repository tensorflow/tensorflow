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

#include "xla/backends/gpu/runtime/nccl_all_reduce_thunk.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::Status RunAllReduce(GpuCollectives* collectives,
                          ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_RETURN_IF_ERROR(collectives->GroupStart());
  for (DeviceBufferPair& buffer : buffers) {
    TF_RETURN_IF_ERROR(comm->AllReduce(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count, reduction_kind, GpuCollectives::On(stream)));
  }

  return collectives->GroupEnd();
}

namespace impl {

absl::Status CheckImplementableInst(const HloInstruction* inst,
                                    Thunk::Kind reduction_op) {
  for (HloInstruction* operand : inst->operands()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand->shape(), reduction_op));
  }

  if (!MatchReductionComputation(inst->called_computations().front())
           .has_value()) {
    return absl::UnimplementedError("Unrecognized reduction computation");
  }

  return absl::OkStatus();
}

template <typename HloInstType>
NcclAllReduceConfig GetNcclAllReduceConfigInst(HloInstType* inst) {
  std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(inst->called_computations().front());
  CHECK(reduction_kind.has_value());

  NcclAllReduceConfig config;
  config.config = GetCollectiveConfig(inst, inst->use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename HloInstType>
CollectiveOpGroupMode GetGroupModeInst(HloInstType* inst) {
  return GetNcclAllReduceConfigInst(inst).config.group_mode;
}

}  // namespace impl

NcclAllReduceReduceScatterThunkBase::NcclAllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, NcclAllReduceConfig config,
    std::vector<Buffer> buffers, bool is_sync)
    : CollectiveThunk(kind, thunk_info, is_sync, AsyncStreamKind::kCollective),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

NcclAllReduceStartThunk::NcclAllReduceStartThunk(
    ThunkInfo thunk_info, const HloAllReduceInstruction* inst,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : NcclAllReduceReduceScatterThunkBase(
          Thunk::kNcclAllReduceStart, thunk_info,
          impl::GetNcclAllReduceConfigInst(inst), std::move(buffers),
          IsGPUSyncCollective(*inst)) {}

absl::Status NcclAllReduceStartThunk::CheckImplementable(
    const HloAllReduceInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<NcclAllReduceStartThunk>(
      impl::CheckImplementableInst(inst, Thunk::kNcclAllReduceStart), inst,
      replica_count, partition_count);
}

CollectiveOpGroupMode NcclAllReduceStartThunk::GetGroupMode(
    const HloAllReduceInstruction* inst) {
  return impl::GetGroupModeInst(inst);
}

absl::Status NcclAllReduceStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  return ::xla::gpu::RunAllReduce(collectives, config_.reduction_kind,
                                  device_buffers, stream, comm_handle.comm);
}

NcclReduceScatterStartThunk::NcclReduceScatterStartThunk(
    ThunkInfo thunk_info, const HloReduceScatterInstruction* inst,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : NcclAllReduceReduceScatterThunkBase(
          Thunk::kNcclReduceScatterStart, thunk_info,
          impl::GetNcclAllReduceConfigInst(inst), std::move(buffers),
          IsGPUSyncCollective(*inst)) {}

/*static*/ absl::Status NcclReduceScatterStartThunk::CheckImplementable(
    const HloReduceScatterInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<NcclReduceScatterStartThunk>(
      impl::CheckImplementableInst(inst, Thunk::kNcclReduceScatterStart), inst,
      replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclReduceScatterStartThunk::GetGroupMode(
    const HloReduceScatterInstruction* inst) {
  return impl::GetGroupModeInst(inst);
}

absl::Status NcclReduceScatterStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  return ::xla::gpu::RunReduceScatter(collectives, config_.reduction_kind,
                                      device_buffers, stream, comm_handle.comm);
}

absl::Status RunReduceScatter(GpuCollectives* collectives,
                              ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  TF_RETURN_IF_ERROR(collectives->GroupStart());

  for (DeviceBufferPair& buffer : buffers) {
    // buffer.element_count is the source buffers element count. For
    // ncclReduceScatter, we need the destination buffers element count.
    TF_RET_CHECK(buffer.element_count % num_ranks == 0)
        << "Source buffer was not an exact multiple of the number of "
           "participants.";

    TF_RETURN_IF_ERROR(comm->ReduceScatter(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count / num_ranks, reduction_kind,
        GpuCollectives::On(stream)));
  }

  return collectives->GroupEnd();
}

}  // namespace gpu
}  // namespace xla

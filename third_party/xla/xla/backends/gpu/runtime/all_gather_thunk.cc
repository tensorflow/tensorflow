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

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace gpu {

namespace impl {
AllGatherConfig GetAllGatherConfig(const HloAllGatherInstruction* inst) {
  AllGatherConfig config;
  config.config = GetCollectiveConfig(inst, inst->use_global_device_ids());
  return config;
}

absl::Status CheckImplementableInst(const HloAllGatherInstruction* inst) {
  for (HloInstruction* operand : inst->operands()) {
    const Shape& shape = operand->shape();

    TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kAllGather));

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

AllGatherStartThunk::AllGatherStartThunk(ThunkInfo thunk_info,
                                         const HloAllGatherInstruction* inst,
                                         std::vector<Buffer> buffers,
                                         bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kAllGatherStart, thunk_info,
                      IsGPUSyncCollective(*inst), AsyncStreamKind::kCollective),
      config_(impl::GetAllGatherConfig(inst)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status AllGatherStartThunk::CheckImplementable(
    const HloAllGatherInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<AllGatherStartThunk>(
      impl::CheckImplementableInst(inst), inst, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode AllGatherStartThunk::GetGroupMode(
    const HloAllGatherInstruction* inst) {
  return impl::GetAllGatherConfig(inst).config.group_mode;
}

absl::StatusOr<bool> AllGatherStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  TF_RETURN_IF_ERROR(xla::gpu::RunAllGather(collectives, device_buffers, stream,
                                            comm_handle.comm));
  return true;
}

absl::Status RunAllGather(GpuCollectives* collectives,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-gather from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(collectives, stream.parent(), buffers, comm));
  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(comm);
  tsl::AsyncValueRef<Communicator::Event> event = gpu_comm->GroupExecute(
      [&buffers, &stream](GpuCommunicator* comm) -> absl::Status {
        for (DeviceBufferPair& buffer : buffers) {
          TF_RETURN_IF_ERROR(comm->LaunchAllGather(
              buffer.source_buffer, buffer.destination_buffer,
              buffer.element_type, buffer.element_count,
              GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });

  tsl::BlockUntilReady(event);
  VLOG(3) << "Done performing all-gather for ordinal: " << device_ordinal;
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

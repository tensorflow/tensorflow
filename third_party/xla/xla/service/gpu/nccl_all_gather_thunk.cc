/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_all_gather_thunk.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using mlir::lmhlo_gpu::AllGatherStartOp;

namespace impl {
NcclAllGatherConfig GetNcclAllGatherConfig(
    const HloAllGatherInstruction* inst) {
  NcclAllGatherConfig config;
  config.config = GetNcclCollectiveConfig(inst, inst->use_global_device_ids());
  return config;
}

NcclAllGatherConfig GetNcclAllGatherConfig(AllGatherStartOp op) {
  NcclAllGatherConfig config;
  config.config =
      GetNcclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds());
  return config;
}

absl::Status CheckImplementableInst(const HloAllGatherInstruction* inst) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());

  for (HloInstruction* operand : inst->operands()) {
    const Shape& shape = operand->shape();

    TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kNcclAllGather));

    if (!ShapeUtil::IsEffectivelyMostMajorDimension(
            shape, inst->all_gather_dimension())) {
      return tsl::errors::Unimplemented(absl::StrFormat(
          "all-gather dim %u is not the most major in input shape %s",
          inst->all_gather_dimension(), shape.ToString(/*print_layout=*/true)));
    }
  }

  return absl::OkStatus();
}

absl::Status CheckImplementable(AllGatherStartOp op) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());
  for (mlir::Value operand : op.getInputs()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand, Thunk::kNcclAllGather));
    Shape shape = GetShape(operand);
    if (!ShapeUtil::IsEffectivelyMostMajorDimension(
            shape, op.getAllGatherDimension())) {
      return tsl::errors::Unimplemented(absl::StrFormat(
          "all-gather dim %u is not the most major in input shape %s",
          op.getAllGatherDimension(), shape.ToString(/*print_layout=*/true)));
    }
  }
  return absl::OkStatus();
}
}  // namespace impl

NcclAllGatherStartThunk::NcclAllGatherStartThunk(
    ThunkInfo thunk_info, AllGatherStartOp op,
    std::vector<NcclCollectiveThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllGatherStart, thunk_info,
                          op.getIsSync()),
      config_(impl::GetNcclAllGatherConfig(op)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

NcclAllGatherStartThunk::NcclAllGatherStartThunk(
    ThunkInfo thunk_info, const HloAllGatherInstruction* inst,
    std::vector<Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllGatherStart, thunk_info,
                          inst->backend_config<GpuBackendConfig>()
                              ->collective_backend_config()
                              .is_sync()),
      config_(impl::GetNcclAllGatherConfig(inst)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status NcclAllGatherStartThunk::CheckImplementable(
    AllGatherStartOp op, int64_t replica_count, int64_t partition_count) {
  return AddOpDescription<NcclAllGatherStartThunk>(
      impl::CheckImplementable(op), op, replica_count, partition_count);
}

/*static*/ absl::Status NcclAllGatherStartThunk::CheckImplementable(
    const HloAllGatherInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<NcclAllGatherStartThunk>(
      impl::CheckImplementableInst(inst), inst, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllGatherStartThunk::GetGroupMode(
    AllGatherStartOp op) {
  return impl::GetNcclAllGatherConfig(op).config.group_mode;
}

/*static*/ CollectiveOpGroupMode NcclAllGatherStartThunk::GetGroupMode(
    const HloAllGatherInstruction* inst) {
  return impl::GetNcclAllGatherConfig(inst).config.group_mode;
}

absl::Status NcclAllGatherStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream, ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return xla::gpu::RunAllGather(
      device_buffers, stream, reinterpret_cast<NcclApi::NcclCommHandle>(comm));
}

absl::Status RunAllGather(std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, NcclApi::NcclCommHandle comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-gather from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(MaybeRegisterBuffers(device_ordinal, buffers,
                                          reinterpret_cast<ncclComm_t>(comm)));

  TF_RETURN_IF_ERROR(NcclApi::GroupStart());

  for (DeviceBufferPair& buffer : buffers) {
    TF_RETURN_IF_ERROR(NcclApi::AllGather(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count, reinterpret_cast<NcclApi::NcclCommHandle>(comm),
        &stream));
  }

  TF_RETURN_IF_ERROR(NcclApi::GroupEnd());

  VLOG(3) << "Done performing all-gather for ordinal: " << device_ordinal;
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

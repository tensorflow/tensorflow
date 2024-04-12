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

#include "xla/service/gpu/runtime/nccl_all_reduce_thunk.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::Status RunAllReduce(NcclApi* nccl_api, ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, NcclApi::NcclCommHandle comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(nccl_api, device_ordinal, buffers, comm));

  TF_RETURN_IF_ERROR(nccl_api->GroupStart());
  for (DeviceBufferPair& buffer : buffers) {
    TF_RETURN_IF_ERROR(nccl_api->AllReduce(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count, reduction_kind, comm, &stream));
  }

  return nccl_api->GroupEnd();
}

namespace {

// Generally, the reduction op should be the only operation in the block, except
// the terminator. However, if the type is bf16, the `FloatNormalization`
// pass will have converted the op to float32 and added type conversions.
// TODO(cjfj): Can we prevent the bf16 conversion for this computation?
absl::StatusOr<mlir::Operation*> FindReductionOp(mlir::Block& block) {
  TF_RET_CHECK(block.getNumArguments() == 2);
  mlir::Operation* terminator = block.getTerminator();
  TF_RET_CHECK(terminator);
  TF_RET_CHECK(terminator->getNumOperands() == 1);
  mlir::Value result = terminator->getOperand(0);
  TF_RET_CHECK(block.getArgument(0).getType() == result.getType());
  TF_RET_CHECK(block.getArgument(1).getType() == result.getType());

  mlir::Operation* result_op = result.getDefiningOp();
  TF_RET_CHECK(result_op);

  // In the bf16 case, the type conversions and op might be fused.
  if (mlir::isa<mlir::mhlo::FusionOp>(result_op)) {
    return FindReductionOp(result_op->getRegion(0).front());
  }

  // Standard case.
  if (absl::c_is_permutation(result_op->getOperands(), block.getArguments())) {
    return result_op;
  }

  // bf16 case.
  TF_RET_CHECK(mlir::isa<mlir::mhlo::ConvertOp>(result_op));
  TF_RET_CHECK(result_op->getNumOperands() == 1);
  mlir::Operation* reduction_op = result_op->getOperand(0).getDefiningOp();
  TF_RET_CHECK(reduction_op);
  TF_RET_CHECK(reduction_op->getNumOperands() == 2);
  mlir::Value operand0 = reduction_op->getOperand(0);
  mlir::Value operand1 = reduction_op->getOperand(1);
  auto operand0_op = operand0.getDefiningOp<mlir::mhlo::ConvertOp>();
  auto operand1_op = operand1.getDefiningOp<mlir::mhlo::ConvertOp>();
  TF_RET_CHECK(operand0_op);
  TF_RET_CHECK(operand1_op);
  TF_RET_CHECK(operand0_op->getNumOperands() == 1);
  TF_RET_CHECK(operand1_op->getNumOperands() == 1);
  std::array<mlir::Value, 2> operands{operand0_op->getOperand(0),
                                      operand1_op->getOperand(0)};
  TF_RET_CHECK(absl::c_is_permutation(operands, block.getArguments()));
  return reduction_op;
}

}  // namespace

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
  config.config = GetNcclCollectiveConfig(inst, inst->use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename HloInstType>
CollectiveOpGroupMode GetGroupModeInst(HloInstType* inst) {
  return GetNcclAllReduceConfigInst(inst).config.group_mode;
}

}  // namespace impl

NcclAllReduceReduceScatterThunkBase::NcclAllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, NcclApi* nccl_api,
    NcclAllReduceConfig config, std::vector<Buffer> buffers, bool is_sync)
    : NcclCollectiveThunk(kind, thunk_info, nccl_api, is_sync),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

NcclAllReduceStartThunk::NcclAllReduceStartThunk(
    ThunkInfo thunk_info, NcclApi* nccl_api,
    const HloAllReduceInstruction* inst, std::vector<Buffer> buffers)
    : NcclAllReduceReduceScatterThunkBase(
          Thunk::kNcclAllReduceStart, thunk_info, nccl_api,
          impl::GetNcclAllReduceConfigInst(inst), std::move(buffers),
          IsSyncCollective(inst)) {}

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

absl::Status NcclAllReduceStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    NcclCommHandleWrapper comm_wrapper) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunAllReduce(nccl_api(), config_.reduction_kind,
                                  device_buffers, stream,
                                  comm_wrapper.comm_handle);
}

NcclReduceScatterStartThunk::NcclReduceScatterStartThunk(
    ThunkInfo thunk_info, NcclApi* nccl_api,
    const HloReduceScatterInstruction* inst, std::vector<Buffer> buffers)
    : NcclAllReduceReduceScatterThunkBase(
          Thunk::kNcclReduceScatterStart, thunk_info, nccl_api,
          impl::GetNcclAllReduceConfigInst(inst), std::move(buffers),
          inst->backend_config<GpuBackendConfig>()
              ->collective_backend_config()
              .is_sync()) {}

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

absl::Status NcclReduceScatterStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    NcclCommHandleWrapper comm_wrapper) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunReduceScatter(nccl_api(), config_.reduction_kind,
                                      device_buffers, stream,
                                      comm_wrapper.comm_handle);
}

absl::Status RunReduceScatter(NcclApi* nccl_api, ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream,
                              NcclApi::NcclCommHandle comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(
      MaybeRegisterBuffers(nccl_api, device_ordinal, buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_participants, nccl_api->CommCount(comm));

  TF_RETURN_IF_ERROR(nccl_api->GroupStart());

  for (DeviceBufferPair& buffer : buffers) {
    // buffer.element_count is the source buffers element count. For
    // ncclReduceScatter, we need the destination buffers element count.
    TF_RET_CHECK(buffer.element_count % num_participants == 0)
        << "Source buffer was not an exact multiple of the number of "
           "participants.";

    TF_RETURN_IF_ERROR(nccl_api->ReduceScatter(
        buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
        buffer.element_count / num_participants, reduction_kind, comm,
        &stream));
  }

  return nccl_api->GroupEnd();
}

}  // namespace gpu
}  // namespace xla

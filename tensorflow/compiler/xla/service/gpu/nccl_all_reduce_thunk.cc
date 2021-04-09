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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Attempts to match computation to one of the possible cases in ReductionKind.
static absl::optional<ReductionKind> MatchReductionComputation(
    mlir::lmhlo::AllReduceOp op) {
  mlir::Block& block = op.computation().front();
  if (!llvm::hasSingleElement(block.without_terminator())) return absl::nullopt;
  // The single operation should use both block arguments and produce a single
  // result (all of the same type)
  mlir::Operation* reduction_op = &block.front();
  if (reduction_op->getNumOperands() != 2 || reduction_op->getNumResults() != 1)
    return absl::nullopt;
  mlir::BlockArgument arg0 =
      reduction_op->getOperand(0).dyn_cast<mlir::BlockArgument>();
  mlir::BlockArgument arg1 =
      reduction_op->getOperand(1).dyn_cast<mlir::BlockArgument>();
  mlir::OpResult result = reduction_op->getResult(0);
  // Both operands should be block arguments of the reduction computation block
  // and be different arguments of that block.
  if (!arg0 || !arg1 || arg0.getOwner() != &block ||
      arg1.getOwner() != &block || arg0 == arg1 ||
      arg0.getType() != arg1.getType() || arg0.getType() != result.getType())
    return absl::nullopt;
  StatusOr<HloOpcode> opcode = MhloToHloOpcode(reduction_op);
  if (!opcode.ok()) return absl::nullopt;
  // Match the operation to a reduction kind. We can represent and/or of pred as
  // min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
  PrimitiveType type = TypeToShape(result.getType()).element_type();
  if (type == PRED) {
    switch (opcode.ValueOrDie()) {
      case HloOpcode::kAnd:
        return ReductionKind::MIN;
      case HloOpcode::kOr:
        return ReductionKind::MAX;
      default:
        return absl::nullopt;
    }
  } else {
    switch (opcode.ValueOrDie()) {
      case HloOpcode::kAdd:
        return ReductionKind::SUM;
      case HloOpcode::kMultiply:
        return ReductionKind::PRODUCT;
      case HloOpcode::kMaximum:
        return ReductionKind::MAX;
      case HloOpcode::kMinimum:
        return ReductionKind::MIN;
      default:
        return absl::nullopt;
    }
  }
}

/*static*/ NcclAllReduceConfig NcclAllReduceThunk::GetNcclAllReduceConfig(
    mlir::lmhlo::AllReduceOp op) {
  auto reduction_kind = MatchReductionComputation(op);
  CHECK(reduction_kind.has_value());

  NcclAllReduceConfig config;
  config.config =
      GetNcclCollectiveConfigForMlir(op, op.use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

/*static*/ bool NcclAllReduceThunk::CanImplement(mlir::lmhlo::AllReduceOp op) {
  bool operands_are_supported =
      absl::c_all_of(op.operands(), [](mlir::Value operand) {
        Shape shape = TypeToShape(operand.getType());
        return LayoutUtil::IsDenseArray(shape) &&
               IsTypeSupportedByNccl(shape.element_type());
      });
  return operands_are_supported && MatchReductionComputation(op).has_value();
}

NcclAllReduceThunk::NcclAllReduceThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllReduceOp op,
    std::vector<NcclAllReduceThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllReduce, thunk_info),
      config_(GetNcclAllReduceConfig(op)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllReduceThunk::RunNcclCollective(const ExecuteParams& params,
                                             ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = params.stream->parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(config_.reduction_kind);

  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      params.stream->implementation()->GpuStreamMemberHack());

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const Buffer& buffer = buffers_[i];
    const void* send_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
            .opaque();
    void* recv_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
            .opaque();

    TF_ASSIGN_OR_RETURN(ncclDataType_t datatype,
                        ToNcclDataType(config_.config.operand_element_type[i]));

    VLOG(3) << absl::StreamFormat(
        "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, buffer.element_count,
        static_cast<const void*>(comm), cu_stream);

    XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(send_buffer, recv_buffer,
                                           buffer.element_count, datatype,
                                           reduce_op, comm, *cu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing all-reduce for ordinal: " << device_ordinal;
  return Status::OK();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla

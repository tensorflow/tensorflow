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
namespace {

Status RunAllReduce(const NcclAllReduceConfig& config,
                    const std::vector<NcclCollectiveThunk::Buffer>& buffers,
                    const BufferAllocations& buffer_allocations,
                    se::Stream& stream, ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(config.reduction_kind);

  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      stream.implementation()->GpuStreamMemberHack());

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers.size(); ++i) {
    const NcclCollectiveThunk::Buffer& buffer = buffers[i];
    const void* send_buffer =
        buffer_allocations.GetDeviceAddress(buffer.source_buffer).opaque();
    void* recv_buffer =
        buffer_allocations.GetDeviceAddress(buffer.destination_buffer).opaque();

    PrimitiveType element_type = config.config.operand_element_type[i];
    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(element_type));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int element_count = buffer.element_count * dtype_and_multiplier.second;

    VLOG(3) << absl::StreamFormat(
        "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, element_count, static_cast<const void*>(comm),
        cu_stream);

    XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(send_buffer, recv_buffer,
                                           element_count, dtype, reduce_op,
                                           comm, *cu_stream));
  }
  return XLA_CUDA_STATUS(ncclGroupEnd());
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

bool IsValidOperand(mlir::Value operand) {
  Shape shape = TypeToShape(operand.getType());
  return LayoutUtil::IsDenseArray(shape) &&
         IsTypeSupportedByNccl(shape.element_type());
}

// Generally, the reduction op should be the only operation in the block, except
// the terminator. However, if the type is bf16, the `BFloat16Normalization`
// pass will have converted the op to float32 and added type conversions.
// TODO(cjfj): Can we prevent the bf16 conversion for this computation?
StatusOr<mlir::Operation*> FindReductionOp(mlir::Block& block) {
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

template <typename OpT>
bool CanImplement(OpT op) {
  return absl::c_all_of(op.operands(), IsValidOperand) &&
         NcclAllReduceThunkBase::MatchAllReduceComputation(op.computation())
             .has_value();
}

template <typename OpT>
NcclAllReduceConfig GetNcclAllReduceConfig(OpT op) {
  absl::optional<ReductionKind> reduction_kind =
      NcclAllReduceThunkBase::MatchAllReduceComputation(op.computation());
  CHECK(reduction_kind.has_value());

  NcclAllReduceConfig config;
  config.config =
      GetNcclCollectiveConfigForMlir(op, op.use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename OpT>
bool IsDegenerate(OpT op, int64_t replica_count, int64_t partition_count) {
  return GetNcclCollectiveConfigForMlir(op, op.use_global_device_ids())
      .IsDegenerate(replica_count, partition_count);
}

template <typename OpT>
CollectiveOpGroupMode GetGroupMode(OpT op) {
  return GetNcclAllReduceConfig(op).config.group_mode;
}

}  // namespace impl

absl::optional<ReductionKind> NcclAllReduceThunkBase::MatchAllReduceComputation(
    mlir::Region& computation) {
  mlir::Block& block = computation.front();
  StatusOr<mlir::Operation*> reduction_op = FindReductionOp(block);
  if (!reduction_op.ok()) return absl::nullopt;
  StatusOr<HloOpcode> opcode = MhloToHloOpcode(*reduction_op);
  if (!opcode.ok()) return absl::nullopt;
  // Match the operation to a reduction kind. We can represent and/or of pred as
  // min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
  PrimitiveType type =
      TypeToShape(block.getArgument(0).getType()).element_type();
  if (type == PRED) {
    switch (opcode.ValueOrDie()) {
      case HloOpcode::kAnd:
        return ReductionKind::MIN;
      case HloOpcode::kOr:
        return ReductionKind::MAX;
      default:
        return absl::nullopt;
    }
  } else if (primitive_util::IsComplexType(type)) {
    // Only addition is supported for complex types.
    if (*opcode == HloOpcode::kAdd) {
      return ReductionKind::SUM;
    } else {
      return absl::nullopt;
    }
  } else {
    switch (*opcode) {
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

NcclAllReduceThunkBase::NcclAllReduceThunkBase(Thunk::Kind kind,
                                               ThunkInfo thunk_info,
                                               NcclAllReduceConfig config,
                                               std::vector<Buffer> buffers)
    : NcclCollectiveThunk(kind, thunk_info),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

NcclAllReduceThunk::NcclAllReduceThunk(ThunkInfo thunk_info,
                                       mlir::lmhlo::AllReduceOp op,
                                       std::vector<Buffer> buffers)
    : NcclAllReduceThunkBase(Thunk::kNcclAllReduce, thunk_info,
                             impl::GetNcclAllReduceConfig(op), buffers) {}

bool NcclAllReduceThunk::CanImplement(mlir::lmhlo::AllReduceOp op) {
  return impl::CanImplement(op);
}

bool NcclAllReduceThunk::IsDegenerate(mlir::lmhlo::AllReduceOp op,
                                      int64_t replica_count,
                                      int64_t partition_count) {
  return impl::IsDegenerate(op, replica_count, partition_count);
}

CollectiveOpGroupMode NcclAllReduceThunk::GetGroupMode(
    mlir::lmhlo::AllReduceOp op) {
  return impl::GetGroupMode(op);
}

Status NcclAllReduceThunk::RunNcclCollective(const ExecuteParams& params,
                                             ncclComm_t comm) {
  se::Stream& stream = *params.stream;
  TF_RETURN_IF_ERROR(RunAllReduce(config_, buffers_, *params.buffer_allocations,
                                  stream, comm));

  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Done performing all-reduce for ordinal: " << device_ordinal;
  return Status::OK();
}

NcclAllReduceStartThunk::NcclAllReduceStartThunk(
    ThunkInfo thunk_info, mlir::lmhlo_gpu::AllReduceStartOp op,
    std::vector<Buffer> buffers)
    : NcclAllReduceThunkBase(Thunk::kNcclAllReduceStart, thunk_info,
                             impl::GetNcclAllReduceConfig(op), buffers) {}

bool NcclAllReduceStartThunk::CanImplement(
    mlir::lmhlo_gpu::AllReduceStartOp op) {
  return impl::CanImplement(op);
}

bool NcclAllReduceStartThunk::IsDegenerate(mlir::lmhlo_gpu::AllReduceStartOp op,
                                           int64_t replica_count,
                                           int64_t partition_count) {
  return impl::IsDegenerate(op, replica_count, partition_count);
}

CollectiveOpGroupMode NcclAllReduceStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::AllReduceStartOp op) {
  return impl::GetGroupMode(op);
}

Status NcclAllReduceStartThunk::RunNcclCollective(const ExecuteParams& params,
                                                  ncclComm_t comm) {
  se::Stream& async_comms_stream = *params.async_comms_stream;
  // Wait until compute inputs are ready.
  async_comms_stream.ThenWaitFor(params.stream);

  TF_RETURN_IF_ERROR(RunAllReduce(config_, buffers_, *params.buffer_allocations,
                                  async_comms_stream, comm));

  // Create an event on the async stream for the completion of the all-reduce.
  se::Event done_event(async_comms_stream.parent());
  TF_RET_CHECK(done_event.Init());
  async_comms_stream.ThenRecordEvent(&done_event);

  int device_ordinal = async_comms_stream.parent()->device_ordinal();

  {
    absl::MutexLock lock(&mu_);
    auto result = done_events_.emplace(device_ordinal, std::move(done_event));
    TF_RET_CHECK(result.second) << "done event has not been consumed";
  }

  VLOG(3) << "Done performing all-reduce-start for ordinal: " << device_ordinal;
  return Status::OK();
}

StatusOr<se::Event> NcclAllReduceStartThunk::TakeDoneEvent(int device_ordinal) {
  absl::MutexLock lock(&mu_);
  auto it = done_events_.find(device_ordinal);
  TF_RET_CHECK(it != done_events_.end()) << "done event not found";
  // Take ownership of the event.
  se::Event done_event = std::move(it->second);
  done_events_.erase(it);
  return done_event;
}

NcclAllReduceDoneThunk::NcclAllReduceDoneThunk(
    ThunkInfo thunk_info, NcclAllReduceStartThunk& start_thunk)
    : Thunk(Thunk::kNcclAllReduceDone, thunk_info), start_thunk_(start_thunk) {}

Status NcclAllReduceDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  int device_ordinal = params.stream->parent()->device_ordinal();
  TF_ASSIGN_OR_RETURN(se::Event done_event,
                      start_thunk_.TakeDoneEvent(device_ordinal));
  params.stream->ThenWaitFor(&done_event);
  return Status::OK();
}

NcclReduceScatterThunk::NcclReduceScatterThunk(
    ThunkInfo thunk_info, mlir::lmhlo::ReduceScatterOp op,
    std::vector<NcclAllReduceThunk::Buffer> buffers)
    : NcclAllReduceThunkBase(Thunk::kNcclReduceScatter, thunk_info,
                             impl::GetNcclAllReduceConfig(op),
                             std::move(buffers)) {}

/*static*/ bool NcclReduceScatterThunk::CanImplement(
    mlir::lmhlo::ReduceScatterOp op) {
  return impl::CanImplement(op);
}

/*static*/ bool NcclReduceScatterThunk::IsDegenerate(
    mlir::lmhlo::ReduceScatterOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::IsDegenerate(op, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclReduceScatterThunk::GetGroupMode(
    mlir::lmhlo::ReduceScatterOp op) {
  return impl::GetGroupMode(op);
}

Status NcclReduceScatterThunk::RunNcclCollective(const ExecuteParams& params,
                                                 ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = params.stream->parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(config_.reduction_kind);

  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      params.stream->implementation()->GpuStreamMemberHack());

  int num_participants = 0;
  XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_participants));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const Buffer& buffer = buffers_[i];
    const void* send_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
            .opaque();
    void* recv_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
            .opaque();

    PrimitiveType element_type = config_.config.operand_element_type[i];
    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(element_type));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int element_count = buffer.element_count * dtype_and_multiplier.second;

    // buffer.element_count is the source buffers element count. For
    // ncclReduceScatter, we need the destination buffers element count.
    TF_RET_CHECK(element_count % num_participants == 0)
        << "Source buffer was not an exact multiple of the number of "
           "participants.";

    int64_t recv_count = element_count / num_participants;
    VLOG(3) << absl::StreamFormat(
        "Calling ncclReduceScatter(send_buffer=%p, recv_buffer=%p, "
        "recvcount=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, recv_count, static_cast<const void*>(comm),
        cu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclReduceScatter(send_buffer, recv_buffer,
                                               recv_count, dtype, reduce_op,
                                               comm, *cu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing reduce-scatter for ordinal: " << device_ordinal;
  return Status::OK();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla

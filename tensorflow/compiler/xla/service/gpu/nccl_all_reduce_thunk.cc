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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/type_to_shape.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#if XLA_ENABLE_XCCL
#ifndef TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/all_reduce_kernel.h"
#endif
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#endif

namespace xla {
namespace gpu {

#if XLA_ENABLE_XCCL

namespace {

struct AllReduceBuffer {
  const void* send_buffer;
  void* recv_buffer;
  ncclDataType_t dtype;
  int64_t num_elements;
};

#ifndef TENSORFLOW_USE_ROCM

// Singleton class for launching all reduce kernel.
class AllReduceLauncher {
 public:
  explicit AllReduceLauncher(se::StreamExecutor* executor)
      : cc_major_(
            executor->GetDeviceDescription().cuda_compute_capability().major),
        counter_(executor->AllocateOwnedArray<uint32_t>(1)),
        max_blocks_per_grid_([&]() -> StatusOr<int> {
          int blocks_per_sm = 0;
          XLA_CUDA_RETURN_IF_ERROR(
              cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks_per_sm,
                  GetAllReduceKernel(ncclFloat, nullptr, cc_major_),
                  kLaunchBounds, 0));
          int device = 0;
          XLA_CUDA_RETURN_IF_ERROR(cudaGetDevice(&device));
          int sm_count = 0;
          XLA_CUDA_RETURN_IF_ERROR(cudaDeviceGetAttribute(
              &sm_count, cudaDevAttrMultiProcessorCount, device));
          return blocks_per_sm * sm_count;
        }()) {}

  // Performs float sum reduction using a custom kernel that is expected to run
  // faster than default NCCL for a small number of elements. Needs to be called
  // `num_gpus` times before the kernel is launched.
  Status Add(se::Stream& stream, int num_gpus,
             const std::vector<AllReduceBuffer>& buffers) {
    assert(num_gpus > 0 && num_gpus <= kMaxNumGpus);
    assert(buffers.size() <= kMaxBuffers);

    if (counter_.is_null()) {
      return InternalError("Failed to allocate counter");
    }
    if (!max_blocks_per_grid_.ok()) {
      return max_blocks_per_grid_.status();
    }

    int index = num_pending_++;
    VLOG(3) << "xla::gpu::AllReduce::Add(stream=" << &stream
            << ", device=" << index + 1 << "/" << num_gpus
            << ", num_buffers=" << buffers.size() << ")";

    streams_[index] = &stream;
    for (size_t i = 0; i < buffers.size(); ++i) {
      buffers_[i].send_buffers[index] = buffers[i].send_buffer;
      buffers_[i].recv_buffers[index] = buffers[i].recv_buffer;
      buffers_[i].dtype = buffers[i].dtype;
      buffers_[i].num_elements = buffers[i].num_elements;
    }

    int num_complete = ++num_complete_;
    if (num_complete < num_gpus) {
      int blocks_per_grid = 1;
      int threads_per_block = 1;
      void* args[] = {&counter_};
      XLA_CUDA_RETURN_IF_ERROR(cudaLaunchKernel(
          GetSyncKernel(), blocks_per_grid, threads_per_block, args,
          /*sharedMem=*/0, se::gpu::AsGpuStreamValue(&stream)));
      return OkStatus();
    }

    num_complete_ = num_pending_ = 0;

    se::StreamExecutor* executor = stream.parent();
    if (executors_.emplace(executor).second) {
      for (int i = 0; i < num_gpus; ++i) {
        if (streams_[i] == &stream) continue;
        TF_RETURN_IF_ERROR(executor->EnablePeerAccessTo(streams_[i]->parent()));
      }
    }

    // Launch kernel for each buffer on launch stream.
    SyncFlag sync_flag = SyncFlag::SYNC_START;
    for (size_t i = 0; i < buffers.size(); ++i) {
      Buffers& buffer = buffers_[i];
      int64_t num_elements = buffer.num_elements;
      const void* kernel =
          GetAllReduceKernel(buffer.dtype, &num_elements, cc_major_);
      if (kernel == nullptr) {
        return InternalError("Unsupported ncclDataType_t: %i", buffer.dtype);
      }
      int threads_per_block = std::min<int>(kLaunchBounds, num_elements);
      int blocks_per_grid = std::min<int>(
          (num_elements + threads_per_block - 1) / threads_per_block,
          *max_blocks_per_grid_);

      void* counter = counter_.ptr()->opaque();
      if (i == buffers.size() - 1) {
        sync_flag = SyncFlag(sync_flag | SyncFlag::SYNC_END);
      }
      void* args[] = {&num_gpus,
                      &buffer.send_buffers,
                      &buffer.recv_buffers,
                      &num_elements,
                      &counter,
                      &sync_flag};
      XLA_CUDA_RETURN_IF_ERROR(cudaLaunchKernel(
          kernel, blocks_per_grid, threads_per_block, args,
          /*sharedMem=*/0, se::gpu::AsGpuStreamValue(&stream)));
      sync_flag = SyncFlag::SYNC_NONE;
    }
    return OkStatus();
  }

 private:
  struct Buffers {
    std::array<const void*, kMaxNumGpus> send_buffers;
    std::array<void*, kMaxNumGpus> recv_buffers;
    ncclDataType_t dtype;
    int64_t num_elements;
  };

  int cc_major_;
  se::ScopedDeviceMemory<uint32_t> counter_;
  StatusOr<int> max_blocks_per_grid_;
  std::atomic<int> num_pending_ = 0;
  std::atomic<int> num_complete_ = 0;
  std::array<se::Stream*, kMaxNumGpus> streams_;
  std::array<Buffers, kMaxBuffers> buffers_;
  std::unordered_set<se::StreamExecutor*> executors_;
};

#endif  // TENSORFLOW_USE_ROCM

}  // namespace

#endif  // XLA_ENABLE_XCCL

Status RunAllReduce(ReductionKind reduction_kind,
                    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm, bool allow_all_reduce_kernel) {
#if XLA_ENABLE_XCCL
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(reduction_kind);

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  // Transform vector of DeviceBufferPair to AllReduceBuffer.
  std::vector<AllReduceBuffer> all_reduce_buffers;
  all_reduce_buffers.reserve(buffers.size());
  for (DeviceBufferPair& buffer : buffers) {
    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(
                            buffer.element_type, Thunk::kNcclAllReduce));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int64_t element_count = buffer.element_count * dtype_and_multiplier.second;
    const void* send_buffer = buffer.source_buffer.opaque();
    void* recv_buffer = buffer.destination_buffer.opaque();
    all_reduce_buffers.push_back(
        AllReduceBuffer{send_buffer, recv_buffer, dtype, element_count});
  }

  // Move buffers supported by kernel to the front of all_reduce_buffers.
  auto begin = all_reduce_buffers.begin();
  if (allow_all_reduce_kernel && reduce_op == ncclSum) {
    begin = std::stable_partition(
        all_reduce_buffers.begin(), all_reduce_buffers.end(),
        [](const AllReduceBuffer& buffer) {
          return buffer.dtype == ncclFloat || buffer.dtype == ncclBfloat16 ||
                 buffer.dtype == ncclInt32;
        });
  }

  // Launch NCCL for buffers not supported by the kernel.
  if (begin != all_reduce_buffers.end()) {
    XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
    for (auto it = begin; it != all_reduce_buffers.end(); ++it) {
      VLOG(3) << absl::StreamFormat(
          "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
          "comm=%p, stream=%p)",
          it->send_buffer, it->recv_buffer, it->num_elements,
          static_cast<const void*>(comm), gpu_stream);
      XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(it->send_buffer, it->recv_buffer,
                                             it->num_elements, it->dtype,
                                             reduce_op, comm, gpu_stream));
    }
    XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());
  }
  all_reduce_buffers.erase(begin, all_reduce_buffers.end());

  // Launch kernels for the remaining buffers.
  if (!all_reduce_buffers.empty()) {
#ifndef TENSORFLOW_USE_ROCM
    int num_gpus = 0;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_gpus));
    static auto* small_all_reduce = new AllReduceLauncher(stream.parent());
    TF_RETURN_IF_ERROR(
        small_all_reduce->Add(stream, num_gpus, all_reduce_buffers));
#else
    return Unimplemented("AllReduce kernel not available on ROCm");
#endif
  }

  return OkStatus();

#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

namespace {

bool IsValidOperand(mlir::Value operand, Thunk::Kind reduction_op) {
  Shape shape = TypeToShape(operand.getType());
  return LayoutUtil::IsDenseArray(shape) &&
         IsTypeSupportedByNccl(shape.element_type(), reduction_op);
}

// Generally, the reduction op should be the only operation in the block, except
// the terminator. However, if the type is bf16, the `FloatNormalization`
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
bool CanImplement(OpT op, Thunk::Kind reduction_op) {
  return absl::c_all_of(op.getInputs(),
                        [reduction_op](mlir::Value operand) {
                          return IsValidOperand(operand, reduction_op);
                        }) &&
         NcclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
             op.getComputation())
             .has_value();
}

template <typename OpT>
NcclAllReduceConfig GetNcclAllReduceConfig(OpT op) {
  std::optional<ReductionKind> reduction_kind =
      NcclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
          op.getComputation());
  CHECK(reduction_kind.has_value());

  NcclAllReduceConfig config;
  config.config =
      GetNcclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename OpT>
bool IsDegenerate(OpT op, int64_t replica_count, int64_t partition_count) {
  return GetNcclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds())
      .IsDegenerate(replica_count, partition_count);
}

template <typename OpT>
CollectiveOpGroupMode GetGroupMode(OpT op) {
  return GetNcclAllReduceConfig(op).config.group_mode;
}

}  // namespace impl

std::optional<ReductionKind>
NcclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
    mlir::Region& computation) {
  mlir::Block& block = computation.front();
  StatusOr<mlir::Operation*> reduction_op = FindReductionOp(block);
  if (!reduction_op.ok()) return std::nullopt;
  StatusOr<HloOpcode> opcode = MhloToHloOpcode(*reduction_op);
  if (!opcode.ok()) return std::nullopt;
  // Match the operation to a reduction kind. We can represent and/or of pred as
  // min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
  PrimitiveType type =
      TypeToShape(block.getArgument(0).getType()).element_type();
  if (type == PRED) {
    switch (opcode.value()) {
      case HloOpcode::kAnd:
        return ReductionKind::MIN;
      case HloOpcode::kOr:
        return ReductionKind::MAX;
      default:
        return std::nullopt;
    }
  } else if (primitive_util::IsComplexType(type)) {
    // Only addition is supported for complex types.
    if (*opcode == HloOpcode::kAdd) {
      return ReductionKind::SUM;
    } else {
      return std::nullopt;
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
        return std::nullopt;
    }
  }
}

NcclAllReduceReduceScatterThunkBase::NcclAllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, NcclAllReduceConfig config,
    std::vector<Buffer> buffers)
    : NcclCollectiveThunk(kind, thunk_info),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllReduceThunkBase::RunAllReduce(const ExecuteParams& params,
                                            se::Stream& stream,
                                            ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunAllReduce(config_.reduction_kind, device_buffers,
                                  stream, comm,
                                  /*allow_all_reduce_kernel=*/false);
}

NcclAllReduceThunk::NcclAllReduceThunk(ThunkInfo thunk_info,
                                       mlir::lmhlo::AllReduceOp op,
                                       std::vector<Buffer> buffers)
    : NcclAllReduceThunkBase(Thunk::kNcclAllReduce, thunk_info,
                             impl::GetNcclAllReduceConfig(op),
                             std::move(buffers)) {}

bool NcclAllReduceThunk::CanImplement(mlir::lmhlo::AllReduceOp op) {
  return impl::CanImplement(op, Thunk::kNcclAllReduce);
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
  return RunAllReduce(params, *params.stream, comm);
}

NcclAllReduceStartThunk::NcclAllReduceStartThunk(
    ThunkInfo thunk_info, mlir::lmhlo_gpu::AllReduceStartOp op,
    std::vector<Buffer> buffers)
    : NcclAllReduceThunkBase(Thunk::kNcclAllReduceStart, thunk_info,
                             impl::GetNcclAllReduceConfig(op),
                             std::move(buffers)) {}

bool NcclAllReduceStartThunk::CanImplement(
    mlir::lmhlo_gpu::AllReduceStartOp op) {
  return impl::CanImplement(op, Thunk::kNcclAllReduceStart);
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
  return async_.Execute(
      [this](const ExecuteParams& params, se::Stream& stream, ncclComm_t comm) {
        return RunAllReduce(params, stream, comm);
      },
      params, comm);
}

Status NcclReduceScatterThunkBase::RunReduceScatter(const ExecuteParams& params,
                                                    se::Stream& stream,
                                                    ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunReduceScatter(config_.reduction_kind, device_buffers,
                                      stream, comm);
}

NcclReduceScatterThunk::NcclReduceScatterThunk(
    ThunkInfo thunk_info, mlir::lmhlo::ReduceScatterOp op,
    std::vector<NcclAllReduceThunk::Buffer> buffers)
    : NcclReduceScatterThunkBase(Thunk::kNcclReduceScatter, thunk_info,
                                 impl::GetNcclAllReduceConfig(op),
                                 std::move(buffers)) {}

/*static*/ bool NcclReduceScatterThunk::CanImplement(
    mlir::lmhlo::ReduceScatterOp op) {
  return impl::CanImplement(op, Thunk::kNcclReduceScatter);
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
  return RunReduceScatter(params, *params.stream, comm);
}

NcclReduceScatterStartThunk::NcclReduceScatterStartThunk(
    ThunkInfo thunk_info, mlir::lmhlo_gpu::ReduceScatterStartOp op,
    std::vector<NcclAllReduceThunk::Buffer> buffers)
    : NcclReduceScatterThunkBase(Thunk::kNcclReduceScatterStart, thunk_info,
                                 impl::GetNcclAllReduceConfig(op),
                                 std::move(buffers)) {}

/*static*/ bool NcclReduceScatterStartThunk::CanImplement(
    mlir::lmhlo_gpu::ReduceScatterStartOp op) {
  return impl::CanImplement(op, Thunk::kNcclReduceScatterStart);
}

/*static*/ bool NcclReduceScatterStartThunk::IsDegenerate(
    mlir::lmhlo_gpu::ReduceScatterStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::IsDegenerate(op, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclReduceScatterStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::ReduceScatterStartOp op) {
  return impl::GetGroupMode(op);
}

Status NcclReduceScatterStartThunk::RunNcclCollective(
    const ExecuteParams& params, ncclComm_t comm) {
  return async_.Execute(
      [this](const ExecuteParams& params, se::Stream& stream, ncclComm_t comm) {
        return RunReduceScatter(params, stream, comm);
      },
      params, comm);
}

Status RunReduceScatter(ReductionKind reduction_kind,
                        std::vector<DeviceBufferPair>& buffers,
                        se::Stream& stream, ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(reduction_kind);

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  int num_participants = 0;
  XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_participants));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers.size(); ++i) {
    DeviceBufferPair& buffer = buffers[i];
    const void* send_buffer = buffer.source_buffer.opaque();
    void* recv_buffer = buffer.destination_buffer.opaque();

    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(
                            buffer.element_type, Thunk::kNcclReduceScatter));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int64_t element_count = buffer.element_count * dtype_and_multiplier.second;

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
        gpu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclReduceScatter(send_buffer, recv_buffer,
                                               recv_count, dtype, reduce_op,
                                               comm, gpu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing reduce-scatter for ordinal: " << device_ordinal;
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla

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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/shape_util.h"

#if XLA_ENABLE_XCCL
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

namespace impl {
template <typename OpT>
NcclAllToAllConfig GetNcclAllToAllConfig(OpT op) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfigForMlir(op, std::nullopt);
  config.has_split_dimension = op.getSplitDimension().has_value();
  return config;
}

template <typename OpT>
bool CanImplement(OpT op) {
  return absl::c_all_of(op.getInputs(), [&op](mlir::Value operand) {
    Shape shape = GetShape(operand);
    return LayoutUtil::IsDenseArray(shape) &&
           IsTypeSupportedByNccl(shape.element_type(), Thunk::kNcclAllToAll) &&
           (!op.getSplitDimension() ||
            LayoutUtil::MinorToMajor(shape).back() == *op.getSplitDimension());
  });
}
}  // namespace impl

NcclAllToAllThunkBase::NcclAllToAllThunkBase(Kind kind, ThunkInfo thunk_info,
                                             NcclAllToAllConfig config,
                                             std::vector<Buffer> buffers)
    : NcclCollectiveThunk(kind, thunk_info),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllToAllThunkBase::RunAllToAll(const ExecuteParams& params,
                                          se::Stream& stream, ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return xla::gpu::RunAllToAll(config_.has_split_dimension, device_buffers,
                               stream, comm);
}

NcclAllToAllThunk::NcclAllToAllThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllToAllOp op,
    std::vector<NcclAllToAllThunk::Buffer> buffers)
    : NcclAllToAllThunkBase(Thunk::kNcclAllToAll, thunk_info,
                            impl::GetNcclAllToAllConfig(op),
                            std::move(buffers)) {}

/*static*/ bool NcclAllToAllThunk::CanImplement(mlir::lmhlo::AllToAllOp op) {
  return impl::CanImplement(op);
}

/*static*/ bool NcclAllToAllThunk::IsDegenerate(mlir::lmhlo::AllToAllOp op,
                                                int64_t replica_count,
                                                int64_t partition_count) {
  return impl::GetNcclAllToAllConfig(op).config.IsDegenerate(replica_count,
                                                             partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllToAllThunk::GetGroupMode(
    mlir::lmhlo::AllToAllOp op) {
  return impl::GetNcclAllToAllConfig(op).config.group_mode;
}

Status NcclAllToAllThunk::RunNcclCollective(const ExecuteParams& params,
                                            ncclComm_t comm) {
  return RunAllToAll(params, *params.stream, comm);
}

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, mlir::lmhlo_gpu::AllToAllStartOp op,
    std::vector<NcclAllToAllThunk::Buffer> buffers)
    : NcclAllToAllThunkBase(Thunk::kNcclAllToAllStart, thunk_info,
                            impl::GetNcclAllToAllConfig(op),
                            std::move(buffers)) {}

/*static*/ bool NcclAllToAllStartThunk::CanImplement(
    mlir::lmhlo_gpu::AllToAllStartOp op) {
  return impl::CanImplement(op);
}

/*static*/ bool NcclAllToAllStartThunk::IsDegenerate(
    mlir::lmhlo_gpu::AllToAllStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::GetNcclAllToAllConfig(op).config.IsDegenerate(replica_count,
                                                             partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::AllToAllStartOp op) {
  return impl::GetNcclAllToAllConfig(op).config.group_mode;
}

Status NcclAllToAllStartThunk::RunNcclCollective(const ExecuteParams& params,
                                                 ncclComm_t comm) {
  return async_.Execute(
      [this](const ExecuteParams& params, se::Stream& stream, ncclComm_t comm) {
        return RunAllToAll(params, stream, comm);
      },
      params, comm);
}

NcclAllToAllDoneThunk::NcclAllToAllDoneThunk(
    ThunkInfo thunk_info, NcclCollectiveThunk::AsyncExecutor& async)
    : NcclCollectiveDoneThunk(Thunk::kNcclAllToAllDone, thunk_info, async) {}

Status RunAllToAll(bool has_split_dimension,
                   std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                   ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-to-all from device ordinal: " << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  int num_participants;
  XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_participants));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(
                              buffer.element_type, Thunk::kNcclAllToAll));
      ncclDataType_t dtype = dtype_and_multiplier.first;
      int64_t element_count =
          buffer.element_count * dtype_and_multiplier.second;

      TF_RET_CHECK(element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_elements = element_count / num_participants;
      size_t chunk_bytes = chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(
                                                buffer.element_type);

      for (int rank = 0; rank < num_participants; ++rank) {
        XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer + rank * chunk_bytes,
                                          chunk_elements, dtype, rank, comm,
                                          gpu_stream));
        XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer + rank * chunk_bytes,
                                          chunk_elements, dtype, rank, comm,
                                          gpu_stream));
      }
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(
                              buffer.element_type, Thunk::kNcclAllToAll));
      ncclDataType_t dtype = dtype_and_multiplier.first;
      int64_t element_count =
          buffer.element_count * dtype_and_multiplier.second;

      XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer, element_count, dtype,
                                        /*rank=*/i, comm, gpu_stream));
      XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer, element_count, dtype,
                                        /*rank=*/i, comm, gpu_stream));
    }
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing all-to-all for ordinal: " << device_ordinal;
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla

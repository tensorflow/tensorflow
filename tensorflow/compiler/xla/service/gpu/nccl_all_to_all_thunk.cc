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
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

/*static*/ NcclAllToAllConfig NcclAllToAllThunk::GetNcclAllToAllConfig(
    mlir::lmhlo::AllToAllOp op) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfigForMlir(op, absl::nullopt);
  config.has_split_dimension = op.split_dimension().hasValue();
  return config;
}

/*static*/ bool NcclAllToAllThunk::CanImplement(mlir::lmhlo::AllToAllOp op) {
  bool operands_are_supported =
      absl::c_all_of(op.operands(), [](mlir::Value operand) {
        Shape shape = TypeToShape(operand.getType());
        return LayoutUtil::IsDenseArray(shape) &&
               IsTypeSupportedByNccl(shape.element_type());
      });
  return op.split_dimension().getValueOr(0) == 0 && operands_are_supported;
}

NcclAllToAllThunk::NcclAllToAllThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllToAllOp op,
    std::vector<NcclAllToAllThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllToAll, thunk_info),
      config_(GetNcclAllToAllConfig(op)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllToAllThunk::RunNcclCollective(const ExecuteParams& params,
                                            ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = params.stream->parent()->device_ordinal();
  VLOG(3) << "Performing all-to-all from device ordinal: " << device_ordinal;

  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      params.stream->implementation()->GpuStreamMemberHack());

  int num_participants;
  XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_participants));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (config_.has_split_dimension) {
    for (size_t i = 0; i < buffers_.size(); ++i) {
      const Buffer& buffer = buffers_[i];
      const uint8* send_buffer = static_cast<uint8*>(
          params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
              .opaque());
      uint8* recv_buffer = static_cast<uint8*>(
          params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
              .opaque());

      PrimitiveType element_type = config_.config.operand_element_type[i];
      TF_ASSIGN_OR_RETURN(ncclDataType_t datatype,
                          ToNcclDataType(element_type));

      TF_RET_CHECK(buffer.element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_elements = buffer.element_count / num_participants;
      size_t chunk_bytes =
          chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);

      for (int rank = 0; rank < num_participants; ++rank) {
        XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer + rank * chunk_bytes,
                                          chunk_elements, datatype, rank, comm,
                                          *cu_stream));
        XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer + rank * chunk_bytes,
                                          chunk_elements, datatype, rank, comm,
                                          *cu_stream));
      }
    }
  } else {
    TF_RET_CHECK(buffers_.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (size_t i = 0; i < buffers_.size(); ++i) {
      const Buffer& buffer = buffers_[i];
      const uint8* send_buffer = static_cast<uint8*>(
          params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
              .opaque());
      uint8* recv_buffer = static_cast<uint8*>(
          params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
              .opaque());

      PrimitiveType element_type = config_.config.operand_element_type[i];
      TF_ASSIGN_OR_RETURN(ncclDataType_t datatype,
                          ToNcclDataType(element_type));

      XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer, buffer.element_count,
                                        datatype, /*rank=*/i, comm,
                                        *cu_stream));
      XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer, buffer.element_count,
                                        datatype, /*rank=*/i, comm,
                                        *cu_stream));
    }
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing all-to-all for ordinal: " << device_ordinal;
  return Status::OK();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla

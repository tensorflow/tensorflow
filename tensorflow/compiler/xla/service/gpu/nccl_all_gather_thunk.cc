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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#endif
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

static NcclAllGatherConfig GetNcclAllGatherConfig(mlir::lmhlo::AllGatherOp op,
                                                  int64 replica_count) {
  NcclAllGatherConfig config;
  config.config = GetNcclCollectiveConfigForMlir(op, replica_count);
  return config;
}

/*static*/ bool NcclAllGatherThunk::CanImplement(const HloInstruction* hlo) {
  auto operands_are_supported = [hlo]() {
    return absl::c_all_of(hlo->operands(), [](HloInstruction* operand) {
      return LayoutUtil::IsDenseArray(operand->shape()) &&
             ToNcclDataType(operand->shape().element_type()).ok();
    });
  };
  return (Cast<HloAllGatherInstruction>(hlo)->all_gather_dimension() == 0) &&
         operands_are_supported();
}

/*static*/ bool NcclAllGatherThunk::CanImplement(mlir::lmhlo::AllGatherOp op) {
  bool operands_are_supported =
      absl::c_all_of(op.operands(), [](mlir::Value operand) {
        Shape shape = TypeToShape(operand.getType());
        return LayoutUtil::IsDenseArray(shape) &&
               ToNcclDataType(shape.element_type()).ok();
      });
  return op.all_gather_dimension() == 0 && operands_are_supported;
}

NcclAllGatherThunk::NcclAllGatherThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllGatherOp op, int64 replica_count,
    std::vector<NcclAllGatherThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllGather, thunk_info),
      config_(GetNcclAllGatherConfig(op, replica_count)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllGatherThunk::RunNcclCollective(const ExecuteParams& params,
                                             ncclComm_t comm) {
  int device_ordinal = params.stream->parent()->device_ordinal();
  VLOG(3) << "Performing all-gather from device ordinal: " << device_ordinal;

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
        "Calling ncclAllGather(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, buffer.element_count,
        static_cast<const void*>(comm), cu_stream);

    XLA_CUDA_RETURN_IF_ERROR(ncclAllGather(send_buffer, recv_buffer,
                                           buffer.element_count, datatype, comm,
                                           *cu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing all-gather for ordinal: " << device_ordinal;
  return Status::OK();
}

const NcclCollectiveConfig& NcclAllGatherThunk::config() const {
  return config_.config;
}

}  // namespace gpu
}  // namespace xla

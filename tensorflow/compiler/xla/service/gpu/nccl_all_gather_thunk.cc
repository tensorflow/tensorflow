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
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

#if XLA_ENABLE_XCCL
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

/*static*/ NcclAllGatherConfig NcclAllGatherThunk::GetNcclAllGatherConfig(
    mlir::lmhlo::AllGatherOp op) {
  NcclAllGatherConfig config;
  config.config =
      GetNcclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds());
  return config;
}

/*static*/ bool NcclAllGatherThunk::CanImplement(mlir::lmhlo::AllGatherOp op) {
  return absl::c_all_of(op.getInputs(), [&](mlir::Value operand) {
    Shape shape = GetShape(operand);
    return LayoutUtil::IsDenseArray(shape) &&
           IsTypeSupportedByNccl(shape.element_type(), Thunk::kNcclAllGather) &&
           LayoutUtil::MinorToMajor(shape).back() == op.getAllGatherDimension();
  });
}

NcclAllGatherThunk::NcclAllGatherThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllGatherOp op,
    std::vector<NcclAllGatherThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllGather, thunk_info),
      config_(GetNcclAllGatherConfig(op)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status RunAllGather(std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-gather from device ordinal: " << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers.size(); ++i) {
    DeviceBufferPair& buffer = buffers[i];
    const void* send_buffer = buffer.source_buffer.opaque();
    void* recv_buffer = buffer.destination_buffer.opaque();

    PrimitiveType element_type = buffer.element_type;
    TF_ASSIGN_OR_RETURN(
        auto dtype_and_multiplier,
        ToNcclDataTypeAndCountMultiplier(element_type, Thunk::kNcclAllGather));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int64_t element_count = buffer.element_count * dtype_and_multiplier.second;

    VLOG(3) << absl::StreamFormat(
        "Calling ncclAllGather(send_buffer=%p, recv_buffer=%p, sendcount=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, element_count, static_cast<const void*>(comm),
        gpu_stream);

    XLA_CUDA_RETURN_IF_ERROR(ncclAllGather(
        send_buffer, recv_buffer, element_count, dtype, comm, gpu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing all-gather for ordinal: " << device_ordinal;
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

Status NcclAllGatherThunk::RunNcclCollective(const ExecuteParams& params,
                                             ncclComm_t comm) {
  se::Stream& stream = *params.stream;
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return RunAllGather(device_buffers, stream, comm);
}

}  // namespace gpu
}  // namespace xla

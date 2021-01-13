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
#include "tensorflow/compiler/xla/service/hlo_computation.h"
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

NcclAllReduceConfig GetNcclAllReduceConfig(const HloInstruction* hlo,
                                           int64 replica_count) {
  auto reduction_kind = MatchReductionComputation(hlo->to_apply());
  CHECK(reduction_kind.has_value());

  NcclAllReduceConfig config;
  config.config = GetNcclCollectiveConfig(hlo, replica_count);
  config.reduction_kind = reduction_kind.value();
  return config;
}

/*static*/ bool NcclAllReduceThunk::CanImplement(const HloInstruction* hlo) {
  auto operands_are_supported = [hlo]() {
    return absl::c_all_of(hlo->operands(), [](HloInstruction* operand) {
      return LayoutUtil::IsDenseArray(operand->shape()) &&
             ToNcclDataType(operand->shape().element_type()).ok();
    });
  };
  return MatchReductionComputation(hlo->to_apply()).has_value() &&
         hlo->IsCrossReplicaAllReduce() && operands_are_supported();
}

NcclAllReduceThunk::NcclAllReduceThunk(
    ThunkInfo thunk_info, NcclAllReduceConfig config,
    std::vector<NcclAllReduceThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllReduce, thunk_info),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllReduceThunk::RunNcclCollective(const ExecuteParams& params,
                                             ncclComm_t comm) {
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
}

const NcclCollectiveConfig& NcclAllReduceThunk::config() const {
  return config_.config;
}

}  // namespace gpu
}  // namespace xla

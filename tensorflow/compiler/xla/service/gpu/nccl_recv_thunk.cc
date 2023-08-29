/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/nccl_recv_thunk.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"

#if XLA_ENABLE_XCCL
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

using mlir::lmhlo::RecvOp;

namespace impl {

NcclP2PConfig GetNcclP2PConfig(RecvOp op, int64_t replica_count,
                               int64_t partition_count) {
  return GetNcclP2PConfigForSendRecv(op, replica_count, partition_count);
}

Status CheckImplementable(RecvOp op) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());
  return IsValidOperand(op.getOutputs()[0], Thunk::kNcclSend);
}

}  // namespace impl

NcclRecvThunk::NcclRecvThunk(ThunkInfo thunk_info, RecvOp op,
                             int64_t replica_count, int64_t partition_count,
                             const Buffer& buffer)
    : NcclCollectiveThunk(Thunk::kNcclRecv, thunk_info, /*is_sync=*/false),
      config_(GetNcclP2PConfig(op, replica_count, partition_count)),
      buffer_(buffer) {}

/*static*/ NcclP2PConfig NcclRecvThunk::GetNcclP2PConfig(
    RecvOp op, int64_t replica_count, int64_t partition_count) {
  return impl::GetNcclP2PConfig(op, replica_count, partition_count);
}

/*static*/ Status NcclRecvThunk::CheckImplementable(RecvOp op,
                                                    int64_t replica_count,
                                                    int64_t partition_count) {
  return AddOpDescription<NcclRecvThunk>(impl::CheckImplementable(op), op,
                                         replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclRecvThunk::GetGroupMode(RecvOp op) {
  return GetGroupModeForSendRecv(op);
}

Status NcclRecvThunk::RunNcclCollective(const ExecuteParams& params,
                                        se::Stream& stream, ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, {buffer_},
                             config_.config.operand_element_type));
  TF_RET_CHECK(device_buffers.size() == 1) << "Expected one buffer pair.";

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.nccl_params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID current_logical_id,
      params.nccl_params.device_assn->LogicalIdForDevice(global_device_id));
  const int64_t current_id =
      config_.config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string = GetDeviceString(params.nccl_params);

  const NcclP2PConfig::SourceTargetMapEntry source_target =
      NcclP2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);

  return ::xla::gpu::RunRecv(source_target, device_buffers[0], stream, comm,
                             device_string, current_id);
}

Status RunRecv(NcclP2PConfig::SourceTargetMapEntry source_target,
               DeviceBufferPair& buffer, se::Stream& stream, ncclComm_t comm,
               absl::string_view device_string, int64_t current_id) {
#if XLA_ENABLE_XCCL
  // Determine the source IDs for this instance. The source ID is the ID for
  // the peer that will copy its data to this instance. If there is no source,
  // just memzero() the destination buffer.
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing Recv from device ordinal: " << device_ordinal
          << "current_id " << current_id;

  const std::optional<int64_t> source_id = source_target.source;
  se::DeviceMemoryBase dest_addr = buffer.destination_buffer;

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d", device_string,
                                current_id, source_id.value_or(-1));

  TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                      ToNcclDataTypeAndCountMultiplier(
                          buffer.element_type, Thunk::kNcclCollectivePermute));
  ncclDataType_t dtype = dtype_and_multiplier.first;
  int64_t element_count = buffer.element_count * dtype_and_multiplier.second;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  // Receive data from the source peer to the destination buffer.
  if (source_id) {
    VLOG(3) << absl::StreamFormat(
        "%s : Calling ncclRecv(recvbuff=%p, count=%d, peer=%d comm=%p, "
        "stream=%p)",
        device_string, dest_addr.opaque(), element_count, *source_id,
        static_cast<const void*>(comm), gpu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclRecv(dest_addr.opaque(), element_count, dtype,
                                      *source_id, comm, gpu_stream));
  } else {
    // If there is no source peer, i.e. no sender to this instance, zero out
    // the destination buffer.
    VLOG(3) << absl::StreamFormat("%s : collective-Permute: Issuing MemZero",
                                  device_string);
    stream.ThenMemZero(&dest_addr, dest_addr.size());
  }
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla

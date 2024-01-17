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

#include "xla/service/gpu/nccl_recv_thunk.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

using mlir::lmhlo::RecvOp;

namespace impl {

NcclP2PConfig GetNcclP2PConfig(RecvOp op, int64_t replica_count,
                               int64_t partition_count) {
  return GetNcclP2PConfigForSendRecv(op, replica_count, partition_count);
}

absl::Status CheckImplementable(RecvOp op) {
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

/*static*/ absl::Status NcclRecvThunk::CheckImplementable(
    RecvOp op, int64_t replica_count, int64_t partition_count) {
  return AddOpDescription<NcclRecvThunk>(impl::CheckImplementable(op), op,
                                         replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclRecvThunk::GetGroupMode(RecvOp op) {
  return GetGroupModeForSendRecv(op);
}

absl::Status NcclRecvThunk::RunNcclCollective(const ExecuteParams& params,
                                              se::Stream& stream,
                                              ncclComm_t comm) {
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

absl::Status RunRecv(NcclP2PConfig::SourceTargetMapEntry source_target,
                     DeviceBufferPair& buffer, se::Stream& stream,
                     ncclComm_t comm, absl::string_view device_string,
                     int64_t current_id) {
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

  // Receive data from the source peer to the destination buffer.
  if (source_id) {
    TF_RETURN_IF_ERROR(NcclApi::Recv(
        dest_addr, buffer.element_type, buffer.element_count, *source_id,
        reinterpret_cast<NcclApi::NcclCommHandle>(comm), &stream));

  } else {
    // If there is no source peer, i.e. no sender to this instance, zero out
    // the destination buffer.
    VLOG(3) << absl::StreamFormat("%s : collective-Permute: Issuing MemZero",
                                  device_string);
    stream.ThenMemZero(&dest_addr, dest_addr.size());
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

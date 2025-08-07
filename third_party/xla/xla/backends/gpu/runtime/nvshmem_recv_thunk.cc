/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/nvshmem_recv_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

NvshmemRecvThunk::NvshmemRecvThunk(
    ThunkInfo thunk_info, const HloRecvInstruction* instr,
    int64_t replica_count, int64_t partition_count,
    const CollectiveThunk::Buffer& buffer,
    std::shared_ptr<NvshmemBufferAddresses> buffer_addresses)
    : NvshmemCollectiveThunk(Thunk::kNvshmemRecv, thunk_info,
                             IsGPUSyncCollective(*instr)),
      config_(GetP2PConfigForSendRecv(instr, instr->shape().tuple_shapes(0),
                                      replica_count, partition_count)),
      buffer_(buffer),
      execution_counters_(config_.validation_kind ==
                                  P2PConfig::ValidationKind::kConditional
                              ? std::make_unique<ExecutionCounters>()
                              : nullptr),
      hlo_name_(instr->name()),
      buffer_addresses_(std::move(buffer_addresses)) {}

absl::Status NvshmemRecvThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(NvshmemCollectiveThunk::Initialize(params));
  if (execution_counters_) {
    TF_RETURN_IF_ERROR(execution_counters_->Initialize(
        params.executor, params.collective_params->run_id));
  }
  return absl::OkStatus();
}

absl::Status NvshmemRecvThunk::RunNvshmemCollective(const ExecuteParams& params,
                                                    se::Stream& stream) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, {buffer_},
                             config_.config.operand_element_type));
  TF_RET_CHECK(device_buffers.size() == 1) << "Expected one buffer pair.";

  GlobalDeviceId global_device_id = params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID current_logical_id,
                      params.collective_params->device_assn->LogicalIdForDevice(
                          global_device_id));
  const int64_t current_id =
      config_.config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string =
      CollectiveThunk::GetDeviceString(*params.collective_params);

  int device_ordinal = stream.parent()->device_ordinal();
  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);
  DeviceBufferPair& buffer = device_buffers[0];

  VLOG(3) << "Performing Recv from device ordinal: " << device_ordinal
          << ", global_id: " << global_device_id
          << ", current_id: " << current_id << ", group mode: "
          << CollectiveOpGroupModeToString(config_.config.group_mode) << " ("
          << hlo_name_ << ")";

  const std::optional<int64_t> source_id = source_target.source;

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d", device_string,
                                current_id, source_id.value_or(-1));

  // For receive operations, we need to register the buffer with NVSHMEM
  // and store it in the global struct for the send operation to use
  if (source_id.value_or(-1) == -1) {
    VLOG(3) << "Storing destination device " << device_ordinal << " and buffer "
            << buffer.destination_buffer.opaque();
    buffer_addresses_->StoreNvshmemPtr(device_ordinal,
                                       buffer.destination_buffer.opaque());
  }

  if (!source_id) {
    VLOG(3) << "No source ID found, skipping Recv operation";
    return absl::OkStatus();
  }

  bool should_run =
      config_.validation_kind != P2PConfig::ValidationKind::kInvalid;

  if (config_.validation_kind == P2PConfig::ValidationKind::kConditional) {
    se::StreamExecutor* executor = params.stream->parent();
    TF_ASSIGN_OR_RETURN(int64_t* counter,
                        execution_counters_->GetCounter(
                            executor, params.collective_params->run_id));
    auto it = config_.source_target_to_bounds.find(
        std::make_pair(*source_target.source, current_id));
    if (it == config_.source_target_to_bounds.end()) {
      return absl::InternalError("Missing bounds for conditional Recv");
    }
    if (*counter < it->second.first || *counter > it->second.second) {
      should_run = false;
    }
    VLOG(3) << "RunNvshmemCollective counter " << *counter << " " << should_run;
    ++(*counter);
  }

  if (!should_run) {
    VLOG(3) << "Skipping Recv operation";
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(auto* collectives, GetNvshmemCollectivesFromRegistry());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Communicator> nvshmem_comm,
                      collectives->CreateCommunicator());
  VLOG(1) << "Running Recv operation"
          << " element_type=" << buffer.element_type
          << " destination_buffer=" << buffer.destination_buffer.opaque()
          << " source_buffer=" << buffer.source_buffer.opaque()
          << " element_count=" << buffer.element_count
          << " source_id=" << *source_id;
  auto recv_event = nvshmem_comm->Recv(
      buffer.destination_buffer, buffer.source_buffer, buffer.element_type,
      buffer.element_count, RankId(*source_id), GpuCollectives::On(stream));
  tsl::BlockUntilReady(recv_event);
  if (recv_event.IsError()) {
    return recv_event.GetError();
  }
  TF_RETURN_IF_ERROR(nvshmem_comm->Quiet(GpuCollectives::On(stream)));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

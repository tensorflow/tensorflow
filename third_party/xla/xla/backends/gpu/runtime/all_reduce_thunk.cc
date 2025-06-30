/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/all_reduce_thunk.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace gpu {
namespace {

absl::Status CheckImplementableInst(const HloInstruction* inst,
                                    Thunk::Kind reduction_op) {
  for (HloInstruction* operand : inst->operands()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand->shape(), reduction_op));
  }

  if (!MatchReductionComputation(inst->called_computations().front())
           .has_value()) {
    return absl::UnimplementedError("Unrecognized reduction computation");
  }

  return absl::OkStatus();
}

template <typename HloInstType>
AllReduceConfig GetAllReduceConfigInst(HloInstType* inst) {
  std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(inst->called_computations().front());
  CHECK(reduction_kind.has_value());

  AllReduceConfig config;
  config.config = GetCollectiveConfig(inst, inst->use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename HloInstType>
CollectiveOpGroupMode GetGroupModeInst(HloInstType* inst) {
  return GetAllReduceConfigInst(inst).config.group_mode;
}

}  // namespace

absl::Status RunAllReduce(ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;
  TF_RETURN_IF_ERROR(MaybeRegisterBuffers(stream.parent(), buffers, comm));

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(comm);
  tsl::AsyncValueRef<Communicator::Event> event =
      gpu_comm->GroupExecute([reduction_kind, &buffers,
                              &stream](GpuCommunicator* comm) -> absl::Status {
        for (DeviceBufferPair& buffer : buffers) {
          TF_RETURN_IF_ERROR(comm->LaunchAllReduce(
              buffer.source_buffer, buffer.destination_buffer,
              buffer.element_type, buffer.element_count, reduction_kind,
              GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });
  tsl::BlockUntilReady(event);
  VLOG(3) << "Done performing all-reduce for ordinal: " << device_ordinal;
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

AllReduceReduceScatterThunkBase::AllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, AllReduceConfig config,
    std::vector<Buffer> buffers, bool is_sync)
    : CollectiveThunk(kind, thunk_info, is_sync, AsyncStreamKind::kCollective),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

AllReduceStartThunk::AllReduceStartThunk(ThunkInfo thunk_info,
                                         const HloAllReduceInstruction* inst,
                                         std::vector<Buffer> buffers,
                                         bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(
          Thunk::kAllReduceStart, thunk_info, GetAllReduceConfigInst(inst),
          std::move(buffers), IsGPUSyncCollective(*inst)),
      collective_kernel_thunk_{
          thunk_info,
          config_.config,
          config_.reduction_kind,
          IsAsync(),
          buffers_,
          /*is_collective_kernel_enabled=*/
          inst->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_unsupported_use_all_reduce_one_shot_kernel(),
      } {}

absl::Status AllReduceStartThunk::CheckImplementable(
    const HloAllReduceInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<AllReduceStartThunk>(
      CheckImplementableInst(inst, Thunk::kAllReduceStart), inst, replica_count,
      partition_count);
}

CollectiveOpGroupMode AllReduceStartThunk::GetGroupMode(
    const HloAllReduceInstruction* inst) {
  return GetGroupModeInst(inst);
}

absl::Status AllReduceStartThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Prepare(params, resource_requests));
  return collective_kernel_thunk_.Prepare(params, resource_requests);
}

absl::Status AllReduceStartThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, config()));
  TF_ASSIGN_OR_RETURN(bool use_collective_kernel,
                      collective_kernel_thunk_.IsSupported(
                          clique_key, params.collective_cliques));
  if (use_collective_kernel) {
    TF_RETURN_IF_ERROR(collective_kernel_thunk_.Initialize(params));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> AllReduceStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));

  TF_ASSIGN_OR_RETURN(bool use_collective_kernel,
                      collective_kernel_thunk_.IsSupported(
                          comm_handle.clique_key, params.collective_cliques));

  if (use_collective_kernel) {
    TF_RETURN_IF_ERROR(collective_kernel_thunk_.ExecuteOnStream(params));
    return false;  // No need for "first" invocation to rendezvous when not
                   // using nccl.
  }

  TF_RETURN_IF_ERROR(RunAllReduce(config_.reduction_kind, device_buffers,
                                  stream, comm_handle.comm));
  return true;
}

ReduceScatterStartThunk::ReduceScatterStartThunk(
    ThunkInfo thunk_info, const HloReduceScatterInstruction* inst,
    std::vector<Buffer> buffers, bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(
          Thunk::kReduceScatterStart, thunk_info, GetAllReduceConfigInst(inst),
          std::move(buffers), IsGPUSyncCollective(*inst)) {}

/*static*/ absl::Status ReduceScatterStartThunk::CheckImplementable(
    const HloReduceScatterInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<ReduceScatterStartThunk>(
      CheckImplementableInst(inst, Thunk::kReduceScatterStart), inst,
      replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode ReduceScatterStartThunk::GetGroupMode(
    const HloReduceScatterInstruction* inst) {
  return GetGroupModeInst(inst);
}

absl::StatusOr<bool> ReduceScatterStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  TF_RETURN_IF_ERROR(RunReduceScatter(config_.reduction_kind, device_buffers,
                                      stream, comm_handle.comm));
  return true;
}

absl::Status RunReduceScatter(ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, Communicator* comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;
  TF_RETURN_IF_ERROR(MaybeRegisterBuffers(stream.parent(), buffers, comm));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(comm);
  tsl::AsyncValueRef<Communicator::Event> event =
      gpu_comm->GroupExecute([num_ranks, reduction_kind, &buffers,
                              &stream](GpuCommunicator* comm) -> absl::Status {
        for (DeviceBufferPair& buffer : buffers) {
          // buffer.element_count is the source buffers element count. For
          // ncclReduceScatter, we need the destination buffers element count.
          TF_RET_CHECK(buffer.element_count % num_ranks == 0)
              << "Source buffer was not an exact multiple of the number of "
                 "participants.";

          TF_RETURN_IF_ERROR(comm->LaunchReduceScatter(
              buffer.source_buffer, buffer.destination_buffer,
              buffer.element_type, buffer.element_count / num_ranks,
              reduction_kind, GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });
  tsl::BlockUntilReady(event);
  VLOG(3) << "Done performing reduce-scatter for ordinal: " << device_ordinal;
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

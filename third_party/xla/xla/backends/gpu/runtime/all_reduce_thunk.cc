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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "xla/tsl/platform/status_macros.h"

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
CollectiveOpGroupMode GetGroupModeInst(const HloInstType* inst) {
  return GetAllReduceConfigInst(inst).config.group_mode;
}

}  // namespace

AllReduceConfig GetAllReduceConfigInst(
    const HloAllReduceInstructionBase* inst) {
  std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(inst->called_computations().front());
  CHECK(reduction_kind.has_value());

  AllReduceConfig config;
  config.config = GetCollectiveConfig(inst, inst->use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

absl::Status RunAllReduce(ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator& comm,
                          bool use_symmetric_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal) << "Performing all-reduce";
  TF_RETURN_IF_ERROR(MaybeRegisterBuffers(stream.parent(), buffers, &comm,
                                          use_symmetric_buffer));

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
  Future<> future =
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
  TF_RETURN_IF_ERROR(future.Await());
  XLA_VLOG_DEVICE(3, device_ordinal) << "Done performing all-reduce";
  return absl::OkStatus();
}

AllReduceReduceScatterThunkBase::AllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, AllReduceConfig config,
    std::vector<Buffer> buffers)
    : CollectiveThunk(kind, thunk_info, false),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_element_type.size(), buffers_.size());
}

AllReduceThunk::AllReduceThunk(
    ThunkInfo thunk_info, const HloAllReduceInstruction* inst,
    std::vector<Buffer> buffers,
    std::unique_ptr<CollectiveKernelThunk> collective_kernel_thunk,
    bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(Thunk::kAllReduce, thunk_info,
                                      GetAllReduceConfigInst(inst),
                                      std::move(buffers)),
      collective_kernel_thunk_(std::move(collective_kernel_thunk)) {}

AllReduceThunk::AllReduceThunk(ThunkInfo thunk_info, AllReduceConfig config,
                               std::vector<Buffer> buffers)
    : AllReduceReduceScatterThunkBase(Thunk::kAllReduce, thunk_info,
                                      std::move(config), std::move(buffers)),
      collective_kernel_thunk_(nullptr) {}

AllReduceThunk::AllReduceThunk(
    ThunkInfo thunk_info, AllReduceConfig config, std::vector<Buffer> buffers,
    std::unique_ptr<CollectiveKernelThunk> collective_kernel_thunk)
    : AllReduceReduceScatterThunkBase(Thunk::kAllReduce, thunk_info,
                                      std::move(config), std::move(buffers)),
      collective_kernel_thunk_(std::move(collective_kernel_thunk)) {}

absl::Status AllReduceThunk::CheckImplementable(
    const HloAllReduceInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<AllReduceThunk>(
      CheckImplementableInst(inst, Thunk::kAllReduce), inst, replica_count,
      partition_count);
}

CollectiveOpGroupMode AllReduceThunk::GetGroupMode(
    const HloAllReduceInstruction* inst) {
  return GetGroupModeInst(inst);
}

absl::Status AllReduceThunk::Prepare(const PrepareParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Prepare(params));
  return collective_kernel_thunk_->Prepare(params);
}

absl::Status AllReduceThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetCollectiveGpuCliqueKey(*params.collective_params,
                                                config(), /*is_p2p=*/false));
  TF_ASSIGN_OR_RETURN(
      bool use_collective_kernel,
      collective_kernel_thunk_->IsSupported(clique_key, *params.executor,
                                            *params.collective_params));
  if (use_collective_kernel) {
    TF_RETURN_IF_ERROR(collective_kernel_thunk_->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status AllReduceThunk::RunCollective(const ExecuteParams& params,
                                           const GpuCliqueKey& clique_key,
                                           se::Stream& stream,
                                           Communicator& comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                             config_.config.operand_element_type));

  TF_ASSIGN_OR_RETURN(
      bool use_collective_kernel,
      collective_kernel_thunk_->IsSupported(
          clique_key, *params.stream->parent(), *params.collective_params));

  if (use_collective_kernel) {
    return collective_kernel_thunk_->ExecuteOnStream(params);
  }

  return RunAllReduce(config_.reduction_kind, device_buffers, stream, comm,
                      config_.config.use_symmetric_buffer);
}

absl::StatusOr<std::unique_ptr<AllReduceThunk>> AllReduceThunk::FromProto(
    ThunkInfo thunk_info, const AllReduceStartThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  CollectiveConfig config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  ASSIGN_OR_RETURN(ReductionKind reduction_kind,
                   FromReductionKindProto(thunk_proto.reduction_kind()));

  std::optional<LaunchDimensions> launch_dimensions = std::nullopt;
  if (thunk_proto.has_launch_dimensions()) {
    TF_ASSIGN_OR_RETURN(
        launch_dimensions,
        LaunchDimensions::FromProto(thunk_proto.launch_dimensions()));
  }
  auto kernel_thunk = std::make_unique<CollectiveKernelThunk>(
      thunk_info, config, reduction_kind, thunk_proto.is_async(), buffers,
      thunk_proto.collective_kernel_enabled(), thunk_proto.kernel_name(),
      launch_dimensions, thunk_proto.shmem_bytes(),
      thunk_proto.is_multimem_enabled());

  return std::make_unique<AllReduceThunk>(
      std::move(thunk_info), AllReduceConfig{config, reduction_kind},
      std::move(buffers), std::move(kernel_thunk));
}

absl::StatusOr<ThunkProto> AllReduceThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  AllReduceStartThunkProto* thunk_proto =
      proto.mutable_all_reduce_start_thunk();

  for (const Buffer& buffer : buffers_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  thunk_proto->set_reduction_kind(ToReductionKindProto(config_.reduction_kind));

  thunk_proto->set_is_multimem_enabled(
      collective_kernel_thunk_->is_multimem_enabled());
  thunk_proto->set_shmem_bytes(collective_kernel_thunk_->shmem_bytes());
  thunk_proto->set_kernel_name(collective_kernel_thunk_->kernel_name());
  thunk_proto->set_collective_kernel_enabled(
      collective_kernel_thunk_->collective_kernel_enabled());
  thunk_proto->set_is_async(collective_kernel_thunk_->is_async());
  if (auto launch_dimensions = collective_kernel_thunk_->launch_dimensions();
      launch_dimensions.has_value()) {
    *thunk_proto->mutable_launch_dimensions() = launch_dimensions->ToProto();
  }

  return proto;
}

ReduceScatterThunk::ReduceScatterThunk(ThunkInfo thunk_info,
                                       const HloReduceScatterInstruction* inst,
                                       std::vector<Buffer> buffers,
                                       bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(Thunk::kReduceScatter, thunk_info,
                                      GetAllReduceConfigInst(inst),
                                      std::move(buffers)) {}

/*static*/ absl::Status ReduceScatterThunk::CheckImplementable(
    const HloReduceScatterInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<ReduceScatterThunk>(
      CheckImplementableInst(inst, Thunk::kReduceScatter), inst, replica_count,
      partition_count);
}

/*static*/ CollectiveOpGroupMode ReduceScatterThunk::GetGroupMode(
    const HloReduceScatterInstruction* inst) {
  return GetGroupModeInst(inst);
}

ReduceScatterThunk::ReduceScatterThunk(ThunkInfo thunk_info,
                                       AllReduceConfig config,
                                       std::vector<Buffer> buffers)
    : AllReduceReduceScatterThunkBase(Thunk::kReduceScatter, thunk_info,
                                      std::move(config), std::move(buffers)) {}

absl::StatusOr<std::unique_ptr<ReduceScatterThunk>>
ReduceScatterThunk::FromProto(
    ThunkInfo thunk_info, const ReduceScatterThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  CollectiveConfig config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  ASSIGN_OR_RETURN(ReductionKind reduction_kind,
                   FromReductionKindProto(thunk_proto.reduction_kind()));

  return std::make_unique<ReduceScatterThunk>(
      std::move(thunk_info), AllReduceConfig{config, reduction_kind},
      std::move(buffers));
}

absl::StatusOr<ThunkProto> ReduceScatterThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  ReduceScatterThunkProto* thunk_proto = proto.mutable_reduce_scatter_thunk();

  for (const Buffer& buffer : buffers_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  thunk_proto->set_reduction_kind(ToReductionKindProto(config_.reduction_kind));

  return proto;
}

absl::Status ReduceScatterThunk::RunCollective(const ExecuteParams& params,
                                               const GpuCliqueKey& clique_key,
                                               se::Stream& stream,
                                               Communicator& comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                             config_.config.operand_element_type));
  return RunReduceScatter(config_.reduction_kind, device_buffers, stream, comm,
                          config_.config.use_symmetric_buffer);
}

absl::Status RunReduceScatter(ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, Communicator& comm,
                              bool use_symmetric_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal) << "Performing reduce-scatter";
  TF_RETURN_IF_ERROR(MaybeRegisterBuffers(stream.parent(), buffers, &comm,
                                          use_symmetric_buffer));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm.NumRanks());

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
  Future<> future =
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
  TF_RETURN_IF_ERROR(future.Await());
  XLA_VLOG_DEVICE(3, device_ordinal) << "Done performing reduce-scatter";
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

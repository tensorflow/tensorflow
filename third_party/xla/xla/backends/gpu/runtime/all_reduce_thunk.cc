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

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

absl::Status CheckImplementableInst(const HloInstruction* inst,
                                    Thunk::Kind reduction_op) {
  for (HloInstruction* operand : inst->operands()) {
    RETURN_IF_ERROR(IsValidOperand(operand->shape(), reduction_op));
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
  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(&comm);
  Future<> future = gpu_comm->GroupExecute([&]() -> absl::Status {
    for (DeviceBufferPair& buffer : buffers) {
      RETURN_IF_ERROR(gpu_comm->LaunchAllReduce(
          buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
          buffer.element_count, reduction_kind, GpuCollectives::On(stream)));
    }
    return absl::OkStatus();
  });
  RETURN_IF_ERROR(future.Await());
  XLA_VLOG_DEVICE(3, device_ordinal) << "Done performing all-reduce";
  return absl::OkStatus();
}

AllReduceReduceScatterThunkBase::AllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, AllReduceConfig config,
    std::vector<Buffer> buffers)
    : CollectiveThunk(kind, thunk_info, std::move(buffers)),
      config_(std::move(config)) {
  CHECK_EQ(config_.config.operand_element_type.size(), this->buffers().size());
}

AllReduceThunk::AllReduceThunk(ThunkInfo thunk_info, AllReduceConfig config,
                               std::vector<Buffer> buffers)
    : AllReduceReduceScatterThunkBase(Thunk::kAllReduce, thunk_info,
                                      std::move(config), std::move(buffers)) {}

AllReduceThunk::AllReduceThunk(ThunkInfo thunk_info,
                               const HloAllReduceInstruction* inst,
                               std::vector<Buffer> buffers,
                               bool p2p_memcpy_enabled)
    : AllReduceReduceScatterThunkBase(Thunk::kAllReduce, thunk_info,
                                      GetAllReduceConfigInst(inst),
                                      std::move(buffers)) {}
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

absl::Status AllReduceThunk::RunCollective(const ExecuteParams& params,
                                           const GpuCliqueKey& clique_key,
                                           se::Stream& stream,
                                           Communicator& comm) {
  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
                                          config_.config.operand_element_type));

  return RunAllReduce(config_.reduction_kind, device_buffers, stream, comm,
                      config_.config.use_symmetric_buffer);
}

absl::StatusOr<std::unique_ptr<AllReduceThunk>> AllReduceThunk::FromProto(
    ThunkInfo thunk_info, const AllReduceThunkProto& thunk_proto,
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

  return std::make_unique<AllReduceThunk>(
      std::move(thunk_info), AllReduceConfig{config, reduction_kind},
      std::move(buffers));
}

absl::StatusOr<ThunkProto> AllReduceThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  AllReduceThunkProto* thunk_proto = proto.mutable_all_reduce_thunk();

  for (const Buffer& buffer : buffers()) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  thunk_proto->set_reduction_kind(ToReductionKindProto(config_.reduction_kind));

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

  for (const Buffer& buffer : buffers()) {
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
  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
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

  ASSIGN_OR_RETURN(int32_t num_ranks, comm.NumRanks());

  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(&comm);
  Future<> future = gpu_comm->GroupExecute([&]() -> absl::Status {
    for (DeviceBufferPair& buffer : buffers) {
      // buffer.element_count is the source buffers element count. For
      // ncclReduceScatter, we need the destination buffers element count.
      TF_RET_CHECK(buffer.element_count % num_ranks == 0)
          << "Source buffer was not an exact multiple of the number of "
             "participants.";

      RETURN_IF_ERROR(gpu_comm->LaunchReduceScatter(
          buffer.source_buffer, buffer.destination_buffer, buffer.element_type,
          buffer.element_count / num_ranks, reduction_kind,
          GpuCollectives::On(stream)));
    }
    return absl::OkStatus();
  });
  RETURN_IF_ERROR(future.Await());
  XLA_VLOG_DEVICE(3, device_ordinal) << "Done performing reduce-scatter";
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

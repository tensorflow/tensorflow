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

#include "xla/backends/gpu/runtime/all_gather_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

namespace {
AllGatherConfig GetAllGatherConfig(const HloAllGatherInstruction* inst) {
  AllGatherConfig config;
  config.config = GetCollectiveConfig(inst, inst->use_global_device_ids());
  return config;
}

absl::Status CheckImplementableInst(const HloAllGatherInstruction* inst) {
  for (HloInstruction* operand : inst->operands()) {
    const Shape& shape = operand->shape();

    RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kAllGather));

    if (!ShapeUtil::IsEffectivelyMostMajorDimension(
            shape, inst->all_gather_dimension())) {
      return absl::AbortedError(absl::StrFormat(
          "all-gather dim %u is not the most major in input shape %s",
          inst->all_gather_dimension(), shape.ToString(/*print_layout=*/true)));
    }
  }

  return absl::OkStatus();
}
}  // namespace

AllGatherStartThunk::AllGatherStartThunk(
    ThunkInfo thunk_info,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events,
    CollectiveConfig config, std::vector<Buffer> buffers)
    : CollectiveThunk(Thunk::kAllGatherStart, thunk_info, async_events, false),
      config_(AllGatherConfig{config}),
      buffers_(std::move(buffers)) {}

AllGatherStartThunk::AllGatherStartThunk(ThunkInfo thunk_info,
                                         const HloAllGatherInstruction* inst,
                                         std::vector<Buffer> buffers,
                                         bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kAllGatherStart, thunk_info,
                      IsGPUSyncCollective(*inst), false),
      config_(GetAllGatherConfig(inst)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_element_type.size(), buffers_.size());
}

/*static*/ absl::Status AllGatherStartThunk::CheckImplementable(
    const HloAllGatherInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<AllGatherStartThunk>(
      CheckImplementableInst(inst), inst, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode AllGatherStartThunk::GetGroupMode(
    const HloAllGatherInstruction* inst) {
  return GetAllGatherConfig(inst).config.group_mode;
}

absl::StatusOr<std::unique_ptr<AllGatherStartThunk>>
AllGatherStartThunk::FromProto(
    ThunkInfo thunk_info, const AllGatherStartThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    CollectiveThunk::AsyncEventsMap& async_events_map) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events;
  if (thunk_proto.has_async_events_unique_id()) {
    std::shared_ptr<CollectiveThunk::AsyncEvents>& events =
        async_events_map[AsyncEventsUniqueId{
            thunk_proto.async_events_unique_id()}];
    if (!events) {
      events = std::make_shared<CollectiveThunk::AsyncEvents>();
    }
    async_events = events;
  }

  return std::make_unique<AllGatherStartThunk>(
      std::move(thunk_info), async_events,
      CollectiveConfig::FromProto(thunk_proto.collective_config()),
      std::move(buffers));
}

absl::StatusOr<ThunkProto> AllGatherStartThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  AllGatherStartThunkProto* thunk_proto =
      proto.mutable_all_gather_start_thunk();

  std::optional<AsyncEventsUniqueId> async_events_id = GetAsyncEventsUniqueId();
  if (async_events_id.has_value()) {
    thunk_proto->set_async_events_unique_id(async_events_id->value());
  }

  for (const Buffer& buffer : buffers_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }
  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  return proto;
}

absl::StatusOr<bool> AllGatherStartThunk::RunCollective(
    const ExecuteParams& params, const GpuCliqueKey& clique_key,
    se::Stream& stream, Communicator& comm) {
  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params, buffers_,
                                          config_.config.operand_element_type));
  RETURN_IF_ERROR(xla::gpu::RunAllGather(device_buffers, stream, comm,
                                         config_.config.use_symmetric_buffer));
  return true;
}

absl::Status RunAllGather(std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator& comm,
                          bool use_symmetric_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal) << "Performing all-gather";
  RETURN_IF_ERROR(MaybeRegisterBuffers(stream.parent(), buffers, &comm,
                                       use_symmetric_buffer));
  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(&comm);
  Future<> future = gpu_comm->GroupExecute(
      [&buffers, &stream](GpuCommunicator* comm) -> absl::Status {
        for (DeviceBufferPair& buffer : buffers) {
          RETURN_IF_ERROR(comm->LaunchAllGather(
              buffer.source_buffer, buffer.destination_buffer,
              buffer.element_type, buffer.element_count,
              GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });

  RETURN_IF_ERROR(future.Await());
  XLA_VLOG_DEVICE(3, device_ordinal) << "Done performing all-gather";
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

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
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/communicator.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/logging.h"
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

AllGatherThunk::AllGatherThunk(ThunkInfo thunk_info,
                               const HloAllGatherInstruction* inst,
                               std::vector<Buffer> buffers,
                               bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kAllGather, thunk_info),
      config_(GetAllGatherConfig(inst)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_element_type.size(), buffers_.size());
}

AllGatherThunk::AllGatherThunk(ThunkInfo thunk_info, CollectiveConfig config,
                               std::vector<Buffer> buffers)
    : CollectiveThunk(Thunk::kAllGather, thunk_info),
      config_(AllGatherConfig{std::move(config)}),
      buffers_(std::move(buffers)) {}

/*static*/ absl::Status AllGatherThunk::CheckImplementable(
    const HloAllGatherInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<AllGatherThunk>(CheckImplementableInst(inst), inst,
                                          replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode AllGatherThunk::GetGroupMode(
    const HloAllGatherInstruction* inst) {
  return GetAllGatherConfig(inst).config.group_mode;
}

absl::StatusOr<std::unique_ptr<AllGatherThunk>> AllGatherThunk::FromProto(
    ThunkInfo thunk_info, const AllGatherStartThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  return std::make_unique<AllGatherThunk>(
      std::move(thunk_info),
      CollectiveConfig::FromProto(thunk_proto.collective_config()),
      std::move(buffers));
}

absl::StatusOr<ThunkProto> AllGatherThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  AllGatherStartThunkProto* thunk_proto =
      proto.mutable_all_gather_start_thunk();

  for (const Buffer& buffer : buffers_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }
  *thunk_proto->mutable_collective_config() = config_.config.ToProto();
  return proto;
}

absl::Status AllGatherThunk::RunCollective(const ExecuteParams& params,
                                           const GpuCliqueKey& clique_key,
                                           se::Stream& stream,
                                           Communicator& comm) {
  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, buffers_,
                                          config_.config.operand_element_type));
  return xla::gpu::RunAllGather(device_buffers, stream, comm,
                                config_.config.use_symmetric_buffer);
}

absl::Status RunAllGather(std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator& comm,
                          bool use_symmetric_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal) << "Performing all-gather";
  RETURN_IF_ERROR(MaybeRegisterBuffers(stream.parent(), buffers, &comm,
                                       use_symmetric_buffer));
  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
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

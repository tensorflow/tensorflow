/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_group_thunk.h"

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/util.h"

namespace xla::gpu {

CollectiveGroupThunk::CollectiveGroupThunk(ThunkInfo thunk_info,
                                           Thunk::Kind kind,
                                           ThunkSequence thunks)
    : TracedCommand(kind, std::move(thunk_info)),
      executor_(std::move(thunks)) {}

absl::Status CollectiveGroupThunk::Prepare(const PrepareParams& params) {
  return executor_.Prepare(params);
}

absl::Status CollectiveGroupThunk::Initialize(const InitializeParams& params) {
  return executor_.Initialize(params);
}

std::string CollectiveGroupThunk::ToString(int indent) const {
  return absl::StrCat("\n", executor_.thunks().ToString(indent + 1));
}

Thunk::BufferUses CollectiveGroupThunk::buffer_uses() const {
  BufferUses uses;
  uses.reserve(thunks().size() * 2);
  for (const std::unique_ptr<Thunk>& thunk : executor_.thunks()) {
    BufferUses sub_uses = thunk->buffer_uses();
    uses.insert(uses.end(), std::make_move_iterator(sub_uses.begin()),
                std::make_move_iterator(sub_uses.end()));
  }
  return uses;
}

absl::Status CollectiveGroupThunk::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  GlobalDeviceId global_device_id = params.collective_params->global_device_id;

  // Collect all communicators used by non-degenerate collective thunks.
  // Degenerate collectives are emitted as device-to-device copies.
  std::vector<GpuCommunicator*> comms;
  for (const std::unique_ptr<Thunk>& thunk : executor_.thunks()) {
    auto* collective_thunk = dynamic_cast<CollectiveThunk*>(thunk.get());

    if (collective_thunk == nullptr) {
      if (dynamic_cast<DeviceToDeviceCopyThunk*>(thunk.get()) != nullptr) {
        continue;
      }

      return InvalidArgument(
          "Unexpected thunk in collective group; expected a collective or "
          "device-to-device copy thunk, got %v",
          thunk->kind());
    }

    ABSL_ASSIGN_OR_RETURN(auto clique_key, collective_thunk->GetCliqueKey(params));
    ABSL_ASSIGN_OR_RETURN(GpuCommunicator * comm, params.collective_cliques->GetComm(
                                                 clique_key, global_device_id));
    if (!absl::c_contains(comms, comm)) {
      comms.push_back(comm);
    }
  }

  // No communicator means every nested thunk is a plain device-to-device copy.
  if (comms.empty()) {
    return executor_.ExecuteOnStream(params);
  }

  // If nested thunks use a single comm, use it directly to execute the group.
  if (comms.size() == 1) {
    Future<> executed = comms.front()->GroupExecute(
        [&] { return executor_.ExecuteOnStream(params); });
    return executed.Await();
  }

  // Otherwise use a multi-comm group launch.
  return params.collective_params->collectives->GroupLaunch(
      comms, [&] { return executor_.ExecuteOnStream(params); });
}

absl::StatusOr<const se::CommandBuffer::Command*> CollectiveGroupThunk::Record(
    const ExecuteParams& execute_params, const RecordParams& record_params,
    RecordAction record_action, se::CommandBuffer* command_buffer) {
  // Like CollectiveThunk::Record, trace directly via TraceCommandBufferFactory
  // rather than TracedCommand::RecordTracedCommand (which uses a per-rank
  // TracedCommandBuffer LRU cache). With NCCL collectives, all participating
  // ranks must enter stream capture together whenever Record is invoked; if one
  // rank hits its local TracedCommandBuffer cache and skips tracing while
  // another rank misses the cache and traces, NCCL will deadlock.
  std::unique_ptr<se::CommandBuffer> nested_cmd;
  ABSL_ASSIGN_OR_RETURN(
      nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, [&](se::Stream* stream) {
            return ExecuteOnStream(execute_params.WithComputeStream(stream));
          }));

  ABSL_RETURN_IF_ERROR(nested_cmd->SetPriority(se::StreamPriority::Highest));

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateChildCommand(*nested_cmd,
                                              create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    ABSL_RETURN_IF_ERROR(
        command_buffer->UpdateChildCommand(update->command, *nested_cmd));
    return update->command;
  }
  return Internal("Invalid record action");
}

absl::Status CollectiveGroupThunk::WalkNested(Walker pre_order,
                                              Walker post_order) {
  return executor_.thunks().WalkNested(pre_order, post_order);
}

absl::Status CollectiveGroupThunk::TransformNested(Transformer callback) {
  return executor_.thunks().TransformNested(callback);
}

absl::StatusOr<std::unique_ptr<CollectiveGroupThunk>>
CollectiveGroupThunk::FromProto(
    ThunkInfo thunk_info, const CollectiveGroupThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const Deserializer& deserializer) {
  ThunkSequence thunk_sequence;
  for (const auto& sub_thunk_proto : thunk_proto.thunks()) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> sub_thunk,
                     deserializer(sub_thunk_proto));
    thunk_sequence.push_back(std::move(sub_thunk));
  }

  ABSL_ASSIGN_OR_RETURN(Thunk::Kind kind,
                   Thunk::KindFromProto(thunk_proto.thunk_kind()));

  return std::make_unique<CollectiveGroupThunk>(std::move(thunk_info), kind,
                                                std::move(thunk_sequence));
}

absl::StatusOr<ThunkProto> CollectiveGroupThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CollectiveGroupThunkProto* thunk_proto =
      proto.mutable_collective_group_thunk();

  thunk_proto->set_thunk_kind(Thunk::KindToProto(kind()));

  for (const auto& thunk : executor_.thunks()) {
    ABSL_ASSIGN_OR_RETURN(*thunk_proto->add_thunks(), thunk->ToProto());
  }

  return proto;
}

}  // namespace xla::gpu

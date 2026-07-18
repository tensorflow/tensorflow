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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/util.h"

namespace xla::gpu {

CollectiveGroupThunk::CollectiveGroupThunk(ThunkInfo thunk_info,
                                           Thunk::Kind kind,
                                           ThunkSequence thunks)
    : Thunk(kind, std::move(thunk_info)), executor_(std::move(thunks)) {}

absl::Status CollectiveGroupThunk::Prepare(const PrepareParams& params) {
  return executor_.Prepare(params);
}

absl::Status CollectiveGroupThunk::Initialize(const InitializeParams& params) {
  return executor_.Initialize(params);
}

std::string CollectiveGroupThunk::ToString(int indent) const {
  return absl::StrCat("\n", executor_.thunks().ToString(indent + 1));
}

absl::Status CollectiveGroupThunk::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  GlobalDeviceId global_device_id = params.collective_params->global_device_id;

  // Collect all communicators used by nested thunks.
  std::vector<GpuCommunicator*> comms;
  for (const std::unique_ptr<Thunk>& thunk : executor_.thunks()) {
    auto* collective_thunk = absl::down_cast<CollectiveThunk*>(thunk.get());
    ASSIGN_OR_RETURN(auto clique_key, collective_thunk->GetCliqueKey(params));
    ASSIGN_OR_RETURN(GpuCommunicator * comm, params.collective_cliques->GetComm(
                                                 clique_key, global_device_id));
    if (!absl::c_contains(comms, comm)) {
      comms.push_back(comm);
    }
  }

  // It is a bug if collective group was formed with no collective ops.
  if (comms.empty()) {
    return InvalidArgument(
        "Collective group must have at least one nested collective thunk");
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

absl::Status CollectiveGroupThunk::WalkNested(Walker callback) {
  return executor_.thunks().WalkNested(callback);
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
    ASSIGN_OR_RETURN(std::unique_ptr<Thunk> sub_thunk,
                     deserializer(sub_thunk_proto));
    thunk_sequence.push_back(std::move(sub_thunk));
  }

  ASSIGN_OR_RETURN(Thunk::Kind kind,
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
    ASSIGN_OR_RETURN(*thunk_proto->add_thunks(), thunk->ToProto());
  }

  return proto;
}

}  // namespace xla::gpu

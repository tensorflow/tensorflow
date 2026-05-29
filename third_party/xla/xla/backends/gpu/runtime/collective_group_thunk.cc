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

#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/core/collectives/communicator.h"
#include "xla/future.h"
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

absl::StatusOr<GpuCliqueKey> CollectiveGroupThunk::GetCliqueKey(
    const Thunk& thunk, const Thunk::ExecuteParams& params) {
  // Gather the set of all cliques. There should be only one.
  absl::flat_hash_set<GpuCliqueKey> clique_set;
  RETURN_IF_ERROR(thunk.Walk([&](const Thunk* nested) -> absl::Status {
    if (auto* collective = dynamic_cast<const CollectiveThunk*>(nested)) {
      ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                       collective->GetCliqueKey(params));
      clique_set.insert(std::move(clique_key));
    }
    return absl::OkStatus();
  }));

  if (clique_set.empty()) {
    return InvalidArgument("No clique in NCCL group");
  }
  if (clique_set.size() > 1) {
    return InvalidArgument("More than one clique in NCCL group");
  }

  return *clique_set.begin();
}

absl::Status CollectiveGroupThunk::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  ASSIGN_OR_RETURN(GpuCliqueKey clique_key, GetCliqueKey(*this, params));
  ASSIGN_OR_RETURN(Communicator * comm,
                   params.collective_cliques->GetComm(
                       clique_key, params.collective_params->global_device_id));
  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(comm);
  Future<> group_future = gpu_comm->GroupExecute(
      [this, &params](GpuCommunicator* comm) -> absl::Status {
        return executor_.ExecuteOnStream(params);
      });
  return group_future.Await();
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

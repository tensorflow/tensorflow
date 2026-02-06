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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/future.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/casts.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

CollectiveGroupThunk::CollectiveGroupThunk(
    ThunkInfo thunk_info, Thunk::Kind kind,
    std::vector<std::unique_ptr<Thunk>> thunks,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : Thunk(kind, std::move(thunk_info)), async_events_(async_events) {
  for (auto& thunk : thunks) {
    thunks_.emplace_back(std::move(thunk));
  }
}
absl::Status CollectiveGroupThunk::Prepare(const PrepareParams& params) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    RETURN_IF_ERROR(thunk->Prepare(params));
  }
  return absl::OkStatus();
}
absl::Status CollectiveGroupThunk::Initialize(const InitializeParams& params) {
  if (async_events_) {
    RETURN_IF_ERROR(async_events_->Initialize(params.executor));
  }
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status CollectiveGroupThunk::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  int64_t async_stream_idx = Thunk::execution_stream_id().value();
  // Async streams are already assigned in gpu_executable.cc::ExecuteThunks.
  // async_streams is therefore guaranteed to be non-null and to have enough
  // elements to index by the AsyncStreamKind enum.
  se::Stream* async_stream =
      params.collective_params->async_streams.at(async_stream_idx);
  RETURN_IF_ERROR(async_stream->WaitFor(params.stream));

  // Gather the set of all communicators. There should be only one.
  absl::flat_hash_set<Communicator*> communicator_set;
  absl::Status s;
  ForAllThunks([&params, &s, &communicator_set](const Thunk* thunk) {
    absl::StatusOr<std::vector<Communicator*>> communicators =
        thunk->GetCommunicators(params);
    if (!communicators.ok()) {
      s = communicators.status();
      return;
    }
    for (Communicator* comm : *communicators) {
      communicator_set.insert(comm);
    }
  });
  if (communicator_set.empty()) {
    return absl::InvalidArgumentError("No communicators in NCCL group");
  }
  if (communicator_set.size() > 1) {
    return absl::InvalidArgumentError(
        "More than one communicator in NCCL group");
  }

  Communicator* comm = *communicator_set.begin();
  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(comm);
  Future<> group_future = gpu_comm->GroupExecute(
      [this, &params](GpuCommunicator* comm) -> absl::Status {
        for (const std::unique_ptr<Thunk>& thunk : thunks_) {
          RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
        }
        return absl::OkStatus();
      });
  RETURN_IF_ERROR(group_future.Await());

  ASSIGN_OR_RETURN(se::Event * event,
                   async_events_->GetEvent(params.stream->parent()));
  RETURN_IF_ERROR(async_stream->RecordEvent(event));

  return absl::OkStatus();
}

void CollectiveGroupThunk::ForAllThunks(
    absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    thunk->ForAllThunks(fn);
  }
}

void CollectiveGroupThunk::ForAllThunksMutable(
    absl::FunctionRef<void(Thunk*)> fn) {
  fn(this);
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    thunk->ForAllThunksMutable(fn);
  }
}

absl::Status CollectiveGroupThunk::TransformAllNestedThunks(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<Thunk>>(std::unique_ptr<Thunk>)>
        fn) {
  for (std::unique_ptr<Thunk>& thunk : thunks_) {
    RETURN_IF_ERROR(thunk->TransformAllNestedThunks(fn));
    ASSIGN_OR_RETURN(thunk, fn(std::move(thunk)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CollectiveGroupThunk>>
CollectiveGroupThunk::FromProto(
    ThunkInfo thunk_info, const CollectiveGroupThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    CollectiveThunk::AsyncEventsMap& async_events_map,
    const Deserializer& deserializer) {
  ThunkSequence thunk_sequence;
  for (const auto& sub_thunk_proto : thunk_proto.thunks()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Thunk> sub_thunk,
                     deserializer(sub_thunk_proto));
    thunk_sequence.push_back(std::move(sub_thunk));
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

  ASSIGN_OR_RETURN(Thunk::Kind kind,
                   Thunk::KindFromProto(thunk_proto.thunk_kind()));

  return std::make_unique<CollectiveGroupThunk>(
      std::move(thunk_info), kind, std::move(thunk_sequence), async_events);
}

absl::StatusOr<ThunkProto> CollectiveGroupThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CollectiveGroupThunkProto* thunk_proto =
      proto.mutable_collective_group_thunk();

  std::optional<AsyncEventsUniqueId> async_events_id = GetAsyncEventsUniqueId();
  if (async_events_id.has_value()) {
    thunk_proto->set_async_events_unique_id(async_events_id->value());
  }
  thunk_proto->set_thunk_kind(Thunk::KindToProto(kind()));

  for (const auto& thunk : thunks_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_thunks(), thunk->ToProto());
  }

  return proto;
}

}  // namespace gpu
}  // namespace xla

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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/nccl_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

CollectiveGroupThunk::CollectiveGroupThunk(
    const HloInstruction* instruction, Thunk::Kind kind,
    std::vector<std::unique_ptr<Thunk>> thunks, AsyncStreamKind stream_kind)
    : Thunk(kind, ThunkInfo::WithProfileAnnotation(instruction)),
      stream_kind_(stream_kind),
      async_events_(new CollectiveThunk::AsyncEvents()) {
  for (auto& thunk : thunks) {
    thunks_.emplace_back(std::move(thunk));
  }
}
absl::Status CollectiveGroupThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}
absl::Status CollectiveGroupThunk::Initialize(const InitializeParams& params) {
  if (async_events_) {
    TF_RETURN_IF_ERROR(async_events_->Initialize(params.executor));
  }
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status CollectiveGroupThunk::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  int64_t async_stream_idx = static_cast<int64_t>(stream_kind_);
  // Async streams are already assigned in gpu_executable.cc::ExecuteThunks.
  // async_streams is therefore guaranteed to be non-null and to have enough
  // elements to index by the AsyncStreamKind enum.
  se::Stream* async_stream =
      params.collective_params->async_streams.at(async_stream_idx);
  TF_RETURN_IF_ERROR(async_stream->WaitFor(params.stream));

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
  TF_RETURN_IF_ERROR(CastCommunicator(comm)->GroupStart());
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
  }
  TF_RETURN_IF_ERROR(CastCommunicator(comm)->GroupEnd());
  TF_ASSIGN_OR_RETURN(se::Event * event,
                      async_events_->GetEvent(params.stream->parent()));
  TF_RETURN_IF_ERROR(async_stream->RecordEvent(event));

  return absl::OkStatus();
}

void CollectiveGroupThunk::ForAllThunks(
    absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    thunk->ForAllThunks(fn);
  }
}

}  // namespace gpu
}  // namespace xla

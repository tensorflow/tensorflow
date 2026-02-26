/* Copyright 2025 The OpenXLA Authors.
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

#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace gpu {

namespace {

bool IsTypeSupportedByNvshmem(PrimitiveType element_type,
                              Thunk::Kind reduction_op) {
  switch (element_type) {
    case S8:
    case PRED:
    case U8:
    case S32:
    case U32:
    case S64:
    case U64:
    case F16:
    case F32:
    case F64:
    case BF16:
      return true;
    case C64:
    case C128:
    case S16:
    case U16:
    case F8E5M2:
    case F8E4M3FN:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
      return !IsReductionCollective(reduction_op);
    default:
      return false;
  }
}

}  // namespace

NvshmemCollectiveThunk::NvshmemCollectiveThunk(Kind kind, ThunkInfo thunk_info,
                                               bool is_sync)
    : Thunk(kind, thunk_info),
      async_events_(is_sync ? nullptr : new CollectiveThunk::AsyncEvents()) {}

absl::StatusOr<xla::gpu::GpuCollectives*> GetNvshmemCollectivesFromRegistry() {
  TF_ASSIGN_OR_RETURN(xla::Collectives * collectives,
                      xla::CollectivesRegistry::Get("gpu", "nvshmem"));
  return tsl::down_cast<xla::gpu::GpuCollectives*>(collectives);
}

absl::Status NvshmemCollectiveThunk::Prepare(const PrepareParams& params) {
  TF_RET_CHECK(params.collective_params &&
               params.collective_params->device_assn);

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, /*is_p2p=*/false));

  TF_ASSIGN_OR_RETURN(std::vector<std::vector<GlobalDeviceId>> device_groups,
                      GetParticipatingDevicesGroups(
                          *params.collective_params->device_assn,
                          config().replica_groups, config().group_mode));

  // Any NVSHMEM collective will need to require a barrier at the end of
  // graph execution to make sure all reads and writes to symmetrics buffers
  // are finished and ready for the next iteration of executable.
  CollectiveCliqueRequests::CliqueRequirements clique_reqs;
  clique_reqs.barrier_reqs = CollectiveCliqueRequests::BarrierRequirements{
      /*module_execution_barrier=*/true};
  return params.collective_clique_requests->RequestClique(
      clique_key, std::move(device_groups), clique_reqs);
}

absl::Status NvshmemCollectiveThunk::Initialize(
    const InitializeParams& params) {
  if (async_events_) {
    TF_RETURN_IF_ERROR(async_events_->Initialize(params.executor));
  }
  return absl::OkStatus();
}

absl::Status NvshmemCollectiveThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  VLOG(1) << absl::StreamFormat("Starting %s %s.", IsAsync() ? "async" : "sync",
                                Thunk::KindToString(kind()));
  AsyncStreamKind stream_kind = GetAsyncStreamKind();
  se::StreamExecutor* executor = params.stream->parent();
  int64_t async_stream_idx = static_cast<int64_t>(stream_kind);

  if (IsAsync()) {
    // Launch collective operation on an async stream.
    se::Stream& async_stream =
        *params.collective_params->async_streams.at(async_stream_idx);

    // Wait for main compute stream to make sure all buffers are ready.
    TF_RETURN_IF_ERROR(async_stream.WaitFor(params.stream));

    TF_RETURN_IF_ERROR(RunNvshmemCollective(params, async_stream));

    // Record collective operation completion.
    TF_ASSIGN_OR_RETURN(se::Event * event, async_events_->GetEvent(executor));
    TF_RETURN_IF_ERROR(async_stream.RecordEvent(event));

  } else {
    // Launch collective operation on a main stream.
    TF_RETURN_IF_ERROR(RunNvshmemCollective(params, *params.stream));
  }
  return absl::OkStatus();
}

NvshmemCollectiveDoneThunk::NvshmemCollectiveDoneThunk(
    Thunk::Kind kind, ThunkInfo thunk_info,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : Thunk(kind, std::move(thunk_info)), async_events_(async_events) {}

absl::Status NvshmemCollectiveDoneThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(se::Event * event, async_events_->GetEvent(executor));
  return params.stream->WaitFor(event);
}

absl::Status IsValidNvshmemOperand(Shape shape, Thunk::Kind reduction_op) {
  if (!shape.IsArray()) {
    return absl::AbortedError(
        absl::StrFormat("input is not a dense array: %s",
                        shape.ToString(/*print_layout=*/true)));
  }
  if (!IsTypeSupportedByNvshmem(shape.element_type(), reduction_op)) {
    return absl::AbortedError(absl::StrFormat(
        "element type %s not suppored by Nvshmem",
        primitive_util::LowercasePrimitiveTypeName(shape.element_type())));
  }
  return absl::OkStatus();
}

absl::StatusOr<void*> NvshmemBufferAddresses::GetNvshmemPtr(
    int device_ordinal) {
  absl::MutexLock lock(mu_);
  auto it = buffer_addrs_.find(device_ordinal);
  if (it != buffer_addrs_.end()) {
    return it->second;
  }
  return absl::NotFoundError("Buffer address not found for device");
}

void NvshmemBufferAddresses::StoreNvshmemPtr(int device_ordinal,
                                             void* buffer_addr) {
  absl::MutexLock lock(mu_);
  buffer_addrs_[device_ordinal] = buffer_addr;
}

std::optional<AsyncEventsUniqueId>
NvshmemCollectiveThunk::GetAsyncEventsUniqueId() const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}

std::optional<AsyncEventsUniqueId>
NvshmemCollectiveDoneThunk::GetAsyncEventsUniqueId() const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}

}  // namespace gpu
}  // namespace xla

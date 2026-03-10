/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/async_execution.h"

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/executable_run_options.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/unique_any.h"
#include "xla/util.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

AsyncExecution::AsyncExecution(const Thunk* start_thunk)
    : start_thunk_(start_thunk) {}

AsyncExecution::ExecutionGuard::ExecutionGuard(se::Event* event,
                                               se::Stream* async_stream)
    : event_(event), async_stream_(async_stream) {}

AsyncExecution::ExecutionGuard::~ExecutionGuard() {
  // If we fail to record completion event on a stream it is unsafe to continue
  // as the following computations might not see all the updates done by the
  // async execution.
  CHECK_OK(async_stream_->RecordEvent(event_))  // Crash OK
      << "Failed to record async execution completion event " << event_
      << " on a stream " << async_stream_;
}

AsyncExecution::EventPool& AsyncExecution::GetOrCreatePool(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(&mu_);
  auto [it, _] = event_pools_.try_emplace(
      executor, [executor] { return executor->CreateEvent(); });
  return it->second;
}

absl::Status AsyncExecution::Initialize(Thunk::ExecutionScopedState* state,
                                        se::StreamExecutor* executor) {
  XLA_VLOG_DEVICE(1, executor->device_ordinal())
      << absl::StreamFormat("Initialize async execution for `%s`",
                            start_thunk_->profile_annotation());
  EventPool& pool = GetOrCreatePool(executor);
  ASSIGN_OR_RETURN(auto borrowed, pool.GetOrCreate());
  auto [_, emplaced] = state->try_emplace(
      start_thunk_->thunk_info().thunk_id,
      std::in_place_type<EventPool::BorrowedObject>, std::move(borrowed));
  return emplaced ? absl::OkStatus()
                  : Internal("Async execution initialized multiple times");
}

static absl::StatusOr<se::Event*> GetEvent(Thunk::ExecutionScopedState* state,
                                           ThunkId thunk_id) {
  auto it = state->find(thunk_id);
  if (it == state->end()) {
    return Internal("Async execution event not found for thunk %v", thunk_id);
  }
  auto* borrowed =
      tsl::any_cast<AsyncExecution::EventPool::BorrowedObject>(&it->second);
  if (!borrowed) {
    return Internal("Async execution state has wrong type for thunk %v",
                    thunk_id);
  }
  return (*borrowed)->get();
}

absl::StatusOr<AsyncExecution::ExecutionGuard> AsyncExecution::Start(
    RunId run_id, Thunk::ExecutionScopedState* state, se::Stream* stream,
    se::Stream* async_stream) {
  XLA_VLOG_DEVICE(1, stream->parent()->device_ordinal()) << absl::StreamFormat(
      "Start async execution for `%s`: run_id=%v; stream=%p, async_stream=%p",
      start_thunk_->profile_annotation(), run_id, stream, async_stream);
  ASSIGN_OR_RETURN(se::Event * event,
                   GetEvent(state, start_thunk_->thunk_info().thunk_id));

  // Record an event on stream and wait for it on async_stream so that
  // operations launched on `async_stream` observe all prior operations on
  // `stream`.
  RETURN_IF_ERROR(stream->RecordEvent(event));
  RETURN_IF_ERROR(async_stream->WaitFor(event));

  return ExecutionGuard(event, async_stream);
}

absl::Status AsyncExecution::Done(Thunk::ExecutionScopedState* state,
                                  se::Stream* stream) {
  XLA_VLOG_DEVICE(1, stream->parent()->device_ordinal())
      << absl::StreamFormat("Done async execution for `%s`: stream=%p",
                            start_thunk_->profile_annotation(), stream);
  ASSIGN_OR_RETURN(se::Event * event,
                   GetEvent(state, start_thunk_->thunk_info().thunk_id));

  // Wait for the async operation to complete by waiting for the event that was
  // recorded by the ExecutionGuard destructor at the end of Start.
  return stream->WaitFor(event);
}

}  // namespace xla::gpu

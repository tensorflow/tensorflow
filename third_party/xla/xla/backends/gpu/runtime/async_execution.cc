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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/unique_any.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

// Per-execution state stored in ExecutionScopedState. Wraps a borrowed event
// and a counter tracking the async execution lifecycle. Start increments the
// counter and Done decrements it. Each pair of AsyncStart/AsyncDone thunks can
// have at most one open async execution scope: Start returns an error if the
// counter is non-zero, and Done returns an error if the counter is zero. When
// destroyed at the end of an execution scope, a non-zero counter indicates
// that some async operations were not properly synchronized with the compute
// stream.
struct ExecutionState {
  ExecutionState(AsyncExecution::EventPool::BorrowedObject event)  // NOLINT
      : event(std::move(event)) {}

  ~ExecutionState() {
    DCHECK_EQ(counter, 0)
        << "Async execution was started but never completed "
           "(missing async-done synchronization with the compute stream)";
  }

  ExecutionState(ExecutionState&&) = default;
  ExecutionState& operator=(ExecutionState&&) = default;

  AsyncExecution::EventPool::BorrowedObject event;
  int32_t counter = 0;
};

}  // namespace

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
  absl::MutexLock lock(mu_);
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
  state->try_emplace(start_thunk_->thunk_info().thunk_id,
                     std::in_place_type<ExecutionState>, std::move(borrowed));
  // For shared async executions (e.g. pipelined send/recv), multiple
  // AsyncStartThunks share the same AsyncExecution and start_thunk_, so
  // Initialize may be called more than once with the same key. The first
  // call wins; subsequent calls are no-ops (borrowed event returns to pool).
  return absl::OkStatus();
}

static absl::StatusOr<ExecutionState*> GetExecutionState(
    Thunk::ExecutionScopedState* state, ThunkId thunk_id) {
  auto it = state->find(thunk_id);
  if (it == state->end()) {
    return Internal("Async execution state not found for thunk %v", thunk_id);
  }
  auto* execution_state = tsl::any_cast<ExecutionState>(&it->second);
  if (!execution_state) {
    return Internal("Async execution state has wrong type for thunk %v",
                    thunk_id);
  }
  return execution_state;
}

absl::StatusOr<AsyncExecution::ExecutionGuard> AsyncExecution::Start(
    Thunk::ExecutionScopedState* state, se::Stream* stream,
    se::Stream* async_stream) {
  XLA_VLOG_DEVICE(1, stream->parent()->device_ordinal()) << absl::StreamFormat(
      "Start async execution for `%s`: stream=%p, async_stream=%p",
      start_thunk_->profile_annotation(), stream, async_stream);
  ASSIGN_OR_RETURN(
      ExecutionState * es,
      GetExecutionState(state, start_thunk_->thunk_info().thunk_id));

  if (++es->counter > 1) {
    return Internal(
        "Async execution for `%s` already started (counter=%d). Async "
        "execution must be completed by Done before it can be started again.",
        start_thunk_->profile_annotation(), es->counter - 1);
  }

  se::Event* event = es->event->get();

  // Wait for all prior operations on `stream` before launching operations on
  // `async_stream`. We use a stream-level wait (not the shared event) so that
  // the event remains exclusively used for the async→main completion signal.
  // This is critical for pipelined send/recv where multiple Start() calls can
  // happen before Done() (the event is safely overwritten on the async stream
  // because the stream is ordered).
  RETURN_IF_ERROR(async_stream->WaitFor(stream));

  return ExecutionGuard(event, async_stream);
}

absl::Status AsyncExecution::Done(Thunk::ExecutionScopedState* state,
                                  se::Stream* stream) {
  XLA_VLOG_DEVICE(1, stream->parent()->device_ordinal())
      << absl::StreamFormat("Done async execution for `%s`: stream=%p",
                            start_thunk_->profile_annotation(), stream);
  ASSIGN_OR_RETURN(
      ExecutionState * es,
      GetExecutionState(state, start_thunk_->thunk_info().thunk_id));

  if (--es->counter < 0) {
    return Internal("Async execution for `%s` not started (counter=%d)",
                    start_thunk_->profile_annotation(), es->counter + 1);
  }

  // Wait for the async operation to complete by waiting for the event that was
  // recorded by the ExecutionGuard destructor at the end of Start.
  return stream->WaitFor(es->event->get());
}

}  // namespace xla::gpu

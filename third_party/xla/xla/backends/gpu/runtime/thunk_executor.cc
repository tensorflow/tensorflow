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

#include "xla/backends/gpu/runtime/thunk_executor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/event_pool.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Executing Thunks.
//===----------------------------------------------------------------------===//

// A lightweight wrapper around the while loop nest span that defers string
// formatting until AbslStringify is called (i.e., when VLOG is enabled).
struct LoopNest {
  absl::Span<const WhileLoopState> nest;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const LoopNest& loop_nest) {
    for (const auto& state : loop_nest.nest) {
      absl::Format(&sink, " [%s iter=%d]", state.loop_name,
                   state.loop_iteration);
    }
  }
};

ThunkExecutor::ThunkExecutor(ThunkSequence thunks)
    : thunks_(std::move(thunks)) {}

absl::Status ThunkExecutor::Prepare(const Thunk::PrepareParams& params) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    ABSL_RETURN_IF_ERROR(thunk->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status ThunkExecutor::Initialize(const Thunk::InitializeParams& params) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    ABSL_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status ThunkExecutor::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  auto* progress_tracker = ScopedProgressTracker::installed;
  auto* definition_tracker = ScopedDefinitionTracker::installed;
  int32_t device_ordinal = params.stream->parent()->device_ordinal();

  for (size_t i = 0; i < thunks_.size(); ++i) {
    const std::unique_ptr<Thunk>& thunk = thunks_[i];
    tsl::profiler::TraceMe trace(thunk->profile_annotation());

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(thunk->profile_annotation());

    // If progress tracker is installed for current thread, verify that a
    // thunk indexing record exists for the given `thunk`.
    if (progress_tracker) {
      if (!progress_tracker->indexing.contains(thunk.get())) {
        return Internal(
            "[thunk=%d/%d] Progress tracker is missing a record for thunk `%s`",
            i, thunks_.size(), thunk->profile_annotation());
      }
    }

    if (params.mock_collectives && thunk->IsCollective()) {
      XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
          "[thunk=%d/%d] Skip ThunkExecutor::ExecuteOnStream: %s (%v)", i,
          thunks_.size(), thunk->profile_annotation(), thunk->kind());
      continue;
    }

    LoopNest loop_nest = {IsInsideWhileLoopNest()};

    XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
        "[thunk=%d/%d] Start ThunkExecutor::ExecuteOnStream: %s (%v)%v", i,
        thunks_.size(), thunk->profile_annotation(), thunk->kind(), loop_nest);

    // Execute thunk and launch "work" on the GPU stream.
    ABSL_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));

    // Maybe notify the caller that all work touching buffer allocations has
    // been scheduled. Nested executors observe the same thread-local tracker,
    // and use their own ExecuteParams::stream when invoking the callback.
    if (definition_tracker) {
      auto it = definition_tracker->plan.find(thunk.get());
      if (it != definition_tracker->plan.end()) {
        ABSL_RETURN_IF_ERROR(
            definition_tracker->callback(params.stream, it->second));
      }
    }

    // Maybe track thunk execution to report the progress.
    if (progress_tracker) {
      // Borrow an event from the pool and record it on the execution stream.
      ABSL_ASSIGN_OR_RETURN(auto event,
                       progress_tracker->event_pool->GetOrCreateEvent());
      ABSL_RETURN_IF_ERROR(params.stream->RecordEvent(event->get()));

      absl::MutexLock lock(progress_tracker->mu);
      progress_tracker->events.emplace_back(thunk.get(), std::move(event),
                                            loop_nest.nest);
    }

    XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
        "[thunk=%d/%d] End ThunkExecutor::ExecuteOnStream: %s (%v)%v", i,
        thunks_.size(), thunk->profile_annotation(), thunk->kind(), loop_nest);
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Tracking buffer definitions.
//===----------------------------------------------------------------------===//

// Thunks nested under these wrappers do not necessarily schedule the last
// device work touching their buffer allocations. Use the wrapper as the
// definition boundary.
static bool IsDefinitionBarrier(Thunk::Kind kind) {
  return kind == Thunk::kWhile || kind == Thunk::kCommandBuffer ||
         kind == Thunk::kGroup;
}

// Definition events are not reported for allocations touched by host execution
// thunks. This is conservative and should be revisited once host execution can
// be sequenced precisely.
static bool IsHostExecutionThunk(Thunk::Kind kind) {
  return kind == Thunk::kHostExecuteDone || kind == Thunk::kHostExecuteStart ||
         kind == Thunk::kHostRecv || kind == Thunk::kHostRecvDone ||
         kind == Thunk::kHostSend || kind == Thunk::kHostSendDone;
}

ThunkExecutor::DefinitionPlan ThunkExecutor::BuildDefinitionPlan(
    const ThunkExecutor& executor) {
  absl::flat_hash_map<BufferAllocation::Index, const Thunk*> last_use;
  absl::flat_hash_set<BufferAllocation::Index> host_touched_allocations;
  const Thunk* definition_barrier = nullptr;

  auto pre_order = [&](const Thunk* thunk) {
    // Buffer definitions are attributed to the outermost enclosing barrier,
    // because thunks nested inside it may not schedule the final device work
    // that touches an allocation.
    if (!definition_barrier && IsDefinitionBarrier(thunk->kind())) {
      definition_barrier = thunk;
    }
    const Thunk* definition_thunk =
        definition_barrier ? definition_barrier : thunk;

    for (const BufferUse& use : thunk->buffer_uses()) {
      BufferAllocation::Index index = use.slice().index();
      if (IsHostExecutionThunk(thunk->kind())) {
        host_touched_allocations.insert(index);
        last_use.erase(index);
      } else if (!host_touched_allocations.contains(index)) {
        last_use[index] = definition_thunk;
      }
    }
  };

  auto post_order = [&](const Thunk* thunk) {
    if (thunk == definition_barrier) {
      definition_barrier = nullptr;
    }
  };

  for (const std::unique_ptr<Thunk>& thunk : executor.thunks()) {
    thunk->Walk(pre_order, post_order);
  }

  ThunkExecutor::DefinitionPlan plan;
  for (const auto& [index, thunk] : last_use) {
    plan[thunk].push_back(index);
  }
  return plan;
}

thread_local ThunkExecutor::ScopedDefinitionTracker::DefinitionTracker*
    ThunkExecutor::ScopedDefinitionTracker::installed = nullptr;

ThunkExecutor::ScopedDefinitionTracker::ScopedDefinitionTracker(
    const DefinitionPlan& plan, DefinitionCallback callback)
    : tracker_(std::make_unique<DefinitionTracker>(plan, callback)) {
  CHECK_EQ(installed, nullptr);
  installed = tracker_.get();
}

ThunkExecutor::ScopedDefinitionTracker::~ScopedDefinitionTracker() {
  if (tracker_) {
    CHECK_EQ(installed, tracker_.get());
    installed = nullptr;
  }
}

absl::StatusOr<ThunkExecutor::ScopedDefinitionTracker> InstallDefinitionTracker(
    const ThunkExecutor::DefinitionPlan& plan,
    ThunkExecutor::DefinitionCallback callback) {
  return ThunkExecutor::ScopedDefinitionTracker(plan, callback);
}

//===----------------------------------------------------------------------===//
// Tracking Thunk execution progress.
//===----------------------------------------------------------------------===//

using ThunkExecution = ThunkExecutor::ScopedProgressTracker::ThunkExecution;

thread_local ThunkExecutor::ScopedProgressTracker::ProgressTracker*
    ThunkExecutor::ScopedProgressTracker::installed = nullptr;

ThunkExecutor::ScopedProgressTracker::ThunkExecutionEvent::ThunkExecutionEvent(
    const Thunk* thunk, EventPool::Event event,
    absl::Span<const WhileLoopState> loop_nest)
    : thunk(thunk),
      executed(absl::Now()),
      event(std::move(event)),
      loop_nest(loop_nest.begin(), loop_nest.end()) {}

ThunkExecutor::ScopedProgressTracker::ScopedProgressTracker(
    EventPool* event_pool, ThunkIndexing indexing)
    : tracker_(
          std::make_unique<ProgressTracker>(std::move(indexing), event_pool)) {
  CHECK_EQ(installed, nullptr)  // Crash OK
      << "Tried to install multiple progress trackers";
  installed = tracker_.get();
}

ThunkExecutor::ScopedProgressTracker::~ScopedProgressTracker() {
  if (tracker_) {  // Skip moved-from ScopedProgressTracker
    CHECK_EQ(installed, tracker_.get())  // Crash OK
        << "Tried to destroy progress tracker on a different thread";
    installed = nullptr;
  }
}

size_t ThunkExecutor::ScopedProgressTracker::num_executions() const {
  absl::MutexLock lock(tracker_->mu);
  return tracker_->events.size();
}

size_t ThunkExecutor::ScopedProgressTracker::NumPendingThunks() {
  absl::MutexLock lock(tracker_->mu);
  return absl::c_count_if(tracker_->events, [](const auto& event) {
    return event.event->get()->PollForStatus() == se::Event::Status::kPending;
  });
}

size_t ThunkExecutor::ScopedProgressTracker::NumCompletedThunks() {
  absl::MutexLock lock(tracker_->mu);
  return absl::c_count_if(tracker_->events, [](const auto& event) {
    return event.event->get()->PollForStatus() == se::Event::Status::kComplete;
  });
}

std::vector<ThunkExecution> ThunkExecutor::ScopedProgressTracker::CollectThunks(
    se::Event::Status status, bool most_recent_first, size_t n) {
  absl::MutexLock lock(tracker_->mu);

  ThunkIndexing& indexing = tracker_->indexing;
  absl::Span<const ThunkExecutionEvent> events = tracker_->events;

  // Events are naturally in chronological order (oldest first). Iterate forward
  // for oldest-first or backward for most-recent-first.
  std::vector<ThunkExecution> result;

  auto collect = [&](size_t exec_idx, const ThunkExecutionEvent& event) {
    if (event.event->get()->PollForStatus() == status) {
      result.push_back({exec_idx, indexing.at(event.thunk), event.executed,
                        event.thunk->kind(), event.thunk->profile_annotation(),
                        event.loop_nest});
    }
  };

  if (most_recent_first) {
    for (size_t i = events.size(); i > 0; --i) {
      if (result.size() >= n) {
        break;
      }
      collect(i - 1, events[i - 1]);
    }
  } else {
    for (size_t i = 0; i < events.size(); ++i) {
      if (result.size() >= n) {
        break;
      }
      collect(i, events[i]);
    }
  }

  return result;
}

std::vector<ThunkExecution>
ThunkExecutor::ScopedProgressTracker::LastCompletedThunks(size_t n) {
  return CollectThunks(se::Event::Status::kComplete, /*most_recent_first=*/true,
                       n);
}

std::vector<ThunkExecution>
ThunkExecutor::ScopedProgressTracker::FirstPendingThunks(size_t n) {
  return CollectThunks(se::Event::Status::kPending,
                       /*most_recent_first=*/false, n);
}

std::vector<ThunkExecution>
ThunkExecutor::ScopedProgressTracker::LastPendingThunks(size_t n) {
  return CollectThunks(se::Event::Status::kPending, /*most_recent_first=*/true,
                       n);
}

absl::StatusOr<ThunkExecutor::ScopedProgressTracker> InstallProgressTracker(
    se::StreamExecutor* stream_executor, ThunkExecutor& executor) {
  tsl::profiler::TraceMe trace("InstallProgressTracker");

  ThunkExecutor::ScopedProgressTracker::ThunkIndexing indexing;
  ABSL_RETURN_IF_ERROR(
      executor.thunks().WalkNested([&](Thunk* thunk) -> absl::Status {
        size_t index = indexing.size();
        indexing[thunk] = index;
        return absl::OkStatus();
      }));

  XLA_VLOG_DEVICE(1, stream_executor->device_ordinal()) << absl::StreamFormat(
      "Installed progress tracker for %d thunks", indexing.size());

  return ThunkExecutor::ScopedProgressTracker(
      stream_executor->GetOrConstructResource<EventPool>(stream_executor),
      std::move(indexing));
}

}  // namespace xla::gpu

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
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/event_pool.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

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
    RETURN_IF_ERROR(thunk->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status ThunkExecutor::Initialize(const Thunk::InitializeParams& params) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status ThunkExecutor::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  auto* tracker = ScopedProgressTracker::installed_progress_tracker;
  int32_t device_ordinal = params.stream->parent()->device_ordinal();

  for (size_t i = 0; i < thunks_.size(); ++i) {
    const std::unique_ptr<Thunk>& thunk = thunks_[i];
    tsl::profiler::TraceMe trace(thunk->profile_annotation());

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(thunk->profile_annotation());

    // If progress tracker is installed for current thread, verify that a
    // thunk indexing record exists for the given `thunk`.
    if (tracker) {
      if (!tracker->indexing.contains(thunk.get())) {
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
    RETURN_IF_ERROR(thunk->ExecuteOnStream(params));

    // Maybe track thunk execution to report the progress.
    if (tracker) {
      // Borrow an event from the pool and record it on the execution stream.
      ASSIGN_OR_RETURN(auto event, tracker->event_pool->GetOrCreateEvent());
      RETURN_IF_ERROR(params.stream->RecordEvent(event->get()));

      absl::MutexLock lock(tracker->mu);
      tracker->events.emplace_back(thunk.get(), std::move(event),
                                   loop_nest.nest);
    }

    XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
        "[thunk=%d/%d] End ThunkExecutor::ExecuteOnStream: %s (%v)%v", i,
        thunks_.size(), thunk->profile_annotation(), thunk->kind(), loop_nest);
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Tracking Thunk execution progress.
//===----------------------------------------------------------------------===//

using ThunkExecution = ThunkExecutor::ScopedProgressTracker::ThunkExecution;

thread_local ThunkExecutor::ScopedProgressTracker::ProgressTracker*
    ThunkExecutor::ScopedProgressTracker::installed_progress_tracker = nullptr;

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
  CHECK_EQ(installed_progress_tracker, nullptr)  // Crash OK
      << "Tried to install multiple progress trackers";
  installed_progress_tracker = tracker_.get();
}

ThunkExecutor::ScopedProgressTracker::~ScopedProgressTracker() {
  if (tracker_) {  // Skip moved-from ScopedProgressTracker
    CHECK_EQ(installed_progress_tracker, tracker_.get())  // Crash OK
        << "Tried to destroy progress tracker on a different thread";
    installed_progress_tracker = nullptr;
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
  RETURN_IF_ERROR(
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

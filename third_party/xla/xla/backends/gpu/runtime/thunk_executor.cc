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
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

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
    // thunk progress record exists for the given `thunk`.
    if (tracker) {
      absl::MutexLock lock(tracker->mu);
      if (!tracker->map.contains(thunk.get())) {
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

    XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
        "[thunk=%d/%d] Start ThunkExecutor::ExecuteOnStream: %s (%v)", i,
        thunks_.size(), thunk->profile_annotation(), thunk->kind());

    // Execute thunk and launch "work" on the GPU stream.
    RETURN_IF_ERROR(thunk->ExecuteOnStream(params));

    // Maybe track thunk execution to report the progress.
    if (tracker) {
      absl::MutexLock lock(tracker->mu);
      tracker->map.at(thunk).RecordExecution(IsInsideWhileLoopNest());

      // Record execution completion event on a stream.
      se::Stream* execution_stream = params.stream;

      // Async collectives launch work on a dedicated async stream, so we
      // must record the event there instead of on the main compute stream.
      if (auto* collective = dynamic_cast<const CollectiveThunk*>(thunk.get());
          thunk->IsAsyncStart() && collective && params.collective_params) {
        execution_stream = params.collective_params->async_streams.at(
            thunk->execution_stream_id().value());
      }
      RETURN_IF_ERROR(
          execution_stream->RecordEvent(tracker->map.at(thunk).event.get()));
    }

    XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
        "[thunk=%d/%d] End ThunkExecutor::ExecuteOnStream: %s (%v)", i,
        thunks_.size(), thunk->profile_annotation(), thunk->kind());
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Tracking Thunk execution progress.
//===----------------------------------------------------------------------===//

using ThunkExecution = ThunkExecutor::ScopedProgressTracker::ThunkExecution;

void ThunkExecutor::ScopedProgressTracker::ThunkProgress::RecordExecution(
    absl::Span<const WhileLoopState> loop_nest) {
  executed = absl::Now();
  this->loop_nest.assign(loop_nest.begin(), loop_nest.end());
}

thread_local ThunkExecutor::ScopedProgressTracker::ThunkEvents*
    ThunkExecutor::ScopedProgressTracker::installed_progress_tracker = nullptr;

ThunkExecutor::ScopedProgressTracker::ScopedProgressTracker(
    absl::flat_hash_map<const Thunk*, ThunkProgress> progress_map)
    : events_(std::make_unique<ThunkEvents>(std::move(progress_map))) {
  CHECK_EQ(installed_progress_tracker, nullptr)  // Crash OK
      << "Tried to install multiple progress trackers";
  installed_progress_tracker = events_.get();
}

ThunkExecutor::ScopedProgressTracker::~ScopedProgressTracker() {
  if (events_ != nullptr) {  // Skip moved-from ScopedProgressTracker
    tsl::profiler::TraceMe trace("~ScopedProgressTracker");
    CHECK_EQ(installed_progress_tracker, events_.get())  // Crash OK
        << "Tried to destroy progress tracker on a different thread";
    installed_progress_tracker = nullptr;
    absl::MutexLock lock(events_->mu);
    events_->map.clear();
  }
}

size_t ThunkExecutor::ScopedProgressTracker::NumPendingThunks() {
  absl::MutexLock lock(events_->mu);
  size_t count = 0;
  for (auto& [thunk, progress] : events_->map) {
    if (progress.executed != absl::InfinitePast() &&
        progress.event->PollForStatus() == se::Event::Status::kPending) {
      ++count;
    }
  }
  return count;
}

std::vector<ThunkExecution> ThunkExecutor::ScopedProgressTracker::CollectThunks(
    se::Event::Status status, bool most_recent_first, size_t n) {
  absl::MutexLock lock(events_->mu);

  // Helper struct for sorting executed thunks by timestamp before lazily
  // polling event status. We keep a pointer to the map entry to avoid copying
  // and to defer the expensive PollForStatus call.
  struct ExecutedThunkEntry {
    const Thunk* thunk;
    ThunkProgress* progress;
  };

  // Collect all thunks that have been executed (cheap: just reads timestamps).
  std::vector<ExecutedThunkEntry> entries;
  entries.reserve(events_->map.size());
  for (auto& [thunk, progress] : events_->map) {
    if (progress.executed != absl::InfinitePast()) {
      entries.push_back({thunk, &progress});
    }
  }

  // Sort by executed time before checking event status.
  absl::c_stable_sort(
      entries, [most_recent_first](const auto& a, const auto& b) {
        return most_recent_first ? a.progress->executed > b.progress->executed
                                 : a.progress->executed < b.progress->executed;
      });

  // Lazily check event status and stop once we have enough results.
  std::vector<ThunkExecution> result;
  for (auto& entry : entries) {
    if (result.size() >= n) break;
    if (entry.progress->event->PollForStatus() == status) {
      result.push_back({entry.progress->index, entry.progress->executed,
                        entry.thunk->profile_annotation(),
                        entry.progress->loop_nest});
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

  using ThunkProgress = ThunkExecutor::ScopedProgressTracker::ThunkProgress;
  absl::flat_hash_map<const Thunk*, ThunkProgress> progress_map;

  RETURN_IF_ERROR(
      executor.thunks().WalkNested([&](Thunk* thunk) -> absl::Status {
        size_t index = progress_map.size();
        ASSIGN_OR_RETURN(auto event, stream_executor->CreateEvent());
        progress_map[thunk] = ThunkProgress{
            index, std::move(event), absl::InfinitePast(), /*loop_nest=*/{}};
        return absl::OkStatus();
      }));

  XLA_VLOG_DEVICE(1, stream_executor->device_ordinal()) << absl::StreamFormat(
      "Installed progress tracker for %d thunks", progress_map.size());

  return ThunkExecutor::ScopedProgressTracker(std::move(progress_map));
}

}  // namespace xla::gpu

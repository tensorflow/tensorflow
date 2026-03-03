/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/sequential_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

SequentialThunk::SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks)
    : Thunk(Kind::kSequential, thunk_info), thunks_(std::move(thunks)) {}

std::string SequentialThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  if (thunks_.empty()) {
    return indent_str + "No thunks.";
  }

  auto thunk_with_longest_kind = absl::c_max_element(
      thunks_,
      [](const std::unique_ptr<Thunk>& a, const std::unique_ptr<Thunk>& b) {
        return Thunk::KindToString(a->kind()).length() <
               Thunk::KindToString(b->kind()).length();
      });
  int64_t max_thunk_kind_len =
      Thunk::KindToString(thunk_with_longest_kind->get()->kind()).length();
  std::string result;
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    absl::StrAppend(&result, indent_str);
    absl::StrAppendFormat(&result,
                          "%03d: ", thunk->thunk_info().thunk_id.value());
    // Write out the thunk kind, padded out to max_thunk_kind_len.
    absl::string_view kind_str = Thunk::KindToString(thunk->kind());
    absl::StrAppend(&result, kind_str,
                    std::string(max_thunk_kind_len - kind_str.length(), ' '),
                    "\t");
    absl::StrAppend(&result, thunk->ToString(indent + 1));
    absl::StrAppend(&result, "\n");
  }
  return result;
}

absl::Status SequentialThunk::Prepare(const PrepareParams& params) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::Initialize(const InitializeParams& params) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  std::optional<tsl::profiler::ScopedAnnotation> seq_annotation =
      GetKernelAnnotation(profile_annotation());

  int32_t device_ordinal = params.stream->parent()->device_ordinal();

  for (size_t i = 0; i < thunks_.size(); ++i) {
    auto* tracker = ScopedProgressTracker::installed_progress_tracker;
    const std::unique_ptr<Thunk>& thunk = thunks_[i];

    tsl::profiler::TraceMe trace(thunk->profile_annotation());

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(thunk->profile_annotation());

    // If progress tracker is installed for current thread, verify that a
    // thunk progress record exists for the given `thunk`.
    if (tracker) {
      absl::MutexLock lock(&tracker->mu);
      if (!tracker->map.contains(thunk.get())) {
        return Internal(
            "[thunk=%d/%d] Progress tracker is missing a record for thunk `%s`",
            i, thunks_.size(), thunk->profile_annotation());
      }
    }

    if (params.mock_collectives && thunk->IsCollective()) {
      XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
          "[thunk=%d/%d] Skip SequentialThunk::ExecuteOnStream: %s", i,
          thunks_.size(), thunk->profile_annotation());
      continue;
    }

    XLA_VLOG_DEVICE(1, device_ordinal) << absl::StreamFormat(
        "[thunk=%d/%d] Start SequentialThunk::ExecuteOnStream: %s", i,
        thunks_.size(), thunk->profile_annotation());

    // Execute thunk and launch "work" on the GPU stream.
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));

    // Maybe track thunk execution to report the progress.
    if (tracker) {
      absl::MutexLock lock(&tracker->mu);
      // Record when thunk was executed last time.
      tracker->map.at(thunk).executed = absl::Now();

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
        "[thunk=%d/%d] End SequentialThunk::ExecuteOnStream: %s", i,
        thunks_.size(), thunk->profile_annotation());
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::WalkNested(Walker callback) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Walk(callback));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::TransformNested(Transformer callback) {
  for (std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->TransformNested(callback));
    TF_ASSIGN_OR_RETURN(thunk, callback(std::move(thunk)));
  }
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> SequentialThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  // This sets the oneof-type to the sequential thunk, even if the thunk list is
  // empty.
  proto.mutable_sequential_thunk();
  for (const auto& thunk : thunks_) {
    TF_ASSIGN_OR_RETURN(*proto.mutable_sequential_thunk()->add_thunks(),
                        thunk->ToProto());
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<SequentialThunk>> SequentialThunk::FromProto(
    ThunkInfo thunk_info, const SequentialThunkProto& thunk_proto,
    const Deserializer& deserializer) {
  ThunkSequence thunk_sequence;
  for (const auto& sub_thunk_proto : thunk_proto.thunks()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> sub_thunk,
                        deserializer(sub_thunk_proto));
    thunk_sequence.push_back(std::move(sub_thunk));
  }

  return std::make_unique<SequentialThunk>(std::move(thunk_info),
                                           std::move(thunk_sequence));
}

std::unique_ptr<SequentialThunk> SequentialThunk::FromThunk(
    std::unique_ptr<Thunk> thunk) {
  if (thunk->kind() == Thunk::kSequential) {
    return std::unique_ptr<SequentialThunk>(
        static_cast<SequentialThunk*>(thunk.release()));
  }

  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(thunk));
  return std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                           std::move(thunks));
}

//===----------------------------------------------------------------------===//
// Tracking XLA execution progress.
//===----------------------------------------------------------------------===//

using ThunkExecution = SequentialThunk::ScopedProgressTracker::ThunkExecution;

thread_local SequentialThunk::ScopedProgressTracker::ThunkEvents*
    SequentialThunk::ScopedProgressTracker::installed_progress_tracker =
        nullptr;

SequentialThunk::ScopedProgressTracker::ScopedProgressTracker(
    absl::flat_hash_map<const Thunk*, ThunkProgress> progress_map)
    : events_(std::make_unique<ThunkEvents>(std::move(progress_map))) {
  CHECK_EQ(installed_progress_tracker, nullptr)  // Crash OK
      << "Tried to install multiple progress trackers";
  installed_progress_tracker = events_.get();
}

SequentialThunk::ScopedProgressTracker::~ScopedProgressTracker() {
  if (events_ != nullptr) {  // Skip moved-from ScopedProgressTracker
    tsl::profiler::TraceMe trace("~ScopedProgressTracker");
    CHECK_EQ(installed_progress_tracker, events_.get())  // Crash OK
        << "Tried to destroy progress tracker on a different thread";
    installed_progress_tracker = nullptr;
    absl::MutexLock lock(&events_->mu);
    events_->map.clear();
  }
}

std::vector<ThunkExecution>
SequentialThunk::ScopedProgressTracker::CollectThunks(se::Event::Status status,
                                                      bool most_recent_first,
                                                      size_t n) {
  absl::MutexLock lock(&events_->mu);

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
  absl::c_sort(entries, [most_recent_first](const auto& a, const auto& b) {
    return most_recent_first ? a.progress->executed > b.progress->executed
                             : a.progress->executed < b.progress->executed;
  });

  // Lazily check event status and stop once we have enough results.
  std::vector<ThunkExecution> result;
  for (auto& entry : entries) {
    if (result.size() >= n) break;
    if (entry.progress->event->PollForStatus() == status) {
      result.push_back({entry.progress->index, entry.progress->executed,
                        entry.thunk->profile_annotation()});
    }
  }
  return result;
}

std::vector<ThunkExecution>
SequentialThunk::ScopedProgressTracker::LastCompletedThunks(size_t n) {
  return CollectThunks(se::Event::Status::kComplete, /*most_recent_first=*/true,
                       n);
}

std::vector<ThunkExecution>
SequentialThunk::ScopedProgressTracker::FirstPendingThunks(size_t n) {
  return CollectThunks(se::Event::Status::kPending,
                       /*most_recent_first=*/false, n);
}

std::vector<ThunkExecution>
SequentialThunk::ScopedProgressTracker::LastPendingThunks(size_t n) {
  return CollectThunks(se::Event::Status::kPending, /*most_recent_first=*/true,
                       n);
}

absl::StatusOr<SequentialThunk::ScopedProgressTracker> InstallProgressTracker(
    se::StreamExecutor* stream_executor, const SequentialThunk& thunk) {
  tsl::profiler::TraceMe trace("InstallProgressTracker");

  using ThunkProgress = SequentialThunk::ScopedProgressTracker::ThunkProgress;
  absl::flat_hash_map<const Thunk*, ThunkProgress> progress_map;

  RETURN_IF_ERROR(thunk.Walk([&](const Thunk* thunk) -> absl::Status {
    size_t index = progress_map.size();
    ASSIGN_OR_RETURN(auto event, stream_executor->CreateEvent());
    progress_map[thunk] = {index, absl::InfinitePast(), std::move(event)};
    return absl::OkStatus();
  }));

  XLA_VLOG_DEVICE(1, stream_executor->device_ordinal()) << absl::StreamFormat(
      "Installed progress tracker for %d thunks", progress_map.size());

  return SequentialThunk::ScopedProgressTracker(std::move(progress_map));
}

}  // namespace xla::gpu

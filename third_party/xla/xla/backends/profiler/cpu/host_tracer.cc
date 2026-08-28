/* Copyright 2018 The OpenXLA Authors.

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
#include "xla/backends/profiler/cpu/host_tracer.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/profiler/backends/cpu/host_tracer_utils.h"
#include "xla/tsl/profiler/backends/cpu/threadpool_listener.h"
#include "xla/tsl/profiler/backends/cpu/traceme_recorder.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

#define TRACEME_FILTER_DEFAULT_MASK std::numeric_limits<uint64_t>::max()

// Controls TraceMeRecorder and converts TraceMeRecorder::Events into XEvents.
//
// Thread-safety: This class is go/thread-compatible.
class HostTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit HostTracer(int host_trace_level);
  explicit HostTracer(int host_trace_level, uint64_t filter_mask,
                      bool enable_source_location = true);
  ~HostTracer() override;

  absl::Status Start() override;  // TENSORFLOW_STATUS_OK

  absl::Status Stop() override;  // TENSORFLOW_STATUS_OK

  absl::Status CollectData(  // TENSORFLOW_STATUS_OK
      tensorflow::profiler::XSpace* space) override;

  absl::StatusOr<tsl::profiler::ConsumeResult> Consume() override;

  absl::Status Serialize(std::any data,
                         tensorflow::profiler::XSpace* space) override;

 private:
  // Level of host tracing.
  const int host_trace_level_;

  // Filter mask for host tracing.
  const uint64_t filter_mask_;

  // Whether to enable source location in TraceMe encode.
  const bool enable_source_location_;

  // True if currently recording.
  bool recording_ = false;

  // Timestamp at the start of tracing.
  uint64_t start_timestamp_ns_ = 0;

  // Container of all traced events.
  tsl::profiler::TraceMeRecorder::Events events_;
};

HostTracer::HostTracer(int host_trace_level)
    : host_trace_level_(host_trace_level),
      filter_mask_(TRACEME_FILTER_DEFAULT_MASK),
      enable_source_location_(true) {}

HostTracer::HostTracer(int host_trace_level, uint64_t filter_mask,
                       bool enable_source_location)
    : host_trace_level_(host_trace_level),
      filter_mask_(filter_mask),
      enable_source_location_(enable_source_location) {}

HostTracer::~HostTracer() { Stop().IgnoreError(); }  // NOLINT

absl::Status HostTracer::Start() {  // TENSORFLOW_STATUS_OK
  VLOG(1) << "HostTracer::Start called";
  if (recording_) {
    return absl::InternalError("TraceMeRecorder already started");
  }

  // All TraceMe captured should have a timestamp greater or equal to
  // start_timestamp_ns_ to prevent timestamp underflow in XPlane.
  // Therefore this have to be done before TraceMeRecorder::Start.
  start_timestamp_ns_ = tsl::profiler::GetCurrentTimeNanos();
  recording_ = tsl::profiler::TraceMeRecorder::Start(
      host_trace_level_, filter_mask_, enable_source_location_);
  if (!recording_) {
    return absl::InternalError("Failed to start TraceMeRecorder");
  }
  return absl::OkStatus();
}

absl::Status HostTracer::Stop() {  // TENSORFLOW_STATUS_OK
  if (!recording_) {
    return absl::InternalError("TraceMeRecorder not started");
  }
  events_ = tsl::profiler::TraceMeRecorder::Stop();
  recording_ = false;
  return absl::OkStatus();
}

absl::Status HostTracer::CollectData(  // TENSORFLOW_STATUS_OK
    tensorflow::profiler::XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from HostTracer.";
  if (recording_) {
    return absl::InternalError("TraceMeRecorder not stopped");
  }
  if (events_.empty()) {
    return absl::OkStatus();
  }
  tensorflow::profiler::XPlane* plane =
      tsl::profiler::FindOrAddMutablePlaneWithName(
          space, tsl::profiler::kHostThreadsPlaneName);
  ConvertCompleteEventsToXPlane(start_timestamp_ns_, std::exchange(events_, {}),
                                plane);
  return absl::OkStatus();
}

absl::StatusOr<tsl::profiler::ConsumeResult> HostTracer::Consume() {
  VLOG(1) << "HostTracer::Consume called, recording=" << recording_;
  tsl::profiler::TraceMeRecorder::Events events;
  if (recording_) {
    events = tsl::profiler::TraceMeRecorder::Flush();
  } else {
    events = std::exchange(events_, {});
  }
  uint64_t current_chunk_start_ns = start_timestamp_ns_;
  uint64_t now_ns = tsl::profiler::GetCurrentTimeNanos();
  start_timestamp_ns_ = now_ns;

  size_t total_events = 0;
  size_t estimated_size = 0;
  constexpr size_t kAverageEventNameCapacity = 64;
  for (const auto& thread : events) {
    size_t num_events = thread.events.size();
    total_events += num_events;
    estimated_size += sizeof(thread.thread);
    estimated_size += thread.thread.name.capacity();
    estimated_size +=
        num_events * sizeof(tsl::profiler::TraceMeRecorder::Event);
    estimated_size += num_events * kAverageEventNameCapacity;
  }
  VLOG(1) << "HostTracer::Consume: flushed " << events.size()
          << " threads, total events=" << total_events;

  HostTracerChunk chunk;
  chunk.start_timestamp_ns = current_chunk_start_ns;
  chunk.events = std::move(events);

  tsl::profiler::ConsumeResult result;
  result.data = std::make_any<HostTracerChunk>(std::move(chunk));
  result.estimated_size_bytes = estimated_size;
  return result;
}

absl::Status HostTracer::Serialize(std::any data,
                                   tensorflow::profiler::XSpace* space) {
  VLOG(1) << "HostTracer::Serialize called";
  if (space == nullptr) {
    return absl::InvalidArgumentError("XSpace pointer cannot be null.");
  }
  auto* chunk = std::any_cast<HostTracerChunk>(&data);
  if (chunk == nullptr) {
    return absl::InvalidArgumentError("Invalid data type passed to Serialize.");
  }
  if (chunk->events.empty()) {
    VLOG(1) << "HostTracer::Serialize: events is empty, doing nothing";
    return absl::OkStatus();
  }
  size_t total_events = 0;
  for (const auto& thread : chunk->events) {
    total_events += thread.events.size();
  }
  VLOG(1) << "HostTracer::Serialize: serializing " << chunk->events.size()
          << " threads, total events=" << total_events;

  // TODO(sannidhya): Find a better way to align the events. This computation
  // of min event time might be redundant as it is also done later.
  uint64_t min_event_time_ns = 0;
  for (const auto& thread : chunk->events) {
    for (const auto& event : thread.events) {
      if (min_event_time_ns == 0 || event.start_time < min_event_time_ns) {
        min_event_time_ns = event.start_time;
      }
    }
  }

  uint64_t baseline_ns =
      (min_event_time_ns > 0) ? min_event_time_ns : chunk->start_timestamp_ns;

  tensorflow::profiler::XPlane* plane =
      tsl::profiler::FindOrAddMutablePlaneWithName(
          space, tsl::profiler::kHostThreadsPlaneName);
  ConvertCompleteEventsToXPlane(baseline_ns, std::move(chunk->events), plane);
  chunk->events.clear();
  return absl::OkStatus();
}

}  // namespace

std::unique_ptr<tsl::profiler::ProfilerInterface> CreateHostTracer(
    const HostTracerOptions& options) {
  if (options.trace_level == 0) return nullptr;
  std::vector<std::unique_ptr<tsl::profiler::ProfilerInterface>> profilers;
  profilers.push_back(
      std::make_unique<HostTracer>(options.trace_level, options.filter_mask,
                                   options.enable_source_location));
  profilers.push_back(
      std::make_unique<tsl::profiler::ThreadpoolProfilerInterface>());
  return std::make_unique<tsl::profiler::ProfilerCollection>(
      std::move(profilers));
}

}  // namespace profiler
}  // namespace xla

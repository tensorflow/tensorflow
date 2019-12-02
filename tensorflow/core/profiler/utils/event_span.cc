/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/utils/event_span.h"

#include <thread>  // NOLINT
#include <vector>

#include "absl/strings/match.h"

namespace tensorflow {
namespace profiler {

namespace {

// Returns the disjoint timespans from the given possibly overlapped events. The
// timespans are sorted in the begin time.
std::vector<Timespan> MakeDisjointTimespans(
    const std::vector<EventTypeSpan>& overlapped_events) {
  std::set<uint64> boundary_times;  // uses a set to eliminate duplicated times.
  for (const auto& event : overlapped_events) {
    boundary_times.insert(event.span.begin_ps());
    boundary_times.insert(event.span.end_ps());
  }
  uint64 prev_time;
  bool is_first = true;
  std::vector<Timespan> timespans;
  for (const auto current_time : boundary_times) {
    // current_time will be in ascending order for std::set.
    if (is_first) {
      is_first = false;
    } else {
      timespans.push_back(Timespan::FromEndPoints(prev_time, current_time));
    }
    prev_time = current_time;
  }
  return timespans;
}

// Assigns an event type to the given timespan. It is the type of the
// event that has the highest preference among all events that include this
// timespan.
EventType AssignEventType(
    const Timespan& timespan,
    const std::vector<EventTypeSpan>& sorted_overlapped_events) {
  EventType event_type = UNKNOWN_TIME;
  for (const auto& event : sorted_overlapped_events) {
    if (timespan.end_ps() < event.span.begin_ps()) {
      // Because sorted_overlapped_events is sorted in the event's begin time,
      // we are sure that timespan won't overlap with the rest of events.
      break;
    }
    if (!event.span.Includes(timespan)) continue;
    event_type = std::max(event_type, event.type);
  }
  return event_type;
}

// Compares two EventTypeSpans using their timespans.
bool CmpStartTime(const EventTypeSpan& a, const EventTypeSpan& b) {
  return a.span < b.span;
}

// Returns the EventTypeSpans corresponding to the given disjoint timespans.
std::vector<EventTypeSpan> AssignTypesToDisjointTimespans(
    const std::vector<Timespan>& disjoint_timespans,
    const std::vector<EventTypeSpan>& overlapped_events) {
  std::vector<EventTypeSpan> sorted_overlapped_events = overlapped_events;
  absl::c_sort(sorted_overlapped_events, CmpStartTime);

  std::vector<EventTypeSpan> non_overlapped_events;
  non_overlapped_events.reserve(disjoint_timespans.size());
  for (const auto& timespan : disjoint_timespans) {
    EventType event_type = AssignEventType(timespan, sorted_overlapped_events);
    non_overlapped_events.push_back({event_type, timespan});
  }
  return non_overlapped_events;
}

// Converts from overlapped events to non-overlapped events.
std::vector<EventTypeSpan> ToNonOverlappedEvents(
    const std::vector<EventTypeSpan>& overlapped_events) {
  std::vector<Timespan> disjoint_timespans =
      MakeDisjointTimespans(overlapped_events);
  std::vector<EventTypeSpan> non_overlapped_events =
      AssignTypesToDisjointTimespans(disjoint_timespans, overlapped_events);
  return non_overlapped_events;
}

void CombineStepDetails(const StepDetails& src, StepDetails* dst) {
  dst->AppendMarkers(src.Markers());
  dst->AppendEvents(src.Events());
}

}  // namespace.

EventType ClassifyGpuEvent(absl::string_view event_name) {
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYHtoD"))
    return HOST_TO_DEVICE;
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYDtoH"))
    return DEVICE_TO_HOST;
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYDtoD"))
    return DEVICE_TO_DEVICE;
  return DEVICE_COMPUTE;
}

EventType ClassifyCpuEvent(absl::string_view event_name, int64 correlation_id) {
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYHtoD"))
    return HOST_TO_DEVICE;
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYHtoH")) return HOST_TO_HOST;
  if (correlation_id >= 0 ||
      absl::StartsWithIgnoreCase(event_name, "ExecutorState::Process")) {
    return HOST_PREPARE;
  } else {
    if (absl::StartsWithIgnoreCase(event_name, "IteratorGetNext"))
      return HOST_WAIT_INPUT;
    return HOST_COMPUTE;
  }
}

std::string PrintEventType(EventType event_type) {
  switch (event_type) {
    case UNKNOWN_TIME:
      return "unknown_time";
    case HOST_COMPUTE:
      return "host_compute";
    case HOST_COMPILE:
      return "host_compile";
    case HOST_TO_HOST:
      return "host_to_host";
    case HOST_TO_DEVICE:
      return "host_to_device";
    case HOST_PREPARE:
      return "host_prepare";
    case HOST_WAIT_INPUT:
      return "host_wait_input";
    case DEVICE_TO_DEVICE:
      return "device_to_device";
    case DEVICE_TO_HOST:
      return "device_to_host";
    case DEVICE_COMPUTE:
      return "device_compute";
    case DEVICE_WAIT_DEVICE:
      return "device_wait_device";
    case DEVICE_WAIT_HOST:
      return "device_wait_host";
    default:
      return "unexpected";
  }
}

void CombineStepEvents(const StepEvents& src, StepEvents* dst) {
  for (const auto& step_details : src) {
    int64 step_id = step_details.first;
    const StepDetails& src_details = step_details.second;
    StepDetails* dst_details = &(*dst)[step_id];
    CombineStepDetails(src_details, dst_details);
  }
}

// Converts from overlapped step-events to non-overlapped step-events.
StepEvents ToNonOverlappedStepEvents(const StepEvents& overlapped_step_events) {
  size_t num_steps = overlapped_step_events.size();
  std::vector<std::thread> workers;
  workers.resize(num_steps);
  std::vector<int64> step_ids;
  step_ids.resize(num_steps);
  std::vector<std::vector<EventTypeSpan>> non_overlapped_events_per_worker;
  non_overlapped_events_per_worker.resize(num_steps);
  StepEvents non_overlapped_step_events;
  int64 i = 0;
  // Sets up 1 worker per step to convert overlapped events to non-overlapped
  // events.
  for (const auto& step_events : overlapped_step_events) {
    step_ids[i] = step_events.first;
    const auto& step_details = step_events.second;
    *non_overlapped_step_events[step_ids[i]].MutableMarkers() =
        step_details.Markers();
    const std::vector<EventTypeSpan>* overlapped_events =
        &step_details.Events();
    std::vector<EventTypeSpan>* non_overlapped_events =
        &non_overlapped_events_per_worker[i];
    workers[i] = std::thread([overlapped_events, non_overlapped_events]() {
      *non_overlapped_events = ToNonOverlappedEvents(*overlapped_events);
    });
    i += 1;
  }
  // Runs the workers in parallel.
  std::for_each(workers.begin(), workers.end(),
                [](std::thread& t) { t.join(); });
  // Moves non-overlapped events to the corresponding step in the map.
  for (i = 0; i < step_ids.size(); i++) {
    *non_overlapped_step_events[step_ids[i]].MutableEvents() =
        std::move(non_overlapped_events_per_worker[i]);
  }
  return non_overlapped_step_events;
}

void StepDetails::AddMarker(const StepMarker& m) { markers_.push_back(m); }

void StepDetails::AddEvent(const EventTypeSpan& e) { events_.push_back(e); }

void StepDetails::AppendMarkers(const std::vector<StepMarker>& other_markers) {
  markers_.insert(markers_.end(), other_markers.begin(), other_markers.end());
}

void StepDetails::AppendEvents(const std::vector<EventTypeSpan>& other_events) {
  events_.insert(events_.end(), other_events.begin(), other_events.end());
}

Timespan StepDetails::StepTime() const {
  // If there are multiple step-markers, uses the one that has the maximum
  // duration.
  Timespan max_steptime;
  for (const auto& marker : markers_) {
    const Timespan& timespan = marker.span;
    if (timespan.duration_ps() > max_steptime.duration_ps())
      max_steptime = timespan;
  }
  return max_steptime;
}

}  // namespace profiler
}  // namespace tensorflow

/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_UTIL_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_UTIL_H_

#include <vector>

#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

// Returns the resource name for the given (device_id, resource_id) in trace.
inline absl::string_view ResourceName(const Trace& trace, uint32_t device_id,
                                      uint32_t resource_id) {
  return trace.devices().at(device_id).resources().at(resource_id).name();
}

// Returns the resource name for the given event in trace.
inline absl::string_view ResourceName(const Trace& trace,
                                      const TraceEvent& event) {
  return ResourceName(trace, event.device_id(), event.resource_id());
}

// Functor that compares trace events for sorting.
// Trace events are sorted by timestamp_ps (ascending) and duration_ps
// (descending) so nested events are sorted from outer to innermost.
struct TraceEventsComparator {
  bool operator()(const TraceEvent* a, const TraceEvent* b) const {
    if (a->timestamp_ps() < b->timestamp_ps()) return true;
    if (a->timestamp_ps() > b->timestamp_ps()) return false;
    return (a->duration_ps() > b->duration_ps());
  }
};

// Creates a Timespan from a TraceEvent.
inline Timespan EventSpan(const TraceEvent& event) {
  return Timespan(event.timestamp_ps(), event.duration_ps());
}

// Creates a Timespan from a Trace.
inline Timespan TraceSpan(const Trace& trace) {
  return Timespan::FromEndPoints(trace.min_timestamp_ps(),
                                 trace.max_timestamp_ps());
}

// A flow of events in the trace-viewer.
// All events in the flow have the same flow_id.
using TraceEventFlow = std::vector<TraceEvent*>;

// In case the flow_id was re-used, split into individual flows based on the
// flow_entry_type.
std::vector<TraceEventFlow> SplitEventFlow(TraceEventFlow&& flow);

// Returns whether the flow is complete.
inline bool IsCompleteFlow(const TraceEventFlow& flow) {
  DCHECK(!flow.empty());
  return flow.front()->flow_entry_type() == TraceEvent::FLOW_START &&
         flow.back()->flow_entry_type() == TraceEvent::FLOW_END;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_EVENTS_UTIL_H_

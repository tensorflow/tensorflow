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
#include "tensorflow/core/profiler/convert/trace_viewer/trace_viewer_visibility.h"

#include <cstdint>

#include "absl/log/check.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

TraceViewerVisibility::TraceViewerVisibility(
    tsl::profiler::Timespan visible_span, uint64_t resolution_ps)
    : visible_span_(visible_span), resolution_ps_(resolution_ps) {}

bool TraceViewerVisibility::Visible(const TraceEvent& event) {
  // If visible_span_ is instant, we cannot usefully filter.
  if (visible_span_.Instant()) return true;

  // Events outside visible_span are not visible.
  tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
  if (!visible_span_.Overlaps(span)) return false;

  // If resolution is zero, no downsampling.
  if (resolution_ps_ == 0) return true;

  return VisibleAtResolution(event);
}

bool TraceViewerVisibility::VisibleAtResolution(const TraceEvent& event) {
  DCHECK_NE(resolution_ps_, 0);
  // A counter event is visible if its distance from the last visible counter
  // event in the same device is large enough. The first counter event in a
  // row is always visible.
  if (!event.has_resource_id()) {
#if 1
    // TODO(b/218368708): Streaming mode does not seem to work for counters:
    // even if more counter events are loaded, the chart does not refresh.
    // For now, the workaround is to make counters always visible.
    return true;
#else
    // TODO(b/218368708): Provided streaming mode works, we should use the
    // difference in counter values as a criteria for visibility: if the height
    // of the bar changes significantly, ignore the time between updates.
    CounterRowId counter_row_id(event.device_id(), event.name());
    auto iter = last_counter_timestamp_ps_.find(counter_row_id);
    bool found = (iter != last_counter_timestamp_ps_.end());
    bool visible =
        !found || ((event.timestamp_ps() - iter->second) >= resolution_ps_);
    if (visible) {
      if (found) {
        iter->second = event.timestamp_ps();
      } else {
        last_counter_timestamp_ps_.emplace(counter_row_id,
                                           event.timestamp_ps());
      }
    }
    return visible;
#endif
  }

  // An event is visible if its duration is large enough.
  tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
  bool visible = (span.duration_ps() >= resolution_ps_);

  auto& row = rows_[RowId(event.device_id(), event.resource_id())];

  // An event is visible if it is the first event at its nesting depth, or its
  // distance from the last visible event at the same depth is large enough.
  size_t depth = row.Depth(span.begin_ps());
  if (!visible) {
    auto last_end_timestamp_ps = row.LastEndTimestampPs(depth);
    visible = !last_end_timestamp_ps ||
              (span.begin_ps() - *last_end_timestamp_ps >= resolution_ps_);
  }

  // A flow event is visible if the first event in the flow is visible.
  // The first event in the flow is visible if the distance between its arrow
  // binding point and the previous visible arrow binding point is large enough.
  // The arrow binds to the end time of the complete event.
  if (event.has_flow_id()) {
    // Only compute visibility for the first event in the flow.
    auto result = flows_.try_emplace(event.flow_id(), visible);
    if (!visible) {
      if (result.second) {
        auto last_flow_timestamp_ps = row.LastFlowTimestampPs();
        result.first->second =
            !last_flow_timestamp_ps ||
            (span.end_ps() - *last_flow_timestamp_ps >= resolution_ps_);
      }
      visible = result.first->second;
    }
    // If we see the last event in the flow, remove it from the map. We don't
    // use flow_entry_type for determining the first event in the flow because
    // for cross-host flows it won't be FLOW_START.
    // This removal prevents the map from growing too large.
    if (event.flow_entry_type() == TraceEvent::FLOW_END) {
      flows_.erase(result.first);
    }
    if (visible) {
      row.SetLastFlowTimestampPs(span.end_ps());
    }
  }

  if (visible) {
    row.SetLastEndTimestampPs(depth, span.end_ps());
  }
  return visible;
}

void TraceViewerVisibility::SetVisibleAtResolution(const TraceEvent& event) {
  DCHECK_NE(resolution_ps_, 0);
  if (!event.has_resource_id()) {
    CounterRowId counter_row_id(event.device_id(), event.name());
    last_counter_timestamp_ps_.insert_or_assign(counter_row_id,
                                                event.timestamp_ps());

  } else {
    tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
    auto& row = rows_[RowId(event.device_id(), event.resource_id())];
    if (event.has_flow_id()) {
      if (event.flow_entry_type() == TraceEvent::FLOW_END) {
        flows_.erase(event.flow_id());
      } else {
        flows_.try_emplace(event.flow_id(), true);
      }
      row.SetLastFlowTimestampPs(span.end_ps());
    }
    size_t depth = row.Depth(span.begin_ps());
    row.SetLastEndTimestampPs(depth, span.end_ps());
  }
}

size_t TraceViewerVisibility::RowVisibility::Depth(
    uint64_t begin_timestamp_ps) const {
  size_t depth = 0;
  for (; depth < last_end_timestamp_ps_.size(); ++depth) {
    if (last_end_timestamp_ps_[depth] <= begin_timestamp_ps) break;
  }
  return depth;
}

}  // namespace profiler
}  // namespace tensorflow

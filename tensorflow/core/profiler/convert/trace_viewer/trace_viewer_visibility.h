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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_VIEWER_VISIBILITY_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_VIEWER_VISIBILITY_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_filter_interface.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

// Determines whether an event will be visible in trace viewer within a visible
// tsl::profiler::Timespan at a certain resolution.
// Events must be evaluated in order by timestamp, because when an event is
// determined to be visible, the internal state of this class is updated.
class TraceViewerVisibility {
 public:
  // Create with visible timespan and resolution (in picoseconds).
  // The visible timespan must have non-zero duration.
  // If resolution is zero, no events are downsampled.
  explicit TraceViewerVisibility(tsl::profiler::Timespan visible_span,
                                 uint64_t resolution_ps = 0);

  // Returns true if the event overlaps the visible span and is distinguishable
  // at resolution_ps.
  bool Visible(const TraceEvent& event);

  // Returns true if the event is distinguishable at resolution_ps.
  bool VisibleAtResolution(const TraceEvent& event);

  // Records that event is distinguishable at resolution_ps.
  void SetVisibleAtResolution(const TraceEvent& event);

  tsl::profiler::Timespan VisibleSpan() const { return visible_span_; }
  // TODO(tf-profiler) Rename ResolutionPs and resolution_ps to be more
  // self-explanatory (eg. MinDurationPs)
  uint64_t ResolutionPs() const { return resolution_ps_; }

 private:
  // Identifier for one Trace Viewer row.
  using RowId = std::pair<uint32_t /*device_id*/, uint32_t /*resource_id*/>;
  using CounterRowId = std::pair<uint32_t /*device_id*/, std::string /*name*/>;

  // Visibility for one Trace Viewer row.
  class RowVisibility {
   public:
    // Returns the nesting depth for an event at begin_timestamp_ps.
    size_t Depth(uint64_t begin_timestamp_ps) const;

    // Returns the end_timestamp_ps of the last visibile event at the given
    // nesting depth.
    std::optional<uint64_t> LastEndTimestampPs(size_t depth) const {
      std::optional<uint64_t> result;
      if (depth < last_end_timestamp_ps_.size()) {
        result = last_end_timestamp_ps_[depth];
      }
      return result;
    }

    // Returns the arrow timestamp of the last visible flow event.
    std::optional<uint64_t> LastFlowTimestampPs() const {
      return last_flow_timestamp_ps_;
    }

    // Sets the last visible timestamp at the given nesting depth.
    void SetLastEndTimestampPs(size_t depth, uint64_t timestamp_ps) {
      last_end_timestamp_ps_.resize(depth);
      last_end_timestamp_ps_.push_back(timestamp_ps);
    }

    // Sets the last visible arrow timestamp.
    void SetLastFlowTimestampPs(uint64_t timestamp_ps) {
      last_flow_timestamp_ps_ = timestamp_ps;
    }

   private:
    // Stack of most recently visible event end times. A stack is used to handle
    // nested events.
    std::vector<uint64_t> last_end_timestamp_ps_;

    // Timestamp of the arrow binding point of the last visible flow event.
    std::optional<uint64_t> last_flow_timestamp_ps_;
  };

  // Constructor arguments.
  tsl::profiler::Timespan visible_span_;
  uint64_t resolution_ps_;

  // Visibility data for all rows.
  absl::flat_hash_map<RowId, RowVisibility> rows_;

  // Visibility of flows.
  absl::flat_hash_map<uint64_t /*flow_id*/, bool> flows_;

  // Visibility data for counter events.
  absl::flat_hash_map<CounterRowId, uint64_t> last_counter_timestamp_ps_;
};

class TraceVisibilityFilter : public TraceEventsFilterInterface {
 public:
  // If visible_span.Instant(), all events are visible.
  // If resolution is 0.0, events aren't downsampled.
  TraceVisibilityFilter(tsl::profiler::Timespan visible_span, double resolution)
      : resolution_(resolution),
        visibility_(visible_span, ResolutionPs(visible_span.duration_ps())) {}

  tsl::profiler::Timespan VisibleSpan() const {
    return visibility_.VisibleSpan();
  }
  uint64_t ResolutionPs() const { return visibility_.ResolutionPs(); }

  void SetUp(const Trace& trace) override {
    // Update visible_span with trace bounds and recompute the resolution in
    // picoseconds.
    tsl::profiler::Timespan visible_span = VisibleSpan();
    uint64_t start_time_ps = visible_span.begin_ps();
    uint64_t end_time_ps = visible_span.end_ps();
    if (end_time_ps == 0 && trace.has_max_timestamp_ps()) {
      end_time_ps = trace.max_timestamp_ps();
    }
    if (start_time_ps == 0 && trace.has_min_timestamp_ps()) {
      start_time_ps = trace.min_timestamp_ps();
    }
    visible_span =
        tsl::profiler::Timespan::FromEndPoints(start_time_ps, end_time_ps);
    visibility_ = TraceViewerVisibility(
        visible_span, ResolutionPs(visible_span.duration_ps()));
  }

  // Updates the visibility based on `resolution`.
  void UpdateVisibility(double resolution) {
    resolution_ = resolution;
    visibility_ = TraceViewerVisibility(
        visibility_.VisibleSpan(),
        ResolutionPs(visibility_.VisibleSpan().duration_ps()));
  }

  bool Filter(const TraceEvent& event) override {
    return !visibility_.Visible(event);
  }

 private:
  // Returns the minimum duration in picoseconds that an event must have in
  // order to be visible.
  uint64_t ResolutionPs(uint64_t duration_ps) {
    return (resolution_ == 0.0) ? 0 : std::llround(duration_ps / resolution_);
  }

  double resolution_;  // number of visible events per row
  TraceViewerVisibility visibility_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_TRACE_VIEWER_TRACE_VIEWER_VISIBILITY_H_

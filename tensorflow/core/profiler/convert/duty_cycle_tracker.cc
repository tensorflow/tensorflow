/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/duty_cycle_tracker.h"

#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "xla/tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::Timespan;

DutyCycleTracker::ActiveTimeSpans::const_iterator
DutyCycleTracker::MergeOrInsert(const Timespan& timespan,
                                ActiveTimeSpans::const_iterator hint) {
  DCHECK(hint == active_time_spans_.end() ||
         hint == active_time_spans_.begin() ||
         hint->begin_ps() <= timespan.begin_ps());
  ActiveTimeSpans::const_iterator merge_begin = hint;
  while (merge_begin != active_time_spans_.end() &&
         merge_begin->end_ps() < timespan.begin_ps()) {
    ++merge_begin;
  }

  // timespan is fully contained in an existing timespan.
  if (merge_begin != active_time_spans_.end() &&
      merge_begin->Includes(timespan)) {
    return merge_begin;
  }

  ActiveTimeSpans::const_iterator merge_end = merge_begin;
  while (merge_end != active_time_spans_.end() &&
         merge_end->begin_ps() <= timespan.end_ps()) {
    ++merge_end;
  }
  if (merge_begin != merge_end) {
    Timespan merged = Timespan::FromEndPoints(
        std::min(timespan.begin_ps(), merge_begin->begin_ps()),
        std::max(timespan.end_ps(), std::prev(merge_end)->end_ps()));
    merge_end = active_time_spans_.erase(merge_begin, merge_end);
    return active_time_spans_.insert(merge_end, merged);
  } else {
    // There is no overlap with the existing timespans.
    return active_time_spans_.insert(merge_begin, timespan);
  }
}

void DutyCycleTracker::AddInterval(tsl::profiler::Timespan time_span,
                                   bool is_active) {
  total_time_span_.ExpandToInclude(time_span);
  if (!is_active) {
    return;
  }

  auto hint = active_time_spans_.lower_bound(time_span);
  if (hint != active_time_spans_.begin()) --hint;
  MergeOrInsert(time_span, hint);
}

void DutyCycleTracker::Union(const DutyCycleTracker& other) {
  total_time_span_.ExpandToInclude(other.total_time_span_);
  if (other.active_time_spans_.empty()) return;
  ActiveTimeSpans::const_iterator hint_it =
      active_time_spans_.lower_bound(*other.active_time_spans_.begin());
  if (hint_it != active_time_spans_.begin()) --hint_it;
  for (const auto& interval : other.active_time_spans_) {
    hint_it = MergeOrInsert(interval, hint_it);
  }
}

uint64_t DutyCycleTracker::GetActiveTimePs() const {
  uint64_t active_time_ps = 0;
  for (const auto& interval : active_time_spans_) {
    DCHECK(!interval.Empty());
    active_time_ps += interval.duration_ps();
  }
  return active_time_ps;
}

uint64_t DutyCycleTracker::GetIdleTimePs() const {
  return total_time_span_.duration_ps() - GetActiveTimePs();
}
}  // namespace profiler
}  // namespace tensorflow

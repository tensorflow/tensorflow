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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_TRACKER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_TRACKER_H_

#include <cstdint>

#include "absl/container/btree_set.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

// Tracks the active time intervals for a given TPU core.
// Disjoint intervals of time in ps for which this core was active.
class DutyCycleTracker {
 public:
  DutyCycleTracker() : active_time_spans_() {}
  void AddInterval(tsl::profiler::Timespan time_span, bool is_active);
  void Union(const DutyCycleTracker& other);
  uint64_t GetActiveTimePs() const;
  uint64_t GetIdleTimePs() const;
  uint64_t GetDurationPs() const { return total_time_span_.duration_ps(); }
  double DutyCycle() const {
    return tsl::profiler::SafeDivide(GetActiveTimePs(), GetDurationPs());
  }

 private:
  struct TimespanComparator {
    // Order by increasing begin_ps, then decreasing duration_ps.
    bool operator()(const tsl::profiler::Timespan& a,
                    const tsl::profiler::Timespan& b) const {
      return a.begin_ps() < b.begin_ps() || (a.begin_ps() == b.begin_ps() &&
                                             a.duration_ps() > b.duration_ps());
    }
  };
  using ActiveTimeSpans =
      absl::btree_set<tsl::profiler::Timespan, TimespanComparator>;

  /**
   * Merge or insert the given timespan into the set of active time spans.
   *
   * @param timespan The timespan to merge or insert.
   * @param hint The iterator indicating where to begin the merge search.
   * @return The iterator where the timespan was merged or inserted.
   */
  ActiveTimeSpans::const_iterator MergeOrInsert(
      const tsl::profiler::Timespan& timespan,
      ActiveTimeSpans::const_iterator hint);

  ActiveTimeSpans active_time_spans_;
  tsl::profiler::Timespan total_time_span_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_TRACKER_H_

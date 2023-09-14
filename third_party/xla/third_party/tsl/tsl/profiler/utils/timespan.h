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

#ifndef TENSORFLOW_TSL_PROFILER_UTILS_TIMESPAN_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_TIMESPAN_H_

#include <algorithm>
#include <string>

#include "absl/strings/str_cat.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/utils/math_utils.h"

namespace tsl {
namespace profiler {

// A Timespan is the time extent of an event: a pair of (begin, duration).
// Events may have duration 0 ("instant events") but duration can't be negative.
class Timespan {
 public:
  static Timespan FromEndPoints(uint64 begin_ps, uint64 end_ps) {
    DCHECK_LE(begin_ps, end_ps);
    return Timespan(begin_ps, end_ps - begin_ps);
  }

  explicit Timespan(uint64 begin_ps = 0, uint64 duration_ps = 0)
      : begin_ps_(begin_ps), duration_ps_(duration_ps) {}

  uint64 begin_ps() const { return begin_ps_; }
  uint64 middle_ps() const { return begin_ps_ + duration_ps_ / 2; }
  uint64 end_ps() const { return begin_ps_ + duration_ps_; }
  uint64 duration_ps() const { return duration_ps_; }

  // Returns true if the Timespan represents an instant in time (duration 0).
  bool Instant() const { return duration_ps() == 0; }

  // Returns true if this is an empty timespan.
  bool Empty() const { return begin_ps() == 0 && duration_ps() == 0; }

  // Note for Overlaps() and Includes(Timespan& other) below:
  //   We have a design choice whether the end-point comparison should be
  //   inclusive or exclusive. We decide to go for inclusive. The implication
  //   is that an instant timespan could belong to two consecutive intervals
  //   (e.g., Timespan(12, 0) will be included in both Timespan(11, 1) and
  //   Timespan(12, 1)). We think this is okay because the common scenario
  //   would be that we search for the interval that includes a point
  //   in time from left to right, and return the first interval found.

  // Returns true if the Timespan overlaps with other.
  bool Overlaps(const Timespan& other) const {
    return begin_ps() <= other.end_ps() && other.begin_ps() <= end_ps();
  }

  // Returns true if this Timespan includes the other.
  bool Includes(const Timespan& other) const {
    return begin_ps() <= other.begin_ps() && other.end_ps() <= end_ps();
  }

  // Returns true if time_ps is within this Timespan.
  bool Includes(uint64 time_ps) const { return Includes(Timespan(time_ps)); }

  // Returns the duration in ps that this Timespan overlaps with the other.
  uint64 OverlappedDurationPs(const Timespan& other) const {
    if (!Overlaps(other)) return 0;
    return std::min(end_ps(), other.end_ps()) -
           std::max(begin_ps(), other.begin_ps());
  }

  // Expands the timespan to include other.
  void ExpandToInclude(const Timespan& other) {
    *this = FromEndPoints(std::min(begin_ps(), other.begin_ps()),
                          std::max(end_ps(), other.end_ps()));
  }

  // Compares timespans by their begin time (ascending), duration (descending)
  // so nested spans are sorted from outer to innermost.
  bool operator<(const Timespan& other) const {
    if (begin_ps_ < other.begin_ps_) return true;
    if (begin_ps_ > other.begin_ps_) return false;
    return duration_ps_ > other.duration_ps_;
  }

  // Returns true if this timespan is equal to the given timespan.
  bool operator==(const Timespan& other) const {
    return begin_ps_ == other.begin_ps_ && duration_ps_ == other.duration_ps_;
  }

  // Returns a string that shows the begin and end times.
  std::string DebugString() const {
    return absl::StrCat("[", begin_ps(), ", ", end_ps(), "]");
  }

  // Compares timespans by their duration_ps (ascending), begin time
  // (ascending).
  static bool ByDuration(const Timespan& a, const Timespan& b) {
    if (a.duration_ps_ < b.duration_ps_) return true;
    if (a.duration_ps_ > b.duration_ps_) return false;
    return a.begin_ps_ < b.begin_ps_;
  }

 private:
  uint64 begin_ps_;
  uint64 duration_ps_;  // 0 for an instant event.
};

// Creates a Timespan from endpoints in picoseconds.
inline Timespan PicoSpan(uint64 start_ps, uint64 end_ps) {
  return Timespan::FromEndPoints(start_ps, end_ps);
}

// Creates a Timespan from endpoints in milliseconds.
inline Timespan MilliSpan(double start_ms, double end_ms) {
  return PicoSpan(MilliToPico(start_ms), MilliToPico(end_ms));
}

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_TIMESPAN_H_

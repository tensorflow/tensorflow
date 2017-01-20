/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_
#define TENSORFLOW_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"

namespace tensorflow {

class ClockCycleProfiler {
 public:
  ClockCycleProfiler() = default;

  // Start counting clock cycle.
  inline void Start() {
    CHECK(!IsStarted()) << "Profiler has been already started.";
    start_clock_ = GetCurrentClockCycleInternal();
  }

  // Stop counting clock cycle.
  inline void Stop() {
    CHECK(IsStarted()) << "Profiler is not started yet.";
    AccumulateClockCycle();
  }

  // Get how many times Start() is called.
  inline double GetCount() {
    CHECK(!IsStarted());
    return count_;
  }

  // Get average clock cycle.
  inline double GetAverageClockCycle() {
    CHECK(!IsStarted());
    return average_clock_cycle_;
  }

  // TODO(satok): Support more statistics (e.g. standard deviation)
  // Get worst clock cycle.
  inline double GetWorstClockCycle() {
    CHECK(!IsStarted());
    return worst_clock_cycle_;
  }

  // Dump statistics
  void DumpStatistics(const string& tag);

 private:
  inline uint64 GetCurrentClockCycleInternal() {
    const uint64 clockCycle = profile_utils::CpuUtils::GetCurrentClockCycle();
    if (clockCycle <= 0) {
      if (valid_) {
        LOG(WARNING) << "GetCurrentClockCycle is not implemented."
                     << " Return 1 instead.";
        valid_ = false;
      }
      return 1;
    } else {
      return clockCycle;
    }
  }

  inline bool IsStarted() const { return start_clock_ > 0; }

  inline void AccumulateClockCycle() {
    const uint64 now = GetCurrentClockCycleInternal();
    const double clock_diff = static_cast<double>(now - start_clock_);
    const double next_count = count_ + 1.0;
    const double next_count_inv = 1.0 / next_count;
    const double next_ave_cpu_clock =
        next_count_inv * (average_clock_cycle_ * count_ + clock_diff);
    count_ = next_count;
    average_clock_cycle_ = next_ave_cpu_clock;
    worst_clock_cycle_ = std::max(worst_clock_cycle_, clock_diff);
    start_clock_ = 0;
  }

  uint64 start_clock_{0};
  double count_{0.0};
  double average_clock_cycle_{0.0};
  double worst_clock_cycle_{0.0};
  bool valid_{true};

  TF_DISALLOW_COPY_AND_ASSIGN(ClockCycleProfiler);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_

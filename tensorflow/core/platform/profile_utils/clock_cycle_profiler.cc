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

#include "tensorflow/core/platform/profile_utils/clock_cycle_profiler.h"

#include <chrono>

namespace tensorflow {

void ClockCycleProfiler::DumpStatistics(const string& tag) {
  CHECK(!IsStarted());
  const double average_clock_cycle = GetAverageClockCycle();
  const double count = GetCount();
  const std::chrono::duration<double> average_time =
      profile_utils::CpuUtils::ConvertClockCycleToTime(
          static_cast<int64>(average_clock_cycle + 0.5));
  LOG(INFO) << tag << ": average = "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   average_time)
                   .count()
            << " us (" << average_clock_cycle << " cycles)"
            << ", count = " << count;
}

}  // namespace tensorflow

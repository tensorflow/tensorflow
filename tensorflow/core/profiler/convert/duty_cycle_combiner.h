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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_COMBINER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_COMBINER_H_

#include <sys/types.h>

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/profiler/convert/duty_cycle_tracker.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Responsible for combining the duty cycle trackers for all cores and chips.
class DutyCycleCombiner {
 public:
  // Combines the given core tracker with the tracker for the given chip.
  // NOTE: The given chip_id should be unique across all chips being combined.
  void CombineCore(const DutyCycleTracker& core_tracker, uint32_t chip_id) {
    chip_duty_cycle_trackers_[chip_id].Union(core_tracker);
  }

  // Combines the given chip tracker with the tracker for other chips.
  void CombineChip(const DutyCycleTracker& chip_tracker) {
    chip_active_time_ps_ += chip_tracker.GetActiveTimePs();
    chip_idle_time_ps_ += chip_tracker.GetIdleTimePs();
  }

  // Returns the total active time across all chips and cores.
  uint64_t GetTotalActiveTimePs() const {
    uint64_t total_busy_time_ps = chip_active_time_ps_;
    for (const auto& [chip_id, tracker] : chip_duty_cycle_trackers_) {
      total_busy_time_ps += tracker.GetActiveTimePs();
    }
    return total_busy_time_ps;
  }

  // Returns the total idle time across all chips and cores.
  uint64_t GetTotalIdleTimePs() const {
    uint64_t total_idle_time_ps = chip_idle_time_ps_;
    for (const auto& [chip_id, tracker] : chip_duty_cycle_trackers_) {
      total_idle_time_ps += tracker.GetIdleTimePs();
    }
    return total_idle_time_ps;
  }

 private:
  absl::flat_hash_map<uint32_t, DutyCycleTracker> chip_duty_cycle_trackers_;
  uint64_t chip_active_time_ps_ = 0;
  uint64_t chip_idle_time_ps_ = 0;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_DUTY_CYCLE_COMBINER_H_

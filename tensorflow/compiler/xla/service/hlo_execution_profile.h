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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EXECUTION_PROFILE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EXECUTION_PROFILE_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_profile_printer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloInstruction;

// Maps all HloInstructions and HloComputations in an HloModule to integers.
// These integers form the contiguous range [0, total_count()).
class HloProfileIndexMap {
 public:
  // Scans `module` to populate this instance of HloProfileIndexMap.
  explicit HloProfileIndexMap(const HloModule& module)
      : HloProfileIndexMap(module, {}) {}
  explicit HloProfileIndexMap(const HloModule& module,
                              absl::Span<const string> extra_metrics);

  HloProfileIndexMap(const HloProfileIndexMap&) = default;
  HloProfileIndexMap(HloProfileIndexMap&&) = default;

  HloProfileIndexMap& operator=(const HloProfileIndexMap&) = default;
  HloProfileIndexMap& operator=(HloProfileIndexMap&&) = default;

  size_t GetProfileIndexFor(const HloInstruction& instruction) const {
    return FindOrDie(instruction_to_profile_idx(), &instruction);
  }

  size_t GetProfileIndexFor(const HloComputation& computation) const {
    return FindOrDie(computation_to_profile_idx(), &computation);
  }

  size_t GetProfileIndexFor(const string& key) const {
    return xla::FindOrDie(extra_metric_to_profile_idx(), key);
  }

  size_t instruction_count() const {
    return instruction_to_profile_idx().size();
  }

  size_t computation_count() const {
    return computation_to_profile_idx().size();
  }

  size_t extra_metrics_count() const {
    return extra_metric_to_profile_idx().size();
  }

  size_t total_count() const {
    return instruction_count() + computation_count() + extra_metrics_count();
  }

  const std::unordered_map<const HloInstruction*, int64>&
  instruction_to_profile_idx() const {
    return instruction_to_profile_idx_;
  }

  const std::unordered_map<const HloComputation*, int64>&
  computation_to_profile_idx() const {
    return computation_to_profile_idx_;
  }

  const std::unordered_map<string, int64>& extra_metric_to_profile_idx() const {
    return extra_metric_to_profile_idx_;
  }

 private:
  std::unordered_map<const HloInstruction*, int64> instruction_to_profile_idx_;
  std::unordered_map<const HloComputation*, int64> computation_to_profile_idx_;
  std::unordered_map<string, int64> extra_metric_to_profile_idx_;
};

// Create an instance of `HloProfilePrinterData`.
std::unique_ptr<HloProfilePrinterData> CreateHloProfilePrinterData(
    const HloProfileIndexMap& hlo_profile_index_map,
    const HloCostAnalysis& cost_analysis, const string& entry_computation_name);

// Describes how much time each HLO operation took.
//
// Each HloComputation takes a certain number of cycles.  This class helps break
// down how much time each HLO took.
class HloExecutionProfile {
 public:
  using DeviceDescription = se::DeviceDescription;

  HloExecutionProfile(const HloProfilePrinterData* hlo_profile_printer_data,
                      const HloProfileIndexMap* hlo_profile_index_map);

  // Record how many cycles this HLO took to execute.
  void SetCyclesTakenBy(const HloInstruction* hlo, uint64 cycles_taken);

  // Returns how many cycles this HLO took to execute.  Profiling information
  // may not be available for some instructions in which case zero is returned.
  uint64 GetCyclesTakenBy(const HloInstruction& hlo) const;

  // Return the number of cycles this computation took to execute.
  uint64 total_cycles_executed(const HloComputation& computation) const {
    return profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(
        computation)];
  }

  // Record how many cycles a computation took to execute.
  void set_total_cycles_executed(const HloComputation& computation,
                                 uint64 total_cycles_executed) {
    profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(computation)] =
        total_cycles_executed;
  }

  // Record extra metric.
  void set_extra_metrics(const string& metric, uint64 value) {
    profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(metric)] =
        value;
  }

  // Returns a version of the execution profile suitable for performance
  // debugging; e.g. emits cycle counts, execution time at the nominal device
  // frequency, and the effective throughput given the provided cost_analysis
  // for the operations in a given computation. Returns an empty string if it
  // wasn't possible to generate a printable version.
  string ToString(const DeviceDescription& device_description) const {
    return PrintHloProfile(hlo_profile_printer_data_, profile_counters_.data(),
                           device_description.clock_rate_ghz());
  }

  std::vector<int64>* mutable_profile_counters() { return &profile_counters_; }
  const std::vector<int64>& profile_counters() const {
    return profile_counters_;
  }

 private:
  const HloProfilePrinterData& hlo_profile_printer_data_;
  const HloProfileIndexMap& hlo_profile_index_map_;

  // Stores per-Hlo profile counters.  This is the only thing that changes when
  // we execute an XLA computation.
  std::vector<int64> profile_counters_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EXECUTION_PROFILE_H_

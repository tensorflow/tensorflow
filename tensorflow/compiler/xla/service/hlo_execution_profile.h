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

#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloInstruction;

// Describes how much time each HLO operation took.
//
// Each HloComputation takes a certain number of cycles.  This class helps break
// down how much time each HLO took.
class HloExecutionProfile {
 public:
  using DeviceDescription = perftools::gputools::DeviceDescription;

  // Record how many cycles this HLO took to execute.
  void SetCyclesTakenBy(const HloInstruction* hlo, uint64 cycles_taken);

  // Returns how many cycles this HLO took to execute.  Profiling information
  // may not be available for some instructions in which case zero is returned.
  uint64 GetCyclesTakenBy(const HloInstruction& hlo) const;

  // Return the number of cycles this computation took to execute.
  uint64 total_cycles_executed(const HloComputation& computation) const {
    auto it = total_cycles_executed_.find(&computation);
    if (it != total_cycles_executed_.end()) {
      return it->second;
    }
    return 0;
  }

  // Record how many cycles a computation took to execute.
  void set_total_cycles_executed(const HloComputation& computation,
                                 uint64 total_cycles_executed) {
    total_cycles_executed_[&computation] = total_cycles_executed;
  }

  // Returns a version of the execution profile suitable for performance
  // debugging; e.g. emits cycle counts, execution time at the nominal device
  // frequency, and the effective throughput given the provided cost_analysis
  // for the operations in a given computation. Returns an empty string if it
  // wasn't possible to generate a printable version. cost_analysis should be a
  // clean analysis that can be used to visit the computation.
  string ToString(const HloComputation& computation,
                  const DeviceDescription& device_description,
                  HloCostAnalysis* cost_analysis) const;

  // Returns the computations we have profiled.
  std::unordered_set<const HloComputation*> profiled_computations() const {
    return profiled_computations_;
  }

 private:
  // Contains a mapping from HLO to the number of cycles it took to execute it.
  std::unordered_map<const HloInstruction*, uint64> hlo_to_cycles_taken_;

  // If non-empty, contains the total number of cycles a computation took to
  // execute.
  std::unordered_map<const HloComputation*, uint64> total_cycles_executed_;

  // The computations we have profiled.
  std::unordered_set<const HloComputation*> profiled_computations_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EXECUTION_PROFILE_H_

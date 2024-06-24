/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_

#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"

namespace xla {
namespace memory_space_assignment {

// A wrapper class around runtime simulator.
class RuntimeSimulator {
 public:
  explicit RuntimeSimulator(CostAnalysis* cost_analysis)
      : cost_analysis_(cost_analysis) {}
  virtual ~RuntimeSimulator() = default;
  // This function is used to predict the effectiveness of the memory space
  // assignment solution. Specifically, it returns the estimated execution time
  // (in seconds) of the HLO module for the given memory space assignment (i.e.,
  // ```allocations```).
  float ComputeEstimatedElapsedTime(const HloLiveRange& hlo_live_range,
                                    const AllocationSequence& allocations);

 private:
  const CostAnalysis* cost_analysis_;
  CostAnalysis::Cache cost_analysis_cache_;
};
}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_

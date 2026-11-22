/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/hlo_scheduling/approximate_latency_estimator.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/hlo_scheduling/hlo_schedule_graph.h"
#include "xla/hlo/transforms/hlo_scheduling/latency_estimator.h"

namespace xla {

LatencyEstimator::TimeCost ApproximateLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  if (IsAsyncPair(from, target)) {
    return kHighLatency;
  }
  // Every other instruction we consider synchronous, which means the
  // latency between each of them is always one unit.
  return kLowLatency;
}

// Uses the approximate function for NodeCost based on a flag.
LatencyEstimator::TimeCost ApproximateLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (instr->IsLoopFusion()) {
    return kMediumCost;
  }
  if (instr->IsOutputFusion() || instr->opcode() == HloOpcode::kConvolution) {
    return kHighCost;
  }
  return kLowCost;
}

}  // namespace xla

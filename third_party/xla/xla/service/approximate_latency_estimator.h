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

#ifndef XLA_SERVICE_APPROXIMATE_LATENCY_ESTIMATOR_H_
#define XLA_SERVICE_APPROXIMATE_LATENCY_ESTIMATOR_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/latency_estimator.h"

namespace xla {

// Implementation of LatencyEstimator using an approximate cost model.
class ApproximateLatencyEstimator : public LatencyEstimator {
 public:
  explicit ApproximateLatencyEstimator(
      GetCanonicalAsyncOpFunc func = DefaultGetCanonicalAsyncOp)
      : LatencyEstimator(func) {}

  // Returns a latency estimation between two instructions.
  // Currently this is in abstract units. When the real/accurate cost model is
  // implemented this will be in cycles.
  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override;
  // Uses the approximate or cost model function for NodeCost based on a flag.
  TimeCost NodeCost(const HloInstruction* instr) const override;
  // ApproximateLatencyEstimator uses abstract units so this returns 1.
  int CyclesPerMicrosecond() const override { return 1; }

 public:
  static constexpr TimeCost kLowCost = 1.0;
  static constexpr TimeCost kMediumCost = 1000.0;
  static constexpr TimeCost kHighCost = 5000.0;

 protected:
  // These values are empirically derived to obtain an overlap of one output
  // fusion/convolution with 1 async op or 5 loop fusions with an async op.
  static constexpr TimeCost kLowLatency = 1.0;
  static constexpr TimeCost kHighLatency = 5000.0;
};

}  // namespace xla

#endif  // XLA_SERVICE_APPROXIMATE_LATENCY_ESTIMATOR_H_

/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_ESTIMATOR_H_
#define XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_ESTIMATOR_H_

#include <memory>
#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
// Implementation of AnalyticalLatencyEstimator using HloAnalysis and
// GPUPerformanceModel to estimate latencies for instructions.
class AnalyticalLatencyEstimator : public LatencyEstimator {
 public:
  AnalyticalLatencyEstimator(
      const SchedulerConfig& config,
      std::unique_ptr<LatencyEstimator> latency_estimator,
      const se::DeviceDescription& gpu_info,
      HloCostAnalysis::ShapeSizeFunction shape_size_function,
      HloComputation* computation);

  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override;
  TimeCost NodeCost(const HloInstruction* instr) const override;
  int CyclesPerMicrosecond() const override {
    return latency_estimator_->CyclesPerMicrosecond();
  }

  static constexpr TimeCost kLowCost = 1.0;
  static constexpr TimeCost kLowLatency = 1.0;

 private:
  const SchedulerConfig config_;
  const se::DeviceDescription& gpu_info_;
  std::optional<GpuHloCostAnalysis> cost_analysis_;
  std::unique_ptr<LatencyEstimator> latency_estimator_;
  HloCostAnalysis::ShapeSizeFunction shape_size_function_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_ESTIMATOR_H_

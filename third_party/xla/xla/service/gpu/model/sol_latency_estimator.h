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

#ifndef XLA_SERVICE_GPU_MODEL_SOL_LATENCY_ESTIMATOR_H_
#define XLA_SERVICE_GPU_MODEL_SOL_LATENCY_ESTIMATOR_H_

#include <memory>
#include <optional>

#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/collective_interpolator.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/matmul_interpolator.h"
#include "xla/service/gpu/model/sol_gpu_cost_model.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class SolLatencyEstimator : public LatencyEstimator {
 public:
  // Implementation of SolLatencyEstimator using HloAnalysis and
  // GPUPerformanceModel to estimate latencies for instructions.
  SolLatencyEstimator(const SchedulerConfig& config,
                      std::unique_ptr<LatencyEstimator> latency_estimator,
                      const se::DeviceDescription& gpu_info,
                      HloCostAnalysis::ShapeSizeFunction shape_size_function,
                      const HloComputation* computation);

  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override;
  TimeCost NodeCost(const HloInstruction* instr) const override;
  int CyclesPerMicrosecond() const override {
    return latency_estimator_->CyclesPerMicrosecond();
  }

  static absl::Duration ComputeCollectiveTime(
      const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const SolGPUCostModel::Config& sol_flags,
      const CollectiveInterpolator* collective_interpolator = nullptr);

  static absl::Duration ComputeCollectiveTime(
      const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const SolGPUCostModel::Config& sol_flags,
      const GpuHloCostAnalysis& cost_analysis,
      const CollectiveInterpolator* collective_interpolator = nullptr);

  static constexpr TimeCost kLowCost = 1.0;
  static constexpr TimeCost kLowLatency = 1.0;

 private:
  const SchedulerConfig config_;
  const se::DeviceDescription& gpu_info_;
  GpuPerformanceModelOwning gpu_performance_model_;
  std::optional<GpuHloCostAnalysis> cost_analysis_;
  std::unique_ptr<LatencyEstimator> latency_estimator_;
  HloCostAnalysis::ShapeSizeFunction shape_size_function_;
  const SolGPUCostModel::Config sol_flags_;
  std::unique_ptr<CollectiveInterpolator> collective_interpolator_;
  std::unique_ptr<MatmulInterpolator> matmul_interpolator_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SOL_LATENCY_ESTIMATOR_H_

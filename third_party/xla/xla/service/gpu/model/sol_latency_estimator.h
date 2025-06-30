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

#include "absl/status/statusor.h"
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

// Static analytical latency estimator for GPU. It uses empirical data and
// interpolation techniques to estimate the runtime of matrix multiplications
// and ICI (NVLINK) collectives, it also uses the GPU performance model to
// estimate runtime of fusion operations and DCN collectives.
//
// The rationale for this mix is that we do not have proper analytical solutions
// to estimate the runtime of GEMMs and we have not yet developed a good enough
// set of parameters to reconstruct a bandwidth derating curve for NVLINK.
// Interpolation/collection happens at the level of the HLO instruction,
// therefore in the case of algorithmic improvements at lower levels of
// abstractions performance tables need to be updated.
//
// Currently this estimator supports H100s and standard DP collectives, that is:
// All-Reduce, All-Gather, Reduce-Scatter.
class SolLatencyEstimator : public LatencyEstimator {
 public:
  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override;
  TimeCost NodeCost(const HloInstruction* instr) const override;
  int CyclesPerMicrosecond() const override {
    return latency_estimator_->CyclesPerMicrosecond();
  }

  // Computes the time it takes to execute the given collective instruction.
  // If `collective_interpolator` is provided, it will be used to estimate the
  // time it takes to execute the collective. Otherwise, just NCCL launch
  // overhead will be returned for ICI collectives.
  static absl::Duration ComputeCollectiveTime(
      const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const SolGPUCostModel::Config& sol_flags,
      const CollectiveInterpolator* collective_interpolator = nullptr);

  // Computes the time it takes to execute the given collective instruction.
  // If `collective_interpolator` is provided, it will be used to estimate the
  // time it takes to execute the collective. Otherwise, just NCCL launch
  // overhead will be returned for ICI collectives.
  // Relies on `cost_analysis` to get the collective size.
  static absl::Duration ComputeCollectiveTime(
      const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const SolGPUCostModel::Config& sol_flags,
      const GpuHloCostAnalysis& cost_analysis,
      const CollectiveInterpolator* collective_interpolator = nullptr);

  // Factory method to create a `SolLatencyEstimator`.
  static absl::StatusOr<std::unique_ptr<SolLatencyEstimator>> Create(
      const SchedulerConfig& config,
      std::unique_ptr<LatencyEstimator> latency_estimator,
      const se::DeviceDescription& gpu_info,
      HloCostAnalysis::ShapeSizeFunction shape_size_function,
      const HloComputation* computation,
      std::unique_ptr<GpuHloCostAnalysis> cost_analysis = nullptr);

  // Returns true if the module is supported by the SoL latency estimator.
  // In particular, it checks that the module contains only supported
  // collectives.
  static bool IsSupportedForModule(
      const HloModule& module, const se::DeviceDescription& gpu_device_info);

  static constexpr TimeCost kLowCost = 1.0;
  static constexpr TimeCost kLowLatency = 1.0;

 private:
  SolLatencyEstimator(
      const SchedulerConfig& config,
      std::unique_ptr<LatencyEstimator> latency_estimator,
      const se::DeviceDescription& gpu_info,
      std::unique_ptr<const GpuHloCostAnalysis> cost_analysis,
      HloCostAnalysis::ShapeSizeFunction shape_size_function,
      SolGPUCostModel::Config sol_flags,
      std::unique_ptr<CollectiveInterpolator> collective_interpolator,
      std::unique_ptr<MatmulInterpolator> matmul_interpolator);

  const SchedulerConfig config_;
  const se::DeviceDescription& gpu_info_;
  const GpuPerformanceModelOwning gpu_performance_model_;
  const std::unique_ptr<const GpuHloCostAnalysis> cost_analysis_;
  const std::unique_ptr<const LatencyEstimator> latency_estimator_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_function_;
  const SolGPUCostModel::Config sol_flags_;
  const std::unique_ptr<const CollectiveInterpolator> collective_interpolator_;
  const std::unique_ptr<const MatmulInterpolator> matmul_interpolator_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SOL_LATENCY_ESTIMATOR_H_

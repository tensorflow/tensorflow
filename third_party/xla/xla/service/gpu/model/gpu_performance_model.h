/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_H_

#include <memory>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class GpuPerformanceModel : public GpuPerformanceModelBase {
 public:
  // Lifetime to all references to this constructor must live at least as long
  GpuPerformanceModel(const se::DeviceDescription& device_info,
                      HloFusionAnalysisCache& fusion_analysis_cache,
                      GpuPerformanceModelCache& gpu_performance_model_cache);

  EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis);

  // TODO(shyshkov): Unify interface with EstimateRunTimeForInstruction.
  absl::Duration EstimateRunTimeForFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      bool producer_writes_side_output = false);

  RunTimes EstimateRunTimes(
      const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
      absl::Span<const HloInstruction* const> fused_consumers = {});

  RunTimes EstimateRunTimesForMultiOutputFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const GpuHloCostAnalysis* cost_analysis);

  // Writes estimated execution time to FusionBackendConfig.reification_cost.
  void RecordEstimatedRunTime(HloInstruction* instruction,
                              const GpuHloCostAnalysis* cost_analysis);

 private:
  EstimateRunTimeData EstimateRunTimeForInstructionImpl(
      const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis);

  absl::Duration EstimateRunTimeForFusionImpl(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      bool producer_writes_side_output);

  const se::DeviceDescription& device_info_;
  HloFusionAnalysisCache& fusion_analysis_cache_;
  // TODO(sohaibiftikhar) Make this an owning member of this class. Currently
  // this is not possible because the cache is used directly by
  // xla::gpu::PriorityFusionQueue
  GpuPerformanceModelCache& gpu_performance_model_cache_;
};

// An owning wrapper around GpuPerformanceModel that also owns the caches.
// This keeps dependencies for the analysis caches out of the files that only
// need to perform the analysis without owning or knowing about the caches.
// If access to the analysis caches are required by the caller then the non
// owning model should be used.
class GpuPerformanceModelOwning {
 public:
  explicit GpuPerformanceModelOwning(const se::DeviceDescription& device_info);

  // Wrapper over GpuPerformanceModel::RecordEstimatedRunTime
  void RecordEstimatedRunTime(HloInstruction* instruction,
                              const GpuHloCostAnalysis* cost_analysis) const;

  // Wrapper over GpuPerformanceModel::EstimateRunTimeForInstruction.
  EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr,
      const GpuHloCostAnalysis* cost_analysis) const;

  // Wrapper over GpuPerformanceModel::EstimateRunTimesForMultiOutputFusion.
  GpuPerformanceModel::RunTimes EstimateRunTimesForMultiOutputFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const GpuHloCostAnalysis* cost_analysis) const;

 private:
  HloFusionAnalysisCache fusion_analysis_cache_;
  GpuPerformanceModelCache gpu_performance_model_cache_{};
  // Unique pointer to allow const access to the model since caches will be
  // mutated after estimation.
  std::unique_ptr<GpuPerformanceModel> gpu_performance_model_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_H_

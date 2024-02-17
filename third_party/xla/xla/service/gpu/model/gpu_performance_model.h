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


#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"

namespace xla {
namespace gpu {

class GpuPerformanceModel : public GpuPerformanceModelBase {
 public:
  static EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis,
      const GpuPerformanceModelOptions& config);

  static EstimateRunTimeData EstimateRunTimeForInstructionCached(
      const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis,
      const GpuPerformanceModelOptions& config);

  // TODO(shyshkov): Unify interface with EstimateRunTimeForInstruction.
  static absl::Duration EstimateRunTimeForFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      const GpuPerformanceModelOptions& config);

  static absl::Duration EstimateRunTimeForFusionCached(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      const GpuPerformanceModelOptions& config);

  static absl::Duration EstimateUnfusedExecTime(
      const HloInstruction* producer,
      const EstimateRunTimeData& producer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      const GpuPerformanceModelOptions& config,
      absl::Span<const HloInstruction* const> fused_consumers);

  static absl::Duration EstimateFusedExecTime(
      const HloInstruction* producer,
      const EstimateRunTimeData& producer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      const GpuPerformanceModelOptions& config,
      absl::Span<const HloInstruction* const> fused_consumers,
      bool multi_output);

  static RunTimes EstimateRunTimes(
      const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
      const GpuPerformanceModelOptions& config,
      absl::Span<const HloInstruction* const> fused_consumers = {},
      bool multi_output = false);

  static RunTimes EstimateRunTimesForPriorityFusion(
      const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
      const GpuPerformanceModelOptions& config,
      absl::Span<const HloInstruction* const> fused_consumers = {},
      bool multi_output = false);

  // Writes estimated execution time to FusionBackendConfig.reification_cost.
  static void RecordEstimatedRunTime(HloInstruction* instruction,
                                     const GpuHloCostAnalysis* cost_analysis,
                                     const GpuPerformanceModelOptions& config);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_PERFORMANCE_MODEL_H_

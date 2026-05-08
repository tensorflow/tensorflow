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

#ifndef XLA_SERVICE_GPU_MODEL_COMBINED_GPU_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_COMBINED_GPU_PERFORMANCE_MODEL_H_

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// A cost model wrapper that uses GpuPerformanceModelWithIndexingAnalysis (for
// Triton fusion HloInstructions) and GpuPerformanceModel (for other
// HloInstructions).
//
// Caches estimated run times for both individual instructions and fusions.
//
// This class is thread-safe.
class CombinedGpuPerformanceModel : public GpuPerformanceModelBase {
 public:
  CombinedGpuPerformanceModel(
      const se::DeviceDescription& device_info ABSL_ATTRIBUTE_LIFETIME_BOUND,
      HloFusionAnalysisCache& fusion_analysis_cache
          ABSL_ATTRIBUTE_LIFETIME_BOUND,
      mlir::MLIRContext& mlir_context ABSL_ATTRIBUTE_LIFETIME_BOUND,
      HloCostAnalysis::ShapeSizeFunction shape_size);

  // Returns runtime data analysis results of a single instruction.
  //
  // Uses one of the wrapped models for analysis and caches the results.
  absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForInstruction(
      const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Returns estimated runtime (fused and unfused) of producer and set of
  // consumers.
  //
  // Caches the results using the set of HloInstruction pointers as the key.
  absl::StatusOr<RunTimes> EstimateRunTimes(
      const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
      absl::Span<const HloInstruction* const> fused_consumers = {})
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Returns estimated runtime (fused and unfused) of producer and consumer,
  // assuming the produced fusion is a multi-output fusion.
  //
  // Caches the results using the set of HloInstruction pointers as the key.
  absl::StatusOr<RunTimes> EstimateRunTimesForMultiOutput(
      const HloInstruction* producer, const HloInstruction* consumer,
      const GpuHloCostAnalysis* cost_analysis)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // Returns the best tiling for a fusion.
  //
  // DOES NOT cache the result.
  absl::StatusOr<TiledRunTimeDataOrError> TryFindBestTilingForFusion(
      const HloFusionAdaptor& fusion_adaptor);

  // Invalidates all cache entries related to given instruction.
  //
  // Note: does NOT invalidate the HloFusionAnalysisCache, which is not owned by
  // this class.
  void Invalidate(const HloInstruction& instruction)
      ABSL_LOCKS_EXCLUDED(cache_mutex_);

  // TODO: b/493907020 Remove this when no longer needed in PriorityFusionQueue.
  // UNSAFE: bypasses cache mutex
  GpuPerformanceModelCache& GetCache() { return cache_; }

 private:
  absl::Duration EstimateRunTimeForFusion(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      bool producer_writes_side_output);

  absl::Duration EstimateRunTimeForFusionUncached(
      const HloInstruction* producer, const HloInstruction* consumer,
      const EstimateRunTimeData& producer_runtime,
      const EstimateRunTimeData& consumer_runtime,
      const GpuHloCostAnalysis* cost_analysis,
      bool producer_writes_side_output);

  const se::DeviceDescription& device_info_;
  HloFusionAnalysisCache& fusion_analysis_cache_;
  mlir::MLIRContext& mlir_context_;

  GpuPerformanceModelWithIndexingAnalysis indexing_model_;
  GpuPerformanceModel model_;

  absl::Mutex cache_mutex_;
  GpuPerformanceModelCache cache_ ABSL_GUARDED_BY(cache_mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_COMBINED_GPU_PERFORMANCE_MODEL_H_

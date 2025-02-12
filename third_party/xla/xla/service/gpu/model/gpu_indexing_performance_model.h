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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Contains informations about block level parameters and run time of a fusion.
struct TiledRunTimeData {
  EstimateRunTimeData runtime_data;
  BlockLevelParameters block_level_parameters;
};

using TiledRunTimeDataOrError = std::variant<TiledRunTimeData, FusionDecision>;

// Implementation of Cost Model that uses indexing analysis to estimate amount
// of compute and memory access time.
class GpuPerformanceModelWithIndexingAnalysis : public GpuPerformanceModelBase {
 public:
  explicit GpuPerformanceModelWithIndexingAnalysis(
      const se::DeviceDescription* device_info,
      HloFusionAnalysisCache* fusion_analysis_cache,
      HloCostAnalysis::ShapeSizeFunction shape_size,
      mlir::MLIRContext* mlir_context)
      : hlo_op_profile_(&HloOpProfiles::Singleton().GetProfile(*device_info)),
        device_info_(device_info),
        fusion_analysis_cache_(fusion_analysis_cache),
        shape_size_(shape_size),
        cost_analysis_(
            GpuHloCostAnalysis::Options{shape_size_,
                                        /*per_second_rates=*/{},
                                        /*min_latencies_seconds=*/{},
                                        /*count_multiple_input_accesses=*/true},
            *device_info_),
        mlir_context_(mlir_context) {}

  // Returns the launch dimensions for the given tiled HLO computation.
  static LaunchDimensions GetLaunchDimensionsForTiledFusion(
      const TiledHloComputation& tiled_hlo_computation,
      const se::DeviceDescription& device_info);

  EstimateRunTimeData EstimateRunTimeForFusion(
      const HloFusionAnalysis& fusion_analysis, bool is_coalesced = true);

  EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* producer);

  EstimateRunTimeData EstimateRunTimeForProducerConsumer(
      const HloInstruction* producer, const HloInstruction* consumer);

  RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<const HloInstruction* const> fused_consumers = {});

  absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForTiledHloComputation(
      const HloFusionAdaptor& fusion_adaptor,
      const TiledHloComputation& tiled_hlo_computation,
      const LaunchDimensions& launch_dimensions);

  // Estimate the run time of the fusion with the given launch dimensions and
  // output tile sizes.
  //
  // The model uses SymbolicTileAnalysis to build a TiledHloComputation with the
  // given tile sizes. This way it can better estimate the amount of memory
  // access and computation.
  absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForTiledFusion(
      const HloFusionAdaptor& fusion_adaptor,
      const LaunchDimensions& launch_dimensions,
      absl::Span<const int64_t> output_tile_sizes);

  // Estimate the run time of producer and consumer fused together, assuming
  // that they will be emitted with Triton.
  // If consumer is nullptr, estimate run time of the producer alone.
  absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForTriton(
      const HloInstruction* producer, const HloInstruction* consumer = nullptr);

  // Estimates the best tile sizes for the given fusion. Iterates over all the
  // good tile sizes provided by SymbolicTileAnalysis, estimates the run time
  // for each of them.
  //
  // Returns status if there is an error that we can't recover from.
  // Returns FusionDecision if the fusion can't be tiled or there are no valid
  // block level parameters.
  // Otherwise returns block level parameters that give the best execution time.
  absl::StatusOr<TiledRunTimeDataOrError> TryFindBestTilingForFusion(
      const HloFusionAdaptor& fusion_adaptor);

  // Returns an estimate how many FLOPs will be used to produce one element of
  // the output.
  int64_t FlopsPerElement(const HloInstruction* instr);

 private:
  int64_t GetShapeSizeRecursive(const Shape& shape) const;

  const HloOpProfiles::HloOpProfile* hlo_op_profile_;
  const se::DeviceDescription* device_info_;
  HloFusionAnalysisCache* fusion_analysis_cache_;
  HloCostAnalysis::ShapeSizeFunction shape_size_;
  GpuHloCostAnalysis cost_analysis_;
  mlir::MLIRContext* mlir_context_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_

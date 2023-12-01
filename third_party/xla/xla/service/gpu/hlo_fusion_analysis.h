/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_
#define XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_

#include <optional>
#include <utility>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernel_mapping_scheme.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class HloFusionAnalysis {
 public:
  // The type of emitted fusion.
  enum class EmitterFusionKind {
    kLoop,
    kCustomFusion,
    kTriton,
    kReduction,
    kTranspose,
    kInputSlices,
    kScatter,
  };

  static StatusOr<HloFusionAnalysis> Create(
      FusionBackendConfig backend_config,
      std::vector<const HloInstruction*> hlo_roots,
      FusionBoundaryFn boundary_fn, const se::DeviceDescription* device_info);
  static StatusOr<HloFusionAnalysis> Create(
      const HloFusionInstruction* fusion,
      const se::DeviceDescription* device_info);

  const std::vector<const HloInstruction*>& fusion_roots() const {
    return fusion_roots_;
  }
  const FusionBoundaryFn& fusion_boundary() const {
    return fusion_boundary_fn_;
  }

  // Determines the fusion type for the emitter.
  EmitterFusionKind GetEmitterFusionKind() const;

  // Determines the launch dimensions for the fusion. The fusion kind must not
  // be `kTriton`.
  StatusOr<LaunchDimensions> GetLaunchDimensions() const;

  // Calculates the reduction information. Returns `nullptr` if the fusion is
  // not a reduction.
  const ReductionCodegenInfo* GetReductionCodegenInfo() const {
    return reduction_codegen_info_.has_value() ? &*reduction_codegen_info_
                                               : nullptr;
  }

  // Calculates the transpose tiling information. Returns `nullptr` if the
  // fusion is not a transpose.
  const TilingScheme* GetTransposeTilingScheme() const {
    return transpose_tiling_scheme_.has_value() ? &*transpose_tiling_scheme_
                                                : nullptr;
  }

  // Calculates the loop fusion config. Returns `nullptr` if the fusion is not a
  // loop.
  const LaunchDimensionsConfig* GetLoopFusionConfig() const {
    return loop_fusion_config_.has_value() ? &*loop_fusion_config_ : nullptr;
  }

  // Returns the hero reduction of the computation.
  const HloInstruction* FindHeroReduction() const;

 private:
  // Precomputed information about inputs (arguments) and outputs (roots) of the
  // fusion.
  struct InputOutputInfo {
    bool has_4_bit_input;
    bool has_4_bit_output;
    int smallest_input_dtype_bits;
  };

  HloFusionAnalysis(FusionBackendConfig fusion_backend_config,
                    std::vector<const HloInstruction*> fusion_roots,
                    FusionBoundaryFn fusion_boundary_fn,
                    std::vector<const HloInstruction*> fusion_heroes,
                    const se::DeviceDescription* device_info,
                    std::optional<TransposeDescription> tiled_transpose,
                    InputOutputInfo input_output_info);

  const Shape& GetElementShape() const;
  int64_t MaxBeneficialColumnReductionUnrollBasedOnBlockSize() const;
  std::vector<std::vector<const HloInstruction*>> GroupDisjointReductions()
      const;
  bool IsUnrollingColumnReductionBeneficial(const Shape& input_shape,
                                            int64_t num_kept_minor,
                                            bool reduction_is_race_free) const;
  bool CanVectorizeReduction(const ReductionDimensions& reduction_dimensions,
                             int num_threads_x, Vector3 reduction_tiling,
                             const Shape& input_shape,
                             bool reduction_is_race_free) const;
  int CalculateVirtualThreadScalingFactorForReduction(
      const ReductionDimensions& reduction_dimensions) const;
  std::optional<ReductionCodegenInfo> ComputeReductionCodegenInfo(
      const HloInstruction* hero_reduction) const;
  std::optional<LaunchDimensionsConfig> ComputeLoopFusionConfig() const;
  bool HasConsistentTransposeHeros() const;

  FusionBackendConfig fusion_backend_config_;
  std::vector<const HloInstruction*> fusion_roots_;
  FusionBoundaryFn fusion_boundary_fn_;
  std::vector<const HloInstruction*> fusion_heroes_;
  const se::DeviceDescription* device_info_;
  std::optional<TransposeDescription> tiled_transpose_;
  InputOutputInfo input_output_info_;

  std::optional<ReductionCodegenInfo> reduction_codegen_info_;
  std::optional<TilingScheme> transpose_tiling_scheme_;
  std::optional<LaunchDimensionsConfig> loop_fusion_config_;
};

// Creates a HloFusionAnalysis that analyzes a hypothetical fusion of producer
// into consumer.
std::optional<HloFusionAnalysis> AnalyzeProducerConsumerFusion(
    const HloInstruction& producer, const HloInstruction& consumer,
    const se::DeviceDescription& device_info);
// Creates a HloFusionAnalysis that analyzes just consumer as a standalone
// fusion.
std::optional<HloFusionAnalysis> AnalyzeFusion(
    const HloInstruction& consumer, const se::DeviceDescription& device_info);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_

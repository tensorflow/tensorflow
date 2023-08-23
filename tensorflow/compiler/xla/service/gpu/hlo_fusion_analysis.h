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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_

#include <optional>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_utils.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class HloFusionAnalysis {
 public:
  // The type of emitted fusion.
  enum class EmitterFusionKind {
    kLoop,
    kTriton,
    kReduction,
    kTranspose,
    kInputSlices,
    kScatter,
  };

  static StatusOr<HloFusionAnalysis> Create(
      const HloFusionInstruction* fusion, const GpuDeviceInfo* device_info,
      se::CudaComputeCapability compute_capability);

  const HloComputation* fused_computation() const { return fused_computation_; }
  const std::vector<HloInstruction*>& fusion_roots() const {
    return fusion_roots_;
  }

  // Determines the fusion type for the emitter.
  EmitterFusionKind GetEmitterFusionKind() const;

  // Determines the launch dimensions for the fusion. The fusion kind must not
  // be `kTriton`.
  StatusOr<LaunchDimensions> GetLaunchDimensions();

  // Calculates the reduction information. Returns `nullptr` if the fusion is
  // not a reduction.
  const ReductionCodegenInfo* GetReductionCodegenInfo();

  // Calculates the transpose tiling information. Returns `nullptr` if the
  // fusion is not a transpose.
  const TilingScheme* GetTransposeTilingScheme();

  // Calculates the loop fusion config. Returns `nullptr` if the fusion is not a
  // loop.
  const LaunchDimensionsConfig* GetLoopFusionConfig();

 private:
  HloFusionAnalysis(const HloFusionInstruction* fusion,
                    FusionBackendConfig fusion_backend_config,
                    std::vector<HloInstruction*> fusion_roots,
                    std::vector<const HloInstruction*> fusion_heroes,
                    const GpuDeviceInfo* device_info,
                    se::CudaComputeCapability compute_capability,
                    std::optional<TransposeDescription> tiled_transpose)
      : fusion_(fusion),
        fusion_backend_config_(std::move(fusion_backend_config)),
        fused_computation_(fusion->fused_instructions_computation()),
        fusion_roots_(std::move(fusion_roots)),
        fusion_heroes_(std::move(fusion_heroes)),
        device_info_(device_info),
        compute_capability_(compute_capability),
        tiled_transpose_(tiled_transpose) {}

  const Shape& GetElementShape() const;
  int SmallestInputDtypeBits() const;
  int64_t MaxBeneficialColumnReductionUnrollBasedOnBlockSize() const;
  std::vector<std::vector<HloInstruction*>> GroupDisjointReductions() const;
  bool IsUnrollingColumnReductionBeneficial(const Shape& input_shape,
                                            int64_t num_kept_minor,
                                            bool reduction_is_race_free) const;
  bool CanVectorizeReduction(const ReductionDimensions& reduction_dimensions,
                             int num_threads_x, Vector3 reduction_tiling,
                             const Shape& input_shape,
                             bool reduction_is_race_free) const;
  int CalculateVirtualThreadScalingFactorForReduction(
      const ReductionDimensions& reduction_dimensions) const;
  ReductionCodegenInfo ComputeReductionCodegenInfo(
      const HloInstruction* hero_reduction) const;
  bool HasConsistentTransposeHeros() const;

  const HloFusionInstruction* fusion_;
  FusionBackendConfig fusion_backend_config_;
  const HloComputation* fused_computation_;
  std::vector<HloInstruction*> fusion_roots_;
  std::vector<const HloInstruction*> fusion_heroes_;
  const GpuDeviceInfo* device_info_;
  se::CudaComputeCapability compute_capability_;
  std::optional<TransposeDescription> tiled_transpose_;

  std::optional<ReductionCodegenInfo> reduction_codegen_info_;
  std::optional<TilingScheme> transpose_tiling_scheme_;
  std::optional<LaunchDimensionsConfig> loop_fusion_config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_

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

#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
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

  HloFusionAnalysis(const HloFusionInstruction* fusion,
                    const GpuDeviceInfo* device_info,
                    se::CudaComputeCapability compute_capability)
      : fusion_(fusion),
        fused_computation_(fusion->fused_instructions_computation()),
        fusion_roots_(GetFusionRoots(fusion->fused_instructions_computation())),
        device_info_(device_info),
        compute_capability_(compute_capability) {}

  // Determine the fusion type for the emitter.
  StatusOr<EmitterFusionKind> GetEmitterFusionKind() const;

  // Determine the launch dimensions for the fusion.
  StatusOr<LaunchDimensions> GetLaunchDimensions() const;

  // Calculate reduction information (kind: kReduction).
  StatusOr<ReductionCodegenInfo> GetReductionCodegenInfo() const;

  // Calculate transpose tiling information (kind: kTranspose).
  StatusOr<TilingScheme> GetTransposeTilingScheme() const;

 private:
  LaunchDimensionsConfig GetLoopFusionConfig() const;
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
  StatusOr<ReductionCodegenInfo> ComputeReductionCodegenInfo(
      HloInstruction* first_reduce) const;

  const HloFusionInstruction* fusion_;
  const HloComputation* fused_computation_;
  std::vector<HloInstruction*> fusion_roots_;
  const GpuDeviceInfo* device_info_;
  se::CudaComputeCapability compute_capability_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HLO_FUSION_ANALYSIS_H_

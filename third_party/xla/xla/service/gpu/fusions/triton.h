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
#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

class TritonFusion : public FusionInterface {
 public:
  struct LaunchConfig {
    LaunchDimensions launch_dimensions;
    BlockLevelParameters block_level_parameters;
  };

  explicit TritonFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis) {}

  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

  // Returns the launch config for Triton fusions that have a block level fusion
  // config.
  // Not supported for MatMul fusions yet.
  std::optional<LaunchConfig> launch_config() const;

 private:
  const HloFusionAnalysis& analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_H_

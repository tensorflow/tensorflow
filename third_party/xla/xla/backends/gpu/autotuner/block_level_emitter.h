/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_BLOCK_LEVEL_EMITTER_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_BLOCK_LEVEL_EMITTER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Codegen backend for the Triton block-level fusion emitter.
//
// This backend enables autotuning of Triton-based fusion computations at the
// block level. It generates tiling configurations, applies them to
// instructions, and prepares them for compilation using the Triton emitter.
class BlockLevelEmitterBackend : public GpuCodegenBackend {
 public:
  explicit BlockLevelEmitterBackend(
      const DebugOptions* absl_nonnull debug_options,
      Compiler* absl_nonnull compiler,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const Compiler::GpuTargetConfig* target_config,
      bool use_default_config = false)
      : GpuCodegenBackend(autotuner::Backend::BLOCK_LEVEL_EMITTER,
                          debug_options, compiler, target_config),
        use_default_config_(use_default_config),
        shape_size_fn_(std::move(shape_size_fn)) {}

  // Returns all supported block-level tiling configurations for the given
  // instruction.
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  // Returns a default block-level configuration for the instruction.
  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  // Applies a given block-level fusion configuration to the instruction.
  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

  // Determines whether the given HLO instruction is supported by this backend.
  bool IsSupported(const HloInstruction& instr) override;

  // We don't want to use the Triton emitter as a reference because it can
  // produce wrong results.
  bool CanProduceWrongResults() const override { return true; }

 private:
  absl::StatusOr<BlockLevelFusionConfig> GetCostModelConfig(
      const HloInstruction& instr) const;
  // If true, the backend will return a single default configuration in
  // GetSupportedConfigs instead of generating all supported configurations.
  // This is useful to autotune between different backends without increasing
  // compile time by too much. It will use the default config, likely already
  // assigned by the cost model.
  bool use_default_config_;
  // A function which returns the size in bytes of the top-level buffer of a
  // shape.
  HloCostAnalysis::ShapeSizeFunction shape_size_fn_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_BLOCK_LEVEL_EMITTER_H_

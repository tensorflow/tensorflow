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

#include "xla/backends/cpu/autotuner/block_level_emitter_backend.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/custom_fusion_configs.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"

namespace xla::cpu {

using ::xla::xtile::BlockLevelFusionConfig;

namespace {

std::unique_ptr<xla::BackendConfig> Pack(
    const BlockLevelFusionConfig& block_level_config) {
  auto config = std::make_unique<xla::BackendConfig>();
  *config->mutable_block_level() = block_level_config;
  return config;
}

absl::StatusOr<std::optional<BlockLevelFusionConfig>> GetPreExistingConfig(
    const HloInstruction& instr) {
  if (!instr.has_backend_config()) {
    return std::nullopt;
  }
  ABSL_ASSIGN_OR_RETURN(xla::cpu::BackendConfig cpu_backend_config,
                   instr.backend_config<xla::cpu::BackendConfig>());
  if (cpu_backend_config.has_fusion_config() &&
      cpu_backend_config.fusion_config().has_block_level_fusion_config()) {
    return cpu_backend_config.fusion_config().block_level_fusion_config();
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<std::unique_ptr<CodegenBackend>>
BlockLevelEmitterBackend::Create(Compiler* compiler) {
  return std::make_unique<BlockLevelEmitterBackend>(compiler);
}

bool BlockLevelEmitterBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  return true;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::BackendConfig>>>
BlockLevelEmitterBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<xla::BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }

  ABSL_ASSIGN_OR_RETURN(std::optional<BlockLevelFusionConfig> pre_existing_config,
                   GetPreExistingConfig(instr));
  if (pre_existing_config.has_value()) {
    configs.push_back(Pack(pre_existing_config.value()));
    return configs;
  }

  // Generate some configurations to try.
  // TODO(538568507): Add real heuristics for CPU blocks.
  BlockLevelFusionConfig default_config;
  default_config.set_num_warps(1);  // Placeholder for CPU tile.
  configs.push_back(Pack(default_config));

  return configs;
}

absl::StatusOr<std::unique_ptr<xla::BackendConfig>>
BlockLevelEmitterBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        absl::StrCat("BlockLevelEmitterBackend: unsupported instruction: ",
                     instr.ToString()));
  }
  ABSL_ASSIGN_OR_RETURN(std::optional<BlockLevelFusionConfig> pre_existing_config,
                   GetPreExistingConfig(instr));
  if (pre_existing_config.has_value()) {
    return Pack(pre_existing_config.value());
  }

  BlockLevelFusionConfig default_config;
  default_config.set_num_warps(1);  // Placeholder for CPU tile.
  return Pack(default_config);
}

absl::Status BlockLevelEmitterBackend::ApplyConfig(
    HloInstruction& instr, const xla::BackendConfig& config) {
  if (!config.has_block_level()) {
    return absl::InvalidArgumentError("Expected BlockLevelFusionConfig.");
  }
  BlockLevelFusionConfig block_level_fusion_config = config.block_level();

  xla::cpu::BackendConfig backend_config;
  if (instr.has_backend_config()) {
    ABSL_ASSIGN_OR_RETURN(backend_config,
                     instr.backend_config<xla::cpu::BackendConfig>());
  }

  xla::cpu::FusionBackendConfig* fusion_config =
      backend_config.mutable_fusion_config();
  fusion_config->set_kind(kXtileFusionKind);
  *fusion_config->mutable_block_level_fusion_config() =
      block_level_fusion_config;

  // In XLA:CPU, BlockLevelEmitter takes normal fusions (not custom fusions like
  // in XLA:GPU), so we don't need to set the fusion kind to kCustom.
  ABSL_RETURN_IF_ERROR(instr.set_backend_config(backend_config));

  return absl::OkStatus();
}

}  // namespace xla::cpu

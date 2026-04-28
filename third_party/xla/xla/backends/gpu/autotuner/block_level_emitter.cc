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

#include "xla/backends/gpu/autotuner/block_level_emitter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/codegen/triton/tma_utils.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {


void ExtendConfigsWithTma(
    std::vector<std::unique_ptr<BackendConfig>>& configs) {
  int64_t original_size = configs.size();
  for (int64_t i = 0; i < original_size; ++i) {
    BlockLevelFusionConfig original_config;
    if (!configs[i]->UnpackTo(&original_config)) {
      // This should not happen based on how configs are created.
      LOG(ERROR) << "Failed to unpack BlockLevelFusionConfig";
      continue;
    }
    if (IsTmaRecommended(original_config)) {
      BlockLevelFusionConfig new_config = original_config;
      new_config.set_is_tma_allowed(true);
      auto any = std::make_unique<google::protobuf::Any>();
      any->PackFrom(new_config);
      configs.push_back(std::move(any));
    }
  }
}
}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
BlockLevelEmitterBackend::GetSupportedConfigs(const HloInstruction& instr) {
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      GetDefaultConfig(instr);
  if (!config.ok()) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(std::move(config.value()));

  if (!instr.has_backend_config() &&
      stream_executor::gpu::IsTmaAvailableForDevice(
          target_config().device_description)) {
    ExtendConfigsWithTma(configs);
  }

  return configs;
}

absl::StatusOr<BlockLevelFusionConfig>
BlockLevelEmitterBackend::GetCostModelConfig(
    const HloInstruction& instr) const {
  auto device_info = target_config().device_description;
  HloFusionAnalysisCache fusion_analysis_cache(device_info);
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);
  GpuPerformanceModelWithIndexingAnalysis indexing_performance_model(
      &device_info, &fusion_analysis_cache, shape_size_fn_, &mlir_context);

  auto fusion_adaptor =
      HloFusionAdaptor::ForInstruction(Cast<HloFusionInstruction>(&instr));

  TF_ASSIGN_OR_RETURN(
      TiledRunTimeDataOrError tiled_runtime_data_or_error,
      indexing_performance_model.TryFindBestTilingForFusion(*fusion_adaptor));

  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&tiled_runtime_data_or_error)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Can't rewrite fusion ", instr.ToString(),
        " because tiling search failed: ", fusion_decision->Explain()));
  }
  TiledRunTimeData tiled_runtime_data =
      std::get<TiledRunTimeData>(std::move(tiled_runtime_data_or_error));

  return tiled_runtime_data.block_level_parameters.ToBlockLevelFusionConfig();
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
BlockLevelEmitterBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        absl::StrCat("BlockLevelEmitterBackend: unsupported instruction: ",
                     instr.ToString()));
  }
  // Attempt to extract an existing BlockLevelFusionConfig from the instruction.
  // Object nesting structure:
  // HloInstruction
  // └── GpuBackendConfig
  //     └── FusionBackendConfig
  //         └── BlockLevelFusionConfig
  if (instr.has_backend_config()) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                        instr.backend_config<GpuBackendConfig>());
    if (gpu_backend_config.has_fusion_backend_config()) {
      const FusionBackendConfig& fusion_backend_config =
          gpu_backend_config.fusion_backend_config();
      // If a BlockLevelFusionConfig is already present, return it directly.
      if (fusion_backend_config.has_block_level_fusion_config()) {
        auto any = std::make_unique<google::protobuf::Any>();
        any->PackFrom(fusion_backend_config.block_level_fusion_config());
        return any;
      }
    }
  }

  // No explicit config found - create one from the cost model if possible.
  TF_ASSIGN_OR_RETURN(BlockLevelFusionConfig config, GetCostModelConfig(instr));
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(config);
  return any;
}

absl::Status BlockLevelEmitterBackend::ApplyConfig(
    HloInstruction& instr, const BackendConfig& config) {
  // Object nesting structure:
  // HloInstruction
  // └── GpuBackendConfig
  //     └── FusionBackendConfig
  //         └── BlockLevelFusionConfig
  // Ensure the provided config is of type BlockLevelFusionConfig.
  BlockLevelFusionConfig block_level_fusion_config;
  if (!config.UnpackTo(&block_level_fusion_config)) {
    return absl::InvalidArgumentError(
        "Invalid backend config type for BlockLevelFusionConfig.");
  }
  // Extract the current GPU backend config from the instruction.
  // This contains the nested FusionBackendConfig we want to modify.
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_backend_config.mutable_fusion_backend_config();
  backend_config.set_kind(kTritonFusionKind);
  // Overwrite the block-level fusion config with the new one provided.
  *backend_config.mutable_block_level_fusion_config() =
      block_level_fusion_config;
  // Re-attach the modified GPU config back to the instruction.
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_backend_config)));
  instr.set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return absl::OkStatus();
}

bool BlockLevelEmitterBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  const HloComputation* fusion_computation =
      Cast<HloFusionInstruction>(&instr)->fused_instructions_computation();
  return IsTritonSupportedComputation(
             *fusion_computation,
             target_config().device_description.gpu_compute_capability())
      .CanFuse();
}

}  // namespace gpu
}  // namespace xla

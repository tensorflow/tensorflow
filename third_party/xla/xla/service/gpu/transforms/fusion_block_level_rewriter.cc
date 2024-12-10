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

#include "xla/service/gpu/transforms/fusion_block_level_rewriter.h"

#include <string>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/triton/triton_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::MLIRContext;

absl::StatusOr<bool> ProcessFusionInstruction(
    HloFusionInstruction* fusion_instruction,
    const se::DeviceDescription& device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size, MLIRContext* ctx) {
  const HloComputation* fusion_computation =
      fusion_instruction->fused_instructions_computation();
  if (CodegenDecision can_codegen = IsTritonSupportedComputation(
          *fusion_computation, device_info.gpu_compute_capability());
      !can_codegen) {
    VLOG(2) << "Can't rewrite fusion " << fusion_instruction->ToString()
            << " because one or more instructions is not supported by Triton: "
            << can_codegen.Explain();
    return false;
  }

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      fusion_instruction->backend_config<GpuBackendConfig>());

  if (backend_config.has_fusion_backend_config() &&
      backend_config.fusion_backend_config().has_block_level_fusion_config()) {
    // Fusion is already block-level! Skip.
    return false;
  }

  HloFusionAnalysisCache fusion_analysis_cache(device_info);
  GpuPerformanceModelWithIndexingAnalysis indexing_performance_model(
      &device_info, &fusion_analysis_cache, shape_size, ctx);

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(
      Cast<HloFusionInstruction>(fusion_instruction));

  TF_ASSIGN_OR_RETURN(
      TiledRunTimeDataOrError tiled_runtime_data_or_error,
      indexing_performance_model.TryFindBestTilingForFusion(*fusion_adaptor));

  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&tiled_runtime_data_or_error)) {
    // Can't rewrite this fusion because we can't tile it, skip!
    VLOG(2) << "Can't rewrite fusion " << fusion_instruction->ToString()
            << " because tiling search failed. (The most likely cause for "
            << "is that SymbolicTileAnalysis failed.)";
    return false;
  }

  TiledRunTimeData tiled_runtime_data =
      std::get<TiledRunTimeData>(std::move(tiled_runtime_data_or_error));
  VLOG(1)
      << "Found parameters "
      << absl::StrCat(
             "sizes=[",
             absl::StrJoin(
                 tiled_runtime_data.block_level_parameters.output_tile_sizes,
                 ", "),
             "], num_warps=",
             tiled_runtime_data.block_level_parameters.num_warps)
      << " for fusion computation " << fusion_computation->ToString();

  *backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      tiled_runtime_data.block_level_parameters.ToBlockLevelFusionConfig();
  backend_config.mutable_fusion_backend_config()->set_kind(
      std::string(kTritonFusionKind));
  TF_RETURN_IF_ERROR(fusion_instruction->set_backend_config(backend_config));
  fusion_instruction->set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return true;
}

}  // anonymous namespace

absl::StatusOr<bool> FusionBlockLevelRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(EnsureTritonSupportsComputeCapability(
      device_info_.gpu_compute_capability()));

  MLIRContext ctx;
  bool has_changed = false;

  for (HloComputation* computation :
       module->MakeComputationSorted(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }

    HloFusionInstruction* fusion_instruction =
        ::xla::Cast<HloFusionInstruction>(computation->FusionInstruction());

    TF_ASSIGN_OR_RETURN(bool should_try_rewrite,
                        should_try_rewrite_if_(fusion_instruction));
    if (!should_try_rewrite) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        bool changed, ProcessFusionInstruction(fusion_instruction, device_info_,
                                               shape_size_, &ctx));

    has_changed |= changed;
  }

  return has_changed;
}

}  // namespace gpu
}  // namespace xla

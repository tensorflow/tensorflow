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

#include "xla/service/gpu/transforms/convert_triton_gemm_config.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/codegen/tiling/symbolic_tile.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

using ::mlir::MLIRContext;

// Extracts the TritonGemmConfig from the given fusion's backend config.
absl::StatusOr<TritonGemmConfig> GetTritonGemmConfig(
    const HloFusionInstruction& fusion) {
  ASSIGN_OR_RETURN(auto gpu_config, fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  if (!backend_config.has_triton_gemm_config()) {
    return absl::InternalError(
        "The fusion's backend config doesn't have a triton_gemm_config.");
  }
  return TritonGemmConfig::FromProto(backend_config.triton_gemm_config());
}

absl::Status IsDot(const HloInstruction& dot) {
  if (HloPredicateIsNotOp<HloOpcode::kDot, HloOpcode::kScaledDot>(&dot)) {
    return absl::InternalError(
        absl::StrCat("Expected a dot instruction but got ", dot.ToString()));
  }
  return absl::OkStatus();
}

bool IsScaledDotWithTritonEnabled(const HloModuleConfig& module_config) {
  const DebugOptions& debug_options = module_config.debug_options();
  return debug_options.xla_gpu_experimental_scaled_dot_with_triton();
}

class ConvertTritonGemmConfigVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ConvertTritonGemmConfigVisitor(
      MLIRContext* mlir_context,
      const se::DeviceDescription& device_description)
      : mlir_context_(mlir_context), device_description_(device_description) {}

 private:
  absl::Status HandleFusion(HloInstruction* instruction) override {
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instruction);
    // Check if we target this fusion.
    absl::StatusOr<TritonGemmConfig> config = GetTritonGemmConfig(*fusion);
    if (!config.ok()) {
      VLOG(2) << "Skipping fusion as it does not have a TritonGemmConfig";
      return absl::OkStatus();
    }
    return RewriteFusion(fusion, *config);
  }

  absl::Status RewriteFusion(HloFusionInstruction* fusion,
                             const TritonGemmConfig& config) {
    HloComputation* computation = fusion->called_computation();

    std::vector<HloOpcode> dot_opcodes = {HloOpcode::kDot};
    bool scaled_dot_enabled =
        IsScaledDotWithTritonEnabled(fusion->GetModule()->config());
    if (scaled_dot_enabled) {
      dot_opcodes.push_back(HloOpcode::kScaledDot);
    }
    HloInstruction* dot =
        hlo_query::GetFirstInstructionWithOpcode(*computation, dot_opcodes);
    if (dot == nullptr) {
      VLOG(2) << "Skipping fusion as it has no dot instruction";
      return absl::OkStatus();
    }

    // Annotate the dot with the contraction tile size.
    ASSIGN_OR_RETURN(auto tile_sizes, dot->backend_config<Tile>());
    tile_sizes.add_sizes(config.block_k);
    RETURN_IF_ERROR(dot->set_backend_config(tile_sizes));

    // Annotate the fusion itself with the block-level parameters.
    ASSIGN_OR_RETURN(auto gpu_config,
                     fusion->backend_config<GpuBackendConfig>());
    FusionBackendConfig& backend_config =
        *gpu_config.mutable_fusion_backend_config();
    backend_config.clear_triton_gemm_config();
    backend_config.set_kind(kTritonNestedGemmFusionKind);

    ASSIGN_OR_RETURN(BlockLevelParameters block_level_parameters,
                     FindBlockLevelParameters(dot, config, mlir_context_,
                                              device_description_));

    *backend_config.mutable_block_level_fusion_config() =
        block_level_parameters.ToBlockLevelFusionConfig();
    RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

    MarkAsChanged();
    if (CodegenDecision can_codegen_computation = IsTritonSupportedComputation(
            *fusion->called_computation(),
            device_description_.gpu_compute_capability());
        !can_codegen_computation) {
      return absl::InternalError(absl::StrCat(
          "Computation of fusion ", fusion->ToString(),
          " is not supported by Triton: ", can_codegen_computation.Explain()));
    }
    return absl::OkStatus();
  }

  MLIRContext* mlir_context_;
  const se::DeviceDescription& device_description_;
};

}  // namespace

absl::StatusOr<bool> ConvertTritonGemmConfig::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  auto call_graph = CallGraph::Build(module, execution_threads);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ConvertTritonGemmConfigVisitor visitor(mlir_context_, device_description_);
    RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

absl::StatusOr<BlockLevelParameters> FindBlockLevelParameters(
    HloInstruction* dot, const TritonGemmConfig& config,
    MLIRContext* mlir_context,
    const se::DeviceDescription& device_description) {
  RETURN_IF_ERROR(IsDot(*dot));
  HloComputation* computation = dot->parent();
  VLOG(3) << "FindOutputTileSizesForEpilogue of computation: "
          << computation->ToString();
  SymbolicTileAnalysisOrError analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(
          *computation, mlir_context,
          TritonEmitterConstraints::GetBuilder(device_description));

  if (const auto* fusion_decision = std::get_if<FusionDecision>(&analysis_or)) {
    std::unique_ptr<HloModule> extracted_computation_module =
        ExtractInstructionIntoNewModule(*computation->FusionInstruction());
    return absl::InternalError(absl::StrCat(
        "Failed to analyze the computation (", fusion_decision->Explain(),
        "):\n", extracted_computation_module->ToString()));
  }

  const auto& analysis = std::get<SymbolicTileAnalysis>(analysis_or);
  const auto& tiled_instructions = analysis.GetSymbolicTiledHloComputation();
  auto is_dot = [&](const auto& instr) { return instr->hlo() == dot; };
  auto tiled_dot_it = absl::c_find_if(tiled_instructions, is_dot);
  if (tiled_dot_it == tiled_instructions.end()) {
    return absl::InternalError(absl::StrCat(
        "Couldn't find a symbolic tiled instruction for ", dot->ToString()));
  }
  const SymbolicTiledHloInstruction& tiled_dot = **tiled_dot_it;

  auto get_tile_sizes = [&](int64_t rank) {
    QCHECK_GE(rank, 2) << "Expected at least rank 2 for the dot, got " << rank
                       << " in computation " << computation->ToString();
    // We always expect the shape to be [1, ..., block_m, block_n], by
    // construction of GemmFusions.
    llvm::SmallVector<int64_t> tile_sizes(rank - 2, 1);
    tile_sizes.append({config.block_m, config.block_n});
    return tile_sizes;
  };

  VLOG(3) << "FindOutputTileSizesForEpilogue: dot shape: "
          << dot->shape().ToString();
  auto expected_dot_tile_sizes =
      get_tile_sizes(dot->shape().dimensions().size());
  VLOG(2) << "FindOutputTileSizesForEpilogue: " << tiled_dot.ToString()
          << "\nConstraints: "
          << analysis.GetTilingSpecification().constraints().ToString()
          << "Expected dot tile sizes: "
          << absl::StrJoin(expected_dot_tile_sizes, " ");

  // Try all permutations of the dot tile sizes to see if any of them satisfy
  // the constraints of the analysis and map to the given config of the dot.
  int64_t out_rank =
      computation->root_instruction()->shape().dimensions().size();
  VLOG(3) << "FindOutputTileSizesForEpilogue: computation root shape: "
          << computation->root_instruction()->shape().ToString();
  llvm::SmallVector<int64_t> output_tile_sizes = get_tile_sizes(out_rank);

  absl::c_sort(output_tile_sizes);

  const TilingSpecification& tiling_specification =
      analysis.GetTilingSpecification();

  do {
    VLOG(4) << "trying output_tile_sizes = ("
            << absl::StrJoin(output_tile_sizes, ",") << ")";
    Tiling::TileMapping tile_mapping;
    tile_mapping[dot] = {config.block_k};
    // If the `dot` is a root, we need to assign both the hidden parameter and
    // the output parameters to it.
    if (dot->IsRoot()) {
      tile_mapping[dot].insert(tile_mapping[dot].end(),
                               output_tile_sizes.begin(),
                               output_tile_sizes.end());
    } else {
      tile_mapping[dot->parent()->root_instruction()] = {
          output_tile_sizes.begin(), output_tile_sizes.end()};
    }

    Tiling tiling(std::move(tile_mapping));
    ASSIGN_OR_RETURN(bool parameters_satisfy_constraints,
                     analysis.ParametersSatisfyConstraints(tiling));
    if (!parameters_satisfy_constraints) {
      VLOG(4) << "Parameters don't satisfy constraints";
      continue;
    }
    ASSIGN_OR_RETURN(FlatTiling flat_tiling_parameters,
                     tiling.Flatten(tiling_specification));
    llvm::SmallVector<int64_t> mapped_dot_tile_sizes =
        EvaluateTileSizes(tiled_dot.symbolic_tile(), flat_tiling_parameters);
    if (mapped_dot_tile_sizes == expected_dot_tile_sizes) {
      BlockLevelParameters params;
      params.output_tile_sizes = {std::vector<int64_t>(
          output_tile_sizes.begin(), output_tile_sizes.end())};
      params.num_warps = config.num_warps;
      params.num_ctas = config.num_ctas;
      params.num_stages = config.num_stages;
      params.is_tma_allowed = config.is_tma_allowed;
      params.is_warp_specialization_allowed =
          config.is_warp_specialization_allowed;
      return params;
    }
    VLOG(4) << "mapped_dot_tile_sizes: "
            << absl::StrJoin(mapped_dot_tile_sizes, ",")
            << " != " << absl::StrJoin(expected_dot_tile_sizes, ",");
  } while (absl::c_next_permutation(output_tile_sizes));

  return absl::InternalError(absl::StrCat(
      "Couldn't find output tile sizes that satisfy ", tiled_dot.ToString()));
}

}  // namespace xla::gpu

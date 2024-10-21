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

#include "xla/service/gpu/transforms/nest_gemm_fusion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/instruction_fusion.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {
// Fuses the given instructions together. The instructions are expected to be
// passed in def-before-use order.  The resulting fusion has a single root
// instruction, which is the last instructions in the input span.  We only
// replace the uses of the root in 'consumer', and leave other users alone.
absl::Status FuseInstructionsForConsumer(
    const std::vector<HloInstruction*>& instructions,
    HloInstruction& consumer) {
  HloComputation::Builder builder(instructions.back()->name());

  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;
  std::vector<HloInstruction*> parameters;

  auto add_parameter = [&](HloInstruction* instruction) -> void {
    int param_index = parameters.size();
    old_to_new_mapping[instruction] =
        builder.AddInstruction(HloInstruction::CreateParameter(
            param_index, instruction->shape(),
            absl::StrCat("parameter_", param_index)));
    parameters.push_back(instruction);
  };

  for (HloInstruction* instruction : instructions) {
    if (old_to_new_mapping.contains(instruction)) {
      continue;
    }

    if (instruction->opcode() == HloOpcode::kParameter) {
      add_parameter(instruction);
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : instruction->mutable_operands()) {
      if (!old_to_new_mapping.contains(operand)) {
        add_parameter(operand);
      }
      new_operands.push_back(old_to_new_mapping[operand]);
    }
    old_to_new_mapping[instruction] = builder.AddInstruction(
        instruction->CloneWithNewOperands(instruction->shape(), new_operands));
  }

  HloInstruction* old_root = instructions.back();
  old_to_new_mapping[old_root]->MarkAsRoot();

  HloComputation* computation =
      old_root->GetModule()->AddComputationAndUnifyNamesAndIds(
          builder.Build(), /*is_entry=*/false);
  HloInstruction* fusion =
      old_root->parent()->AddInstruction(HloInstruction::CreateFusion(
          old_root->shape(), HloInstruction::FusionKind::kCustom, parameters,
          computation));
  fusion->GetModule()->SetAndUniquifyInstrName(fusion, "block_fusion");

  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kTritonFusionKind));
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  for (int64_t operand_index : consumer.OperandIndices(old_root)) {
    TF_RETURN_IF_ERROR(consumer.ReplaceOperandWith(operand_index, fusion));
  }

  return absl::OkStatus();
}

// Annotates the given nested fusion with the given tile sizes.
// Implementation for AnnotateDotLhs/RhsNestedFusion().
absl::Status AnnotateDotOperandNestedFusionImpl(
    HloFusionInstruction& nested_fusion, const HloDotInstruction& dot,
    const TritonGemmConfig& config,
    absl::Span<const int64_t> contracting_dimensions,  // Must be single element
    absl::Span<const int64_t> batch_dimensions, int64_t contracting_dim_size,
    int64_t non_contracting_dim_size) {
  if (contracting_dimensions.size() != 1) {
    return absl::InternalError(
        absl::StrCat("Expected a single lhs contracting dimension but got ",
                     contracting_dimensions.size()));
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> non_contracting_dimensions,
      GetNonContractingDims(dot.operand(0)->shape(), batch_dimensions,
                            contracting_dimensions));

  if (non_contracting_dimensions.size() != 1) {
    return absl::InternalError(
        absl::StrCat("Expected a single non-contracting dimension but got ",
                     non_contracting_dimensions.size()));
  }

  // We have a single contracting dimension, and a single non-contracting
  // dimension. All the other output tile sizes are set to 1.
  std::vector<int64_t> output_tile_sizes(dot.operand(0)->shape().rank(), 1);
  output_tile_sizes[contracting_dimensions[0]] = contracting_dim_size;
  output_tile_sizes[non_contracting_dimensions[0]] = non_contracting_dim_size;

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = std::move(output_tile_sizes);

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      nested_fusion.backend_config<GpuBackendConfig>());
  *backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  TF_RETURN_IF_ERROR(nested_fusion.set_backend_config(backend_config));

  return absl::OkStatus();
}

absl::Status AnnotateDotLhsNestedFusion(HloFusionInstruction& nested_fusion,
                                        const HloDotInstruction& dot,
                                        const TritonGemmConfig& config) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  return AnnotateDotOperandNestedFusionImpl(
      nested_fusion, dot, config,
      dimension_numbers.lhs_contracting_dimensions(),
      dimension_numbers.lhs_batch_dimensions(), config.block_k, config.block_m);
}

absl::Status AnnotateDotRhsNestedFusion(HloFusionInstruction& nested_fusion,
                                        const HloDotInstruction& dot,
                                        const TritonGemmConfig& config) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  return AnnotateDotOperandNestedFusionImpl(
      nested_fusion, dot, config,
      dimension_numbers.rhs_contracting_dimensions(),
      dimension_numbers.rhs_batch_dimensions(), config.block_k, config.block_n);
}

// Finds tile sizes for the root of the analysis that satisfy the
// requirements of the dot. That is, the tile sizes need to satisfy the
// constraints of the analysis and map to the given config of the dot.
absl::StatusOr<llvm::SmallVector<int64_t>> FindOutputTileSizesForEpilogue(
    const SymbolicTiledHloInstruction& tiled_dot,
    const SymbolicTileAnalysis& analysis, const TritonGemmConfig& config) {
  int64_t dot_rank = tiled_dot.symbolic_tile().tile_map().GetDimensionCount();
  llvm::SmallVector<int64_t> expected_dot_tile_sizes(dot_rank, 1);
  // We always expect the shape of the dot to be [1, ..., block_m, block_n].
  expected_dot_tile_sizes[dot_rank - 2] = config.block_m;
  expected_dot_tile_sizes[dot_rank - 1] = config.block_n;

  // Try all permutations of the dot tile sizes to see if any of them satisfy
  // the constraints of the analysis and map to the given config of the dot.
  llvm::SmallVector<int64_t> output_tile_sizes = expected_dot_tile_sizes;
  std::sort(output_tile_sizes.begin(), output_tile_sizes.end());
  do {
    TF_ASSIGN_OR_RETURN(
        bool parameters_satisfy_constraints,
        analysis.ParametersSatisfyConstraints(output_tile_sizes));
    if (!parameters_satisfy_constraints) {
      continue;
    }
    auto mapped_dot_tile_sizes = tiled_dot.TileSizes(output_tile_sizes);
    if (mapped_dot_tile_sizes == expected_dot_tile_sizes) {
      return output_tile_sizes;
    }
  } while (std::next_permutation(output_tile_sizes.begin(),
                                 output_tile_sizes.end()));

  return absl::InternalError(absl::StrCat(
      "Couldn't find output tile sizes that satisfy ", tiled_dot.ToString()));
}

// Extracts the TritonGemmConfig from the given fusion's backend config.
absl::StatusOr<TritonGemmConfig> GetTritonGemmConfig(
    const HloFusionInstruction& fusion) {
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  if (!backend_config.has_triton_gemm_config()) {
    return absl::InternalError(
        "The fusion's backend config doesn't have a triton_gemm_config.");
  }
  return TritonGemmConfig::FromProto(backend_config.triton_gemm_config());
}

// Transforms a fusion into an equivalent nested fusion if it has a single dot.
// Returns true if the transformation was successful.
absl::Status MakeNestedFusionFromGemmFusion(
    HloFusionInstruction* fusion, const TritonGemmConfig& config,
    const SymbolicTileAnalysis& analysis,
    const SymbolicTiledHloInstruction& tiled_dot, HloDotInstruction* dot) {
  DCHECK(GetTritonGemmConfig(*fusion).value() == config);
  DCHECK_EQ(tiled_dot.hlo(), dot);

  HloComputation* computation = fusion->called_computation();

  // Left-hand side of the dot.
  TF_RETURN_IF_ERROR(FuseInstructionsForConsumer(
      computation->MakeInstructionPostOrderFrom(*dot->mutable_operand(0)),
      *dot));
  TF_RETURN_IF_ERROR(AnnotateDotLhsNestedFusion(
      *::xla::Cast<HloFusionInstruction>(dot->mutable_operand(0)), *dot,
      config));

  // Right-hand side of the dot.
  TF_RETURN_IF_ERROR(FuseInstructionsForConsumer(
      computation->MakeInstructionPostOrderFrom(*dot->mutable_operand(1)),
      *dot));
  TF_RETURN_IF_ERROR(AnnotateDotRhsNestedFusion(
      *::xla::Cast<HloFusionInstruction>(dot->mutable_operand(1)), *dot,
      config));

  // Delete newly unused instructions, if any.
  TF_ASSIGN_OR_RETURN([[maybe_unused]] bool changed,
                      HloDCE::RunOnComputation(
                          computation,
                          /*remove_cross_partition_collective_ops=*/false));

  // Annotate the fusion itself.
  TF_ASSIGN_OR_RETURN(
      llvm::SmallVector<int64_t> output_tile_sizes,
      FindOutputTileSizesForEpilogue(tiled_dot, analysis, config));

  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kTritonFusionKind));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes.assign(output_tile_sizes.begin(),
                                                  output_tile_sizes.end());

  *backend_config.mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  return absl::OkStatus();
}

size_t GetDotCount(HloComputation* computation) {
  return absl::c_count_if(computation->instructions(), [](HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kDot;
  });
}

class NestGemmFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit NestGemmFusionVisitor(mlir::MLIRContext* ctx) : ctx_(ctx) {}

  absl::Status HandleFusion(HloInstruction* instruction) override {
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instruction);

    absl::StatusOr<TritonGemmConfig> config = GetTritonGemmConfig(*fusion);
    if (!config.ok()) {
      return absl::OkStatus();  // Skip because it's not a Triton gemm fusion.
    }

    HloComputation* computation = fusion->called_computation();
    HloInstruction* dot =
        hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
    if (dot == nullptr) {
      return absl::OkStatus();  // Skip because fusion has no dot.
    }
    DCHECK_EQ(GetDotCount(computation), 1) << "Fusion has more than one dot.";
    SymbolicTileAnalysisOrError analysis_or =
        SymbolicTileAnalysis::AnalyzeComputation(
            *fusion->called_computations()[0], ctx_);

    if (std::holds_alternative<FusionDecision>(analysis_or)) {
      return absl::InternalError(
          absl::StrCat("Failed to analyze the computation (",
                       std::get<FusionDecision>(analysis_or).Explain(),
                       "): ", fusion->called_computation()->ToString()));
    }

    auto& analysis = std::get<SymbolicTileAnalysis>(analysis_or);
    auto tiled_dot_it = absl::c_find_if(
        analysis.GetSymbolicTiledHloComputation(),
        [&](const auto& tiled_hlo) { return tiled_hlo->hlo() == dot; });
    if (tiled_dot_it == analysis.GetSymbolicTiledHloComputation().end()) {
      return absl::InternalError(absl::StrCat(
          "Couldn't find a symbolic tiled instruction for ", dot->ToString()));
    }

    TF_RETURN_IF_ERROR(MakeNestedFusionFromGemmFusion(
        fusion, config.value(), analysis, **tiled_dot_it,
        Cast<HloDotInstruction>(dot)));
    this->MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  mlir::MLIRContext* ctx_;
};

}  // namespace

absl::StatusOr<bool> NestGemmFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  mlir::MLIRContext ctx;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    NestGemmFusionVisitor visitor(&ctx);
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace xla::gpu

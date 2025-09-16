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
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tools/hlo_extractor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// Creates a fusion for instructions starting from 'root' and returns it.
absl::StatusOr<HloInstruction*> FuseInstructionsFromRoot(HloInstruction& root) {
  std::vector<HloInstruction*> instructions =
      root.parent()->MakeInstructionPostOrderFrom(root);

  HloComputation::Builder builder(root.name());

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

    if (HloPredicateIsOp<HloOpcode::kParameter>(instruction)) {
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
  old_to_new_mapping[&root]->MarkAsRoot();

  HloComputation* computation =
      root.GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                          /*is_entry=*/false);
  HloInstruction* fusion =
      root.parent()->AddInstruction(HloInstruction::CreateFusion(
          root.shape(), HloInstruction::FusionKind::kCustom, parameters,
          computation));
  fusion->GetModule()->SetAndUniquifyInstrName(fusion, "block_fusion");

  return fusion;
}

// Fuses the instructions starting from 'root' for 'consumer'. Other consumers
// of 'root' are not affected. Annotates fusion with
// `kTritonNestedGemmFusionKind`.
absl::Status FuseInstructionsForConsumer(HloInstruction& root,
                                         HloInstruction& consumer) {
  CHECK(absl::c_count(consumer.operands(), &root) != 0)
      << "Consumer " << consumer.ToString() << " does not use root "
      << root.ToString();

  TF_ASSIGN_OR_RETURN(HloInstruction * fusion, FuseInstructionsFromRoot(root));

  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion->backend_config<GpuBackendConfig>());
  gpu_config.mutable_fusion_backend_config()->set_kind(
      kTritonNestedGemmFusionKind);
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  for (int64_t operand_index : consumer.OperandIndices(&root)) {
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
  std::vector<int64_t> output_tile_sizes(
      dot.operand(0)->shape().dimensions().size(), 1);
  output_tile_sizes[contracting_dimensions[0]] = contracting_dim_size;
  output_tile_sizes[non_contracting_dimensions[0]] = non_contracting_dim_size;

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {std::move(output_tile_sizes)};
  block_level_parameters.num_warps = config.num_warps;
  block_level_parameters.num_ctas = config.num_ctas;
  block_level_parameters.num_stages = config.num_stages;
  block_level_parameters.is_tma_allowed = config.is_tma_allowed;

  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      nested_fusion.backend_config<GpuBackendConfig>());
  *gpu_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  TF_RETURN_IF_ERROR(nested_fusion.set_backend_config(gpu_config));

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

// Constructs nested fusion nodes for the operands of `concatenate` instructions
// and annotates them with `kTritonNestedGemmFusionKind`.
absl::Status FuseAndAnnotateConcatOperands(HloComputation* computation) {
  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() != HloOpcode::kConcatenate) {
      continue;
    }
    for (HloInstruction* operand : instr->mutable_operands()) {
      TF_RETURN_IF_ERROR(FuseInstructionsForConsumer(*operand, *instr));
    }
  }
  return absl::OkStatus();
}

// Transforms a fusion into an equivalent nested fusion if it has a single dot.
// Returns ok if the transformation was successful.
absl::Status MakeNestedFusionFromGemmFusion(HloFusionInstruction* fusion,
                                            HloDotInstruction* dot,
                                            mlir::MLIRContext* ctx) {
  TF_ASSIGN_OR_RETURN(TritonGemmConfig config, GetTritonGemmConfig(*fusion));

  HloComputation* computation = fusion->called_computation();

  // First, create nested fusions for the operands of `concatenate` instructions
  // if they exist.
  TF_RETURN_IF_ERROR(FuseAndAnnotateConcatOperands(computation));

  // Left-hand side of the dot.
  TF_RETURN_IF_ERROR(
      FuseInstructionsForConsumer(*dot->mutable_operand(0), *dot));
  TF_RETURN_IF_ERROR(AnnotateDotLhsNestedFusion(
      *::xla::Cast<HloFusionInstruction>(dot->mutable_operand(0)), *dot,
      config));

  // Right-hand side of the dot.
  TF_RETURN_IF_ERROR(
      FuseInstructionsForConsumer(*dot->mutable_operand(1), *dot));
  TF_RETURN_IF_ERROR(AnnotateDotRhsNestedFusion(
      *::xla::Cast<HloFusionInstruction>(dot->mutable_operand(1)), *dot,
      config));

  // Delete newly unused instructions, if any.
  TF_ASSIGN_OR_RETURN([[maybe_unused]] bool changed,
                      HloDCE::RunOnComputation(
                          computation,
                          /*remove_cross_partition_collective_ops=*/false));

  // Annotate the fusion itself.
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.clear_triton_gemm_config();
  backend_config.set_kind(kTritonNestedGemmFusionKind);

  TF_ASSIGN_OR_RETURN(
      BlockLevelParameters block_level_parameters,
      ::xla::gpu::detail::FindBlockLevelParameters(dot, config, ctx));

  *backend_config.mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  return absl::OkStatus();
}

size_t GetDotCount(HloComputation* computation) {
  return absl::c_count_if(computation->instructions(),
                          HloPredicateIsOp<HloOpcode::kDot>);
}

using HloInstructionSetVector =
    llvm::SetVector<HloInstruction*, std::vector<HloInstruction*>,
                    HloInstructionSet>;

// Returns the set of instructions that are reachable from 'instruction' using
// the given accessor.
template <typename T>
HloInstructionSetVector GetTransitiveInstructionSet(
    const HloInstruction* instruction, T (HloInstruction::*get)() const) {
  std::deque<HloInstruction*> worklist;
  auto append = [&](const auto& instructions) {
    worklist.insert(worklist.end(), instructions.begin(), instructions.end());
  };
  append((instruction->*get)());
  HloInstructionSetVector result;
  while (!worklist.empty()) {
    HloInstruction* front = worklist.front();
    worklist.pop_front();
    if (result.insert(front)) {
      append((front->*get)());
    }
  }
  return result;
}

// Returns the set of producers reachable from 'instruction' in use-before-def
// order.
HloInstructionSetVector GetProducerSet(const HloInstruction* instruction) {
  return GetTransitiveInstructionSet(instruction, &HloInstruction::operands);
}
// Returns the set of consumers reachable from 'instruction' in def-before-use
// order.
HloInstructionSetVector GetConsumerSet(const HloInstruction* instruction) {
  return GetTransitiveInstructionSet(instruction, &HloInstruction::users);
}

// Verifies that the set of instructions is closed under the given accessor,
// i.e. that the set of instructions reachable through the given accessor are
// either in the set itself or the root.
template <typename T>
absl::Status VerifyIsClosedInstructionSet(
    const HloInstructionSetVector& instructions, const HloInstruction* root,
    T (HloInstruction::*get)() const) {
  for (HloInstruction* instruction : instructions) {
    for (HloInstruction* reachable : (instruction->*get)()) {
      if (reachable != root && instructions.count(reachable) == 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Instruction ", reachable->ToString(),
                         " is reachable from ", instruction->ToString(),
                         ", which is not in the recursive set of, or ",
                         root->ToString(), " itself."));
      }
    }
  }

  return absl::OkStatus();
}

bool IsFeatureEnabled(const HloModule* module,
                      DebugOptions::GenericTritonEmitterFeature feature) {
  return absl::c_contains(
      module->config()
          .debug_options()
          .xla_gpu_unsupported_generic_triton_emitter_features(),
      feature);
}

class NestGemmFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit NestGemmFusionVisitor(
      mlir::MLIRContext* ctx, CallGraph* call_graph,
      const se::GpuComputeCapability compute_capability)
      : ctx_(ctx),
        call_graph_(call_graph),
        compute_capability_(compute_capability) {}

 private:
  absl::Status AcceptDotOperand(const HloInstruction* operand,
                                absl::Span<const int64_t> batch_dims,
                                absl::Span<const int64_t> contracting_dims,
                                bool is_lhs) {
    if (contracting_dims.size() != 1) {
      return absl::InternalError(
          absl::StrCat("Expected ", is_lhs ? "LHS" : "RHS",
                       " operand with exactly one contracting dimension, got ",
                       contracting_dims.size()));
    }

    TF_ASSIGN_OR_RETURN(
        std::vector<int64_t> non_contracting_dimensions,
        GetNonContractingDims(operand->shape(), batch_dims, contracting_dims));

    if (non_contracting_dimensions.size() != 1) {
      return absl::InternalError(absl::StrCat(
          "Expected ", is_lhs ? "LHS" : "RHS",
          " operand with exactly one non-contracting dimension, got ",
          non_contracting_dimensions.size()));
    }

    if (is_lhs) {
      if (non_contracting_dimensions[0] >= contracting_dims[0]) {
        return absl::InternalError(absl::StrCat(
            "Expected LHS non-contracting dimension to be before contracting "
            "dimension, got ",
            non_contracting_dimensions[0], " >= ", contracting_dims[0]));
      }
    } else {
      if (non_contracting_dimensions[0] <= contracting_dims[0]) {
        return absl::InternalError(absl::StrCat(
            "Expected RHS non-contracting dimension to be after contracting "
            "dimension, got ",
            non_contracting_dimensions[0], " <= ", contracting_dims[0]));
      }
    }
    return absl::OkStatus();
  }

  absl::Status AcceptDotInstruction(const HloDotInstruction* dot) {
    if (IsFeatureEnabled(
            dot->GetModule(),
            DebugOptions::GENERIC_TRITON_EMITTER_ALLOW_ALL_GEMM_SHAPES)) {
      return absl::OkStatus();
    }
    const HloInstruction* lhs = dot->operand(0);
    const HloInstruction* rhs = dot->operand(1);
    auto dims = dot->dot_dimension_numbers();
    TF_RETURN_IF_ERROR(AcceptDotOperand(lhs, dims.lhs_batch_dimensions(),
                                        dims.lhs_contracting_dimensions(),
                                        /*is_lhs=*/true));
    TF_RETURN_IF_ERROR(AcceptDotOperand(rhs, dims.rhs_batch_dimensions(),
                                        dims.rhs_contracting_dimensions(),
                                        /*is_lhs=*/false));
    return absl::OkStatus();
  }

  absl::Status AcceptNestedInstruction(const HloInstruction* instruction) {
    if (instruction->IsElementwise()) {
      return absl::OkStatus();
    }
    switch (instruction->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kBroadcast:
        return absl::OkStatus();
      case HloOpcode::kFusion:
        return AcceptResultingFusion(Cast<HloFusionInstruction>(instruction));
      case HloOpcode::kDot:
        return AcceptDotInstruction(Cast<HloDotInstruction>(instruction));
      default:
        if (!IsFeatureEnabled(
                instruction->GetModule(),
                DebugOptions::
                    GENERIC_TRITON_EMITTER_ALLOW_ALL_OPS_IN_GEMM_FUSION)) {
          return absl::InternalError(absl::StrCat(
              "Instruction ", HloOpcodeString(instruction->opcode()),
              " is not allowed in nested GEMM fusion."));
        }
        return absl::OkStatus();
    }
  }

  // Checks whether all operations are from the "tested" set that we confirmed
  // to not cause regressions.
  // That enables a progressive rollout of the new emitter. Eventually we should
  // remove this check completely as all computations will be supported by the
  // generic emitter and performance regressions will be addressed.
  absl::Status AcceptResultingFusion(const HloFusionInstruction* fusion) {
    const HloComputation* computation = fusion->called_computation();
    for (const HloInstruction* instruction : computation->instructions()) {
      TF_RETURN_IF_ERROR(AcceptNestedInstruction(instruction));
    }
    return absl::OkStatus();
  }

  absl::Status RewriteFusion(HloFusionInstruction* fusion,
                             CallGraph* call_graph) {
    HloComputation* computation = fusion->called_computation();
    HloInstruction* instr =
        hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
    if (instr == nullptr) {
      return absl::InternalError(absl::StrCat("Computation of fusion ",
                                              fusion->ToString(),
                                              " has no dot instruction"));
    }
    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
    TF_RETURN_IF_ERROR(MakeNestedFusionFromGemmFusion(fusion, dot, ctx_));

    MarkAsChanged();

    if (CodegenDecision can_codegen_computation = IsTritonSupportedComputation(
            *fusion->called_computation(), compute_capability_);
        !can_codegen_computation) {
      return absl::InternalError(absl::StrCat(
          "Computation of fusion ", fusion->ToString(),
          " is not supported by Triton: ", can_codegen_computation.Explain()));
    }

    return AcceptResultingFusion(fusion);
  }

  absl::Status HandleFusion(HloInstruction* instruction) override {
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instruction);

    // Check if we target this fusion.
    absl::StatusOr<TritonGemmConfig> config = GetTritonGemmConfig(*fusion);
    if (!config.ok()) {
      VLOG(2) << "Skipping fusion as it does not have a TritonGemmConfig";
      return absl::OkStatus();
    }
    HloComputation* computation = fusion->called_computation();
    HloInstruction* instr =
        hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
    if (instr == nullptr) {
      VLOG(2) << "Skipping fusion as it has no dot instruction";
      return absl::OkStatus();
    }

    {
      // Symbolic tile analysis and nesting do not support all HLOs yet and
      // might leave the module in an invalid state. To avoid that we first dry
      // run the rewrite on an extracted module.
      // TODO(b/393299275): remove dry-run once we can handle all HLOs.
      std::unique_ptr<HloModule> extracted_module =
          ExtractInstructionIntoNewModule(*fusion);
      extracted_module->mutable_config().set_debug_options(
          fusion->GetModule()->config().debug_options());
      HloComputation* entry = extracted_module->entry_computation();
      HloFusionInstruction* extracted_fusion =
          Cast<HloFusionInstruction>(entry->root_instruction());
      if (extracted_fusion == nullptr) {
        return absl::InternalError(absl::StrCat(
            "Failed to create a cloned module for fusion ", fusion->name()));
      }
      std::unique_ptr<CallGraph> cloned_call_graph =
          CallGraph::Build(extracted_module.get(), {});
      absl::Status status =
          RewriteFusion(extracted_fusion, cloned_call_graph.get());
      if (!status.ok()) {
        VLOG(2) << "Failed to rewrite the fusion " << fusion->ToString()
                << " in a cloned module: " << status;
        if (IsFeatureEnabled(
                fusion->GetModule(),
                DebugOptions::GENERIC_TRITON_EMITTER_DISABLE_LEGACY_GEMM)) {
          // As legacy emitter is disabled we are doomed to fail now, returning
          // the dry run result failure as it is a better diagnostic.
          return status;
        }
        return absl::OkStatus();
      }
    }
    absl::Status status = RewriteFusion(fusion, call_graph_);
    VLOG(2) << "RewriteFusion " << fusion->name() << ": " << status;
    return status;
  }

 private:
  mlir::MLIRContext* ctx_;
  CallGraph* call_graph_;
  const se::GpuComputeCapability compute_capability_;
};

}  // namespace

absl::StatusOr<bool> NestGemmFusion::RunOnModule(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  auto call_graph = CallGraph::Build(module, execution_threads);
  mlir::MLIRContext ctx;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    NestGemmFusionVisitor visitor(&ctx, call_graph.get(), compute_capability_);
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

absl::StatusOr<bool> NestGemmFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "--xla_gpu_unsupported_generic_triton_emitter_features="
          << absl::StrJoin(
                 module->config()
                     .debug_options()
                     .xla_gpu_unsupported_generic_triton_emitter_features(),
                 ",");
  if (!IsFeatureEnabled(
          module, DebugOptions::GENERIC_TRITON_EMITTER_ENABLE_NESTED_GEMM)) {
    VLOG(1) << "Generic Triton emitter for gemms is disabled, exiting";
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool result, RunOnModule(module, execution_threads));
  return result;
}

namespace detail {

absl::StatusOr<BlockLevelParameters> FindBlockLevelParameters(
    HloDotInstruction* dot, const TritonGemmConfig& config,
    mlir::MLIRContext* ctx) {
  HloComputation* computation = dot->parent();
  VLOG(3) << "FindOutputTileSizesForEpilogue of computation: "
          << computation->ToString();
  SymbolicTileAnalysisOrError analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(*computation, ctx);
  if (std::holds_alternative<FusionDecision>(analysis_or)) {
    std::unique_ptr<HloModule> extracted_computation_module =
        ExtractModule(computation->FusionInstruction());
    return absl::InternalError(
        absl::StrCat("Failed to analyze the computation (",
                     std::get<FusionDecision>(analysis_or).Explain(),
                     "): ", extracted_computation_module->ToString()));
  }

  auto& analysis = std::get<SymbolicTileAnalysis>(analysis_or);
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

  std::sort(output_tile_sizes.begin(), output_tile_sizes.end());

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
    TF_ASSIGN_OR_RETURN(bool parameters_satisfy_constraints,
                        analysis.ParametersSatisfyConstraints(tiling));
    if (!parameters_satisfy_constraints) {
      VLOG(4) << "Parameters don't satisfy constraints";
      continue;
    }
    TF_ASSIGN_OR_RETURN(FlatTiling flat_tiling_parameters,
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
      return params;
    }
    VLOG(4) << "mapped_dot_tile_sizes: "
            << absl::StrJoin(mapped_dot_tile_sizes, ",")
            << " != " << absl::StrJoin(expected_dot_tile_sizes, ",");
  } while (std::next_permutation(output_tile_sizes.begin(),
                                 output_tile_sizes.end()));

  return absl::InternalError(absl::StrCat(
      "Couldn't find output tile sizes that satisfy ", tiled_dot.ToString()));
}

}  // namespace detail
}  // namespace xla::gpu

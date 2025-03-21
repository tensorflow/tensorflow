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
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
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
#include "xla/service/call_graph.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {
// Fuses the given instructions together. The instructions are expected to be
// passed in def-before-use order.  The resulting fusion has a single root
// instruction, which is the last instructions in the input span.  We only
// replace the uses of the root in 'consumer', and leave other users alone.
absl::Status FuseInstructionsForConsumer(
    absl::Span<HloInstruction* const> instructions, HloInstruction& consumer) {
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
  backend_config.set_kind(std::string(kTritonNestedGemmFusionKind));
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
  std::vector<int64_t> output_tile_sizes(
      dot.operand(0)->shape().dimensions_size(), 1);
  output_tile_sizes[contracting_dimensions[0]] = contracting_dim_size;
  output_tile_sizes[non_contracting_dimensions[0]] = non_contracting_dim_size;

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {std::move(output_tile_sizes)};

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
    HloDotInstruction* dot, const TritonGemmConfig& config,
    mlir::MLIRContext* ctx) {
  HloComputation* computation = dot->parent();
  SymbolicTileAnalysisOrError analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(*computation, ctx);
  if (std::holds_alternative<FusionDecision>(analysis_or)) {
    return absl::InternalError(
        absl::StrCat("Failed to analyze the computation (",
                     std::get<FusionDecision>(analysis_or).Explain(),
                     "): ", computation->ToString()));
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
    CHECK_GE(rank, 2);
    // We always expect the shape to be [1, ..., block_m, block_n], by
    // construction of GemmFusions.
    llvm::SmallVector<int64_t> tile_sizes(rank - 2, 1);
    tile_sizes.append({config.block_m, config.block_n});
    return tile_sizes;
  };

  auto expected_dot_tile_sizes = get_tile_sizes(dot->shape().dimensions_size());
  if (VLOG_IS_ON(2)) {
    std::ostringstream oss;
    for (const auto& size : expected_dot_tile_sizes) {
      oss << size << " ";
    }
    LOG(INFO) << "FindOutputTileSizesForEpilogue: " << tiled_dot.ToString()
              << "Constraints: " << analysis.GetConstraints().ToString()
              << "Expected dot tile sizes: " << oss.str();
  }

  // Try all permutations of the dot tile sizes to see if any of them satisfy
  // the constraints of the analysis and map to the given config of the dot.
  int64_t out_rank = computation->root_instruction()->shape().dimensions_size();
  auto output_tile_sizes = get_tile_sizes(out_rank);
  std::sort(output_tile_sizes.begin(), output_tile_sizes.end());
  do {
    TF_ASSIGN_OR_RETURN(
        bool parameters_satisfy_constraints,
        analysis.ParametersSatisfyConstraints(output_tile_sizes));
    if (!parameters_satisfy_constraints) {
      continue;
    }
    auto mapped_dot_tile_sizes =
        EvaluateTileSizes(tiled_dot.symbolic_tile(), output_tile_sizes);
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
// Returns ok if the transformation was successful.
absl::Status MakeNestedFusionFromGemmFusion(HloFusionInstruction* fusion,
                                            const TritonGemmConfig& config,
                                            HloDotInstruction* dot,
                                            mlir::MLIRContext* ctx) {
  DCHECK(GetTritonGemmConfig(*fusion).value() == config);

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
  TF_ASSIGN_OR_RETURN(llvm::SmallVector<int64_t> output_tile_sizes,
                      FindOutputTileSizesForEpilogue(dot, config, ctx));

  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.clear_triton_gemm_config();
  backend_config.set_kind(std::string(kTritonNestedGemmFusionKind));

  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {
      std::vector<int64_t>(output_tile_sizes.begin(), output_tile_sizes.end())};

  *backend_config.mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  return absl::OkStatus();
}

size_t GetDotCount(HloComputation* computation) {
  return absl::c_count_if(computation->instructions(),
                          HloPredicateIsOp<HloOpcode::kDot>);
}

// Returns the set of instructions that are reachable from 'instruction' using
// the given accessor.
template <typename T>
HloInstructionSet GetTransitiveInstructionSet(HloInstruction* instruction,
                                              T (HloInstruction::*get)()
                                                  const) {
  std::deque<HloInstruction*> worklist;
  auto append = [&](const auto& instructions) {
    worklist.insert(worklist.end(), instructions.begin(), instructions.end());
  };
  append((instruction->*get)());
  HloInstructionSet result;
  while (!worklist.empty()) {
    HloInstruction* front = worklist.front();
    worklist.pop_front();
    if (result.insert(front).second) {
      append((front->*get)());
    }
  }
  return result;
}

// Returns the set of producers reachable from 'instruction'.
HloInstructionSet GetProducerSet(HloInstruction* instruction) {
  return GetTransitiveInstructionSet(instruction, &HloInstruction::operands);
}
// Returns the set of consumers reachable from 'instruction'.
HloInstructionSet GetConsumerSet(HloInstruction* instruction) {
  return GetTransitiveInstructionSet(instruction, &HloInstruction::users);
}

// Verifies that the set of instructions is closed under the given accessor,
// i.e. that the set of instructions reachable through the given accessor are
// either in the set itself or the root.
template <typename T>
absl::Status VerifyIsClosedInstructionSet(const HloInstructionSet& instructions,
                                          HloInstruction* root,
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

absl::Status VerifyIsClosedProducerSet(const HloInstructionSet& instructions,
                                       HloInstruction* root) {
  return VerifyIsClosedInstructionSet(instructions, root,
                                      &HloInstruction::users);
}
absl::Status VerifyIsClosedConsumerSet(const HloInstructionSet& instructions,
                                       HloInstruction* root) {
  return VerifyIsClosedInstructionSet(instructions, root,
                                      &HloInstruction::operands);
}

// Returns true if it's safe to hoist a bitcast past the given instruction.
bool IsSafeToHoistPast(HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kBitcast:
      return true;
    default:
      return instruction->IsElementwise();
  }
}

// Hoists the given 'bitcast' upwards out of its computation, to the parent of
// each caller.
absl::Status HoistBitcastUpwardsToCallers(
    HloInstruction* bitcast, const std::vector<HloInstruction*>& callers) {
  HloInstructionSet producers = GetProducerSet(bitcast);
  TF_RETURN_IF_ERROR(VerifyIsClosedProducerSet(producers, bitcast));

  if (auto it = absl::c_find_if_not(producers, IsSafeToHoistPast);
      it != producers.end()) {
    return absl::InternalError(
        absl::StrCat("Cannot hoist bitcast past ", (*it)->ToString()));
  }

  // Adjust the shape of every producer instruction.
  Shape shape = bitcast->shape();
  for (HloInstruction* instruction : producers) {
    *instruction->mutable_shape() = shape;
    if (HloPredicateIsNotOp<HloOpcode::kParameter>(instruction)) {
      continue;
    }
    // For parameters, we need to bitcast the caller's operand.
    int64_t number = instruction->parameter_number();
    for (HloInstruction* caller : callers) {
      HloInstruction* new_bitcast =
          caller->AddInstruction(HloInstruction::CreateBitcast(
              shape, caller->mutable_operand(number)));
      TF_RETURN_IF_ERROR(
          caller->ReplaceOperandWithDifferentShape(number, new_bitcast));
    }
  }

  TF_RETURN_IF_ERROR(bitcast->ReplaceAllUsesWith(bitcast->mutable_operand(0)));
  TF_RETURN_IF_ERROR(bitcast->parent()->RemoveInstruction(bitcast));
  return absl::OkStatus();
}

// Hoists the given 'bitcast' downwards out of its computation, to the parent of
// each caller.
absl::Status HoistBitcastDownwardsToCallers(
    HloInstruction* bitcast, const std::vector<HloInstruction*>& callers) {
  HloInstructionSet consumers = GetConsumerSet(bitcast);
  TF_RETURN_IF_ERROR(VerifyIsClosedConsumerSet(consumers, bitcast));
  auto is_root = [](HloInstruction* instr) { return instr->IsRoot(); };
  CHECK(is_root(bitcast) || absl::c_any_of(consumers, is_root))
      << "Expected" << bitcast->ToString()
      << " to be a root or have a root consumer.";

  if (auto it = absl::c_find_if_not(consumers, IsSafeToHoistPast);
      it != consumers.end()) {
    return absl::InternalError(
        absl::StrCat("Cannot hoist bitcast past ", (*it)->ToString()));
  }

  // Adjust the shape of of every consumer instruction.
  Shape shape = bitcast->operand(0)->shape();
  for (HloInstruction* instruction : consumers) {
    *instruction->mutable_shape() = shape;
  }

  // Insert new bitcast for each caller's result.
  for (HloInstruction* caller : callers) {
    HloInstruction* new_bitcast = caller->AddInstruction(
        HloInstruction::CreateBitcast(caller->shape(), caller));
    TF_RETURN_IF_ERROR(caller->ReplaceAllUsesWith(new_bitcast));
    *caller->mutable_shape() = shape;
  }

  TF_RETURN_IF_ERROR(
      bitcast->ReplaceAllUsesWithDifferentShape(bitcast->mutable_operand(0)));
  TF_RETURN_IF_ERROR(bitcast->parent()->RemoveInstruction(bitcast));
  return absl::OkStatus();
}

// Try hoisting bitcasts in the computation away from 'dot' to the callers of
// the computation. Some bitcasts may remain in the computation, because they
// cannot be hoisted across all ops (e.g. across a transpose). This is not
// reported as an error.
absl::Status TryHoistBitcastsInComputationToCallers(HloInstruction* dot,
                                                    CallGraph* call_graph) {
  auto callers = call_graph->GetComputationCallers(dot->parent());
  for (HloInstruction* instruction : GetProducerSet(dot)) {
    if (HloPredicateIsNotOp<HloOpcode::kBitcast>(instruction)) {
      continue;
    }
    VLOG(2) << "Hoisting bitcast upwards " << instruction->ToString();
    auto status = HoistBitcastUpwardsToCallers(instruction, callers);
    if (!status.ok()) {
      VLOG(2) << "Failed to hoist bitcast upwards: " << status;
    }
  }
  for (HloInstruction* instruction : GetConsumerSet(dot)) {
    if (HloPredicateIsNotOp<HloOpcode::kBitcast>(instruction)) {
      continue;
    }
    VLOG(2) << "Hoisting bitcast downwards " << instruction->ToString();
    auto status = HoistBitcastDownwardsToCallers(instruction, callers);
    if (!status.ok()) {
      VLOG(2) << "Failed to hoist bitcast downwards: " << status;
    }
  }
  return absl::OkStatus();
}

class NestGemmFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit NestGemmFusionVisitor(mlir::MLIRContext* ctx, CallGraph* call_graph)
      : ctx_(ctx), call_graph_(call_graph) {}

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

    TF_RETURN_IF_ERROR(
        TryHoistBitcastsInComputationToCallers(dot, call_graph_));
    VLOG(2) << "After hoisting bitcasts: " << computation->ToString();
    TF_RETURN_IF_ERROR(MakeNestedFusionFromGemmFusion(
        fusion, config.value(), Cast<HloDotInstruction>(dot), ctx_));

    this->MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  mlir::MLIRContext* ctx_;
  CallGraph* call_graph_;
};

}  // namespace

absl::StatusOr<bool> NestGemmFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  auto call_graph = CallGraph::Build(module, execution_threads);
  mlir::MLIRContext ctx;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    NestGemmFusionVisitor visitor(&ctx, call_graph.get());
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace xla::gpu

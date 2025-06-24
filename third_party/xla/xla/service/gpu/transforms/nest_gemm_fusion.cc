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
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
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
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
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
      std::string(kTritonNestedGemmFusionKind));
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
                                            const TritonGemmConfig& config,
                                            HloDotInstruction* dot,
                                            mlir::MLIRContext* ctx) {
  DCHECK(GetTritonGemmConfig(*fusion).value() == config);

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
  backend_config.set_kind(std::string(kTritonNestedGemmFusionKind));

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

absl::Status VerifyIsClosedProducerSet(
    const HloInstructionSetVector& instructions, const HloInstruction* root) {
  return VerifyIsClosedInstructionSet(instructions, root,
                                      &HloInstruction::users);
}
absl::Status VerifyIsClosedConsumerSet(
    const HloInstructionSetVector& instructions, const HloInstruction* root) {
  return VerifyIsClosedInstructionSet(instructions, root,
                                      &HloInstruction::operands);
}

absl::Status VerifyDefaultLayout(const Layout& layout) {
  if (!LayoutUtil::IsMonotonicWithDim0Major(layout)) {
    // TODO(b/393299275): do we need to support non-default layouts?
    return absl::UnimplementedError(absl::StrCat(
        "Only default layouts are supported, got ", layout.ToString()));
  }
  return absl::OkStatus();
}

// Parameters to rewrite a reshape(broadcast/transpose) as
// broadcast/transpose(reshape) and vice versa.
struct ReshapeParams {
  Shape new_shape;                // The reshape output shape.
  std::vector<int64_t> new_dims;  // The dims of the broadcast/transpose.
};

// Returns parameters to rewrite a broadcast + reshape as reshape + broadcast.
//
// Example:
//
// b = broadcast(operand)
// c = T[reshape_shape] reshape(b)
//
// to
//
// d = T[new_shape] reshape(operand)
// c = T[reshape_shape] broadcast(d), dimensions={new_dims}.
//
// Assumes that:
// - broadcast does not transpose dimensions (checked by hlo_verifier);
// - reshape does not mix operand and broadcast dimensions (checks);
absl::StatusOr<ReshapeParams> CalculateBroadcastOutputReshape(
    const HloBroadcastInstruction* broadcast,
    absl::Span<const int64_t> reshape_shape) {
  TF_RETURN_IF_ERROR(VerifyDefaultLayout(broadcast->shape().layout()));

  absl::Span<const int64_t> broadcast_shape = broadcast->shape().dimensions();
  llvm::SmallVector<bool> is_broadcast_dim(broadcast_shape.size(), true);
  for (const int64_t i : broadcast->dimensions()) {
    is_broadcast_dim[i] = false;
  }

  ReshapeParams result;
  result.new_shape.set_element_type(broadcast->shape().element_type());
  result.new_dims.reserve(reshape_shape.size());

  auto factors = CommonFactors(broadcast_shape, reshape_shape);
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [broadcast_from, reshape_from] = factors[i - 1];
    auto [broadcast_to, reshape_to] = factors[i];

    auto subspan = absl::MakeSpan(is_broadcast_dim)
                       .subspan(broadcast_from, broadcast_to - broadcast_from);
    auto identity = [](bool b) { return b; };
    if (absl::c_all_of(subspan, identity)) {
      continue;  // All dimensions in this group are broadcast dimensions.
    }
    if (!absl::c_none_of(subspan, identity)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot reshape broadcast for ", broadcast->ToString(),
                       " as it mixes operand and broadcast dimensions."));
    }

    // Update the new shape and broadcast dimensions.
    for (int64_t j = reshape_from; j < reshape_to; ++j) {
      result.new_shape.add_dimensions(reshape_shape[j]);
      result.new_dims.push_back(j);
    }
  }

  LayoutUtil::SetToDefaultLayout(&result.new_shape);
  return result;
}

// Returns parameters to rewrite a reshape + broadcast as broadcast + reshape.
//
// Example:
//
// b = reshape(T[reshape_shape] operand)
// c = broadcast(b)
//
// to
//
// d = T[new_shape] broadcast(operand), dimensions={new_dims}.
// c = reshape(d)
//
// Assumes that:
// - broadcast does not transpose dimensions (checked by hlo_verifier);
// - reshape does not mix operand and broadcast dimensions (checks);
absl::StatusOr<ReshapeParams> CalculateBroadcastInputReshape(
    const HloBroadcastInstruction* broadcast,
    absl::Span<const int64_t> reshape_shape) {
  TF_RETURN_IF_ERROR(VerifyDefaultLayout(broadcast->shape().layout()));

  absl::Span<const int64_t> broadcast_shape =
      broadcast->operand(0)->shape().dimensions();

  ReshapeParams result;
  result.new_shape.set_element_type(broadcast->shape().element_type());
  result.new_dims.reserve(broadcast->shape().dimensions().size() -
                          broadcast_shape.size() + reshape_shape.size());

  auto factors = CommonFactors(broadcast_shape, reshape_shape);
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [broadcast_from, reshape_from] = factors[i - 1];
    auto [broadcast_to, reshape_to] = factors[i];

    auto subspan = broadcast->dimensions().subspan(
        broadcast_from, broadcast_to - broadcast_from);

    // The dimensions of an operand group must be contiguous.
    if (subspan.back() - subspan.front() != subspan.size() - 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot reshape broadcast for ", broadcast->ToString(),
                       " as it mixes operand and broadcast dimensions."));
    }

    // Add the leading broadcast dimensions to the new shape.
    auto result_from = broadcast->dimensions()[broadcast_from];
    for (auto j = result.new_shape.dimensions().size(); j < result_from; ++j) {
      result.new_shape.add_dimensions(broadcast->shape().dimensions()[j]);
    }

    // Add the operand dimensions to the new shape.
    for (int64_t j = reshape_from; j < reshape_to; ++j) {
      result.new_shape.add_dimensions(reshape_shape[j]);
      result.new_dims.push_back(j + result_from);
    }
  }

  // Add the trailing broadcast dimensions to the new shape.
  auto result_from = std::min<int64_t>(broadcast->dimensions().back() + 1,
                                       broadcast->shape().dimensions().size());
  for (int64_t dim : broadcast->shape().dimensions().subspan(result_from)) {
    result.new_shape.add_dimensions(dim);
  }

  LayoutUtil::SetToDefaultLayout(&result.new_shape);
  return result;
}

// A helper to calculate reshape and transpose parameters when swapping the two
// operations. Both transpose_dims and the returned new_dims are the transpose
// dimensions assuming the transpose is the producer of the reshape. If the
// transpose is the consumer of the reshape, they correspond to the inverse
// permutation of the transpose dimensions.
absl::StatusOr<ReshapeParams> CalculateTransposeReshape(
    const HloTransposeInstruction* transpose,
    absl::Span<const int64_t> reshape_shape,
    absl::Span<const int64_t> transpose_shape,
    absl::Span<const int64_t> transpose_dims) {
  TF_RETURN_IF_ERROR(VerifyDefaultLayout(transpose->shape().layout()));

  auto factors = CommonFactors(transpose_shape, reshape_shape);
  struct IntermediateDims {
    int64_t input_index;   // The index of the first input dimension.
    int64_t target_index;  // The index of the intermediate dimension.
    int64_t target_dim;    // The size of the intermediate dimension.
  };
  absl::InlinedVector<IntermediateDims, 8> intermediate_shape;
  intermediate_shape.reserve(reshape_shape.size());

  // Transform adjacent factors into intermediate dimensions, holding the
  // information to construct the operand shape and permutation of the transpose
  // after the reshape.
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [transpose_from, reshape_from] = factors[i - 1];
    auto [transpose_to, reshape_to] = factors[i];
    for (int64_t j = transpose_from + 1; j < transpose_to; ++j) {
      if (transpose_dims[j - 1] + 1 != transpose_dims[j]) {
        return absl::InvalidArgumentError(
            absl::StrCat("Cannot hoist reshape across ", transpose->ToString(),
                         " because input dimensions are not contiguous."));
      }
    };
    int64_t input_index = transpose_dims[transpose_from];
    for (int64_t j = reshape_from; j < reshape_to; ++j) {
      intermediate_shape.emplace_back(IntermediateDims{
          input_index,
          static_cast<int64_t>(intermediate_shape.size()),
          reshape_shape[j],
      });
    }
  }

  // Sort by the input index, which is the order after the reshape.
  absl::c_stable_sort(intermediate_shape, [](const auto& a, const auto& b) {
    return a.input_index < b.input_index;
  });

  ReshapeParams result;
  result.new_shape.set_element_type(transpose->shape().element_type());
  result.new_dims.reserve(intermediate_shape.size());
  for (const auto& dim : intermediate_shape) {
    result.new_shape.add_dimensions(dim.target_dim);
    result.new_dims.push_back(dim.target_index);
  }

  LayoutUtil::SetToDefaultLayout(&result.new_shape);
  return result;
}

std::vector<int64_t> InversePermutation(absl::Span<const int64_t> permutation) {
  std::vector<int64_t> result(permutation.size());
  for (int64_t i = 0; i < permutation.size(); ++i) {
    result[permutation[i]] = i;
  }
  return result;
}

// Returns parameters to rewrite a transpose + reshape as reshape + transpose.
//
// Example:
//
// b = transpose(operand)
// c = T[reshape_shape] reshape(b)
//
// to
//
// d = T[new_shape] reshape(operand)
// c = T[reshape_shape] transpose(d), dimensions={new_dims}.
//
// Assumes that:
// - reshape only mixes contiguous dimensions (checks);
absl::StatusOr<ReshapeParams> CalculateTransposeOutputReshape(
    const HloTransposeInstruction* transpose,
    absl::Span<const int64_t> reshape_shape) {
  TF_ASSIGN_OR_RETURN(ReshapeParams result,
                      CalculateTransposeReshape(transpose, reshape_shape,
                                                transpose->shape().dimensions(),
                                                transpose->dimensions()));
  result.new_dims = InversePermutation(result.new_dims);
  return result;
}

// Returns parameters to rewrite a reshape + transpose as transpose + reshape.
//
// Example:
//
// b = reshape(T[reshape_shape] operand)
// c = transpose(b)
//
// to
//
// d = T[new_shape] transpose(operand), dimensions={new_dims}.
// c = reshape(d)
//
// Assumes that:
// - reshape only mixes contiguous dimensions (checks);
absl::StatusOr<ReshapeParams> CalculateTransposeInputReshape(
    const HloTransposeInstruction* transpose,
    absl::Span<const int64_t> reshape_shape) {
  return CalculateTransposeReshape(transpose, reshape_shape,
                                   transpose->operand(0)->shape().dimensions(),
                                   InversePermutation(transpose->dimensions()));
}

// Simulates a rewrite of all producers of a given bitcast or reshape, moving
// the instruction outside of the computation.
// Returns the new shapes of affected instructions in order of traversal from
// consumers to producers.
absl::StatusOr<std::vector<std::pair<HloInstruction*, Shape>>>
PlanHoistBitcastUpwardsToCallers(const HloInstruction* bitcast) {
  // Check that all producers only affect the bitcast/reshape. If there are any
  // other consumers: refuse the hoisting.
  // It is possible to support more cases by sinking the bitcast/reshape from
  // such producers downward.
  HloInstructionSetVector producers = GetProducerSet(bitcast);
  TF_RETURN_IF_ERROR(VerifyIsClosedProducerSet(producers, bitcast));
  if (bitcast->shape().element_type() !=
      bitcast->operand(0)->shape().element_type()) {
    return absl::UnimplementedError(
        absl::StrCat("Hoisting bitcast with type conversion is not supported: ",
                     bitcast->ToString()));
  }
  HloInstructionMap<Shape> to_update;

  auto set_shape = [&](const absl::Span<HloInstruction* const> instructions,
                       const Shape& shape) -> absl::Status {
    for (HloInstruction* instruction : instructions) {
      auto it = to_update.find(instruction);
      // Only update the dimensions keeping the type intact.
      Shape updated_shape(shape);
      updated_shape.set_element_type(instruction->shape().element_type());
      if (it == to_update.end()) {
        to_update.emplace(instruction, updated_shape);
      } else if (it->second != updated_shape) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Conflicting shape assignment for ", instruction->ToString(),
            " got ", ShapeUtil::HumanStringWithLayout(it->second), " and ",
            ShapeUtil::HumanStringWithLayout(shape)));
      }
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(set_shape(bitcast->operands(), bitcast->shape()));
  std::vector<std::pair<HloInstruction*, Shape>> result;
  // We want to visit instructions in order from consumers to producers: we
  // hoist the bitcast/reshape upwards and having a valid HLO at every rewrite
  // step helps a lot.
  // A simple DFS or BFS over operands will not work in non-tree situations when
  // there are multiple consumers of the same producer. Instead of writing a
  // custom traversal we can simply walk the post-order (producers before
  // consumers) list backward and only update the instructions affected.
  // TODO(b/393299275): use MakeInstructionPostOrderFrom(bitcast) - that should
  // be slightly more efficient.
  auto use_before_def = bitcast->parent()->MakeInstructionPostOrder();
  absl::c_reverse(use_before_def);
  for (HloInstruction* instruction : use_before_def) {
    auto it = to_update.find(instruction);
    if (it == to_update.end()) {
      continue;  // Not affected.
    }
    Shape& shape = it->second;
    result.emplace_back(instruction, shape);
    VLOG(2) << absl::StrCat(
        "updating the result shape of ", instruction->ToString(), " from ",
        ShapeUtil::HumanStringWithLayout(instruction->shape()), " to ",
        ShapeUtil::HumanStringWithLayout(shape));
    switch (instruction->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
        // No operands.
        break;
      case HloOpcode::kBitcast:
      case HloOpcode::kReshape:
        // Other bitcast/reshape will be hoisted separately so we don't need to
        // update its operand.
        break;
      case HloOpcode::kBroadcast: {
        TF_ASSIGN_OR_RETURN(ReshapeParams params,
                            CalculateBroadcastOutputReshape(
                                Cast<HloBroadcastInstruction>(instruction),
                                shape.dimensions()));
        TF_RETURN_IF_ERROR(
            set_shape(instruction->operands(), params.new_shape));
        break;
      }
      case HloOpcode::kTranspose: {
        TF_ASSIGN_OR_RETURN(ReshapeParams params,
                            CalculateTransposeOutputReshape(
                                Cast<HloTransposeInstruction>(instruction),
                                shape.dimensions()));
        TF_RETURN_IF_ERROR(
            set_shape(instruction->operands(), params.new_shape));
        break;
      }
      default:
        if (!instruction->IsElementwise()) {
          return absl::FailedPreconditionError(absl::StrCat(
              "Cannot hoist bitcast/reshape past ", instruction->ToString()));
        }
        TF_RETURN_IF_ERROR(set_shape(instruction->operands(), shape));
        break;
    }
  }
  return result;
}

// Simulates a rewrite of all consumers of a given bitcast/reshape, moving the
// instruction outside of the computation. Returns the new operand shapes of
// affected instructions in order of traversal from producers to consumers.
absl::StatusOr<std::vector<std::pair<HloInstruction*, Shape>>>
PlanHoistBitcastDownwardsToCallers(const HloInstruction* bitcast) {
  // Check that all consumers only affect the bitcast/reshape. If there are any
  // other producers: refuse the hoisting.
  // It is possible to support more cases by hoisting the bitcast/reshape from
  // such consumers upward.
  HloInstructionSetVector consumers = GetConsumerSet(bitcast);
  TF_RETURN_IF_ERROR(VerifyIsClosedConsumerSet(consumers, bitcast));
  if (bitcast->shape().element_type() !=
      bitcast->operand(0)->shape().element_type()) {
    return absl::UnimplementedError(
        absl::StrCat("Hoisting bitcast with type conversion is not supported: ",
                     bitcast->ToString()));
  }
  HloInstructionMap<Shape> to_update;

  auto set_shape = [&](const absl::Span<HloInstruction* const> instructions,
                       const Shape& shape) -> absl::Status {
    for (HloInstruction* instruction : instructions) {
      auto it = to_update.find(instruction);
      // Only update the dimensions keeping the type intact.
      Shape updated_shape(shape);
      updated_shape.set_element_type(instruction->shape().element_type());
      if (it == to_update.end()) {
        to_update.emplace(instruction, updated_shape);
      } else if (it->second != updated_shape) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Conflicting shape assignment for ", instruction->ToString(),
            " got ", ShapeUtil::HumanStringWithLayout(it->second), " and ",
            ShapeUtil::HumanStringWithLayout(shape)));
      }
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(set_shape(bitcast->users(), bitcast->operand(0)->shape()));
  std::vector<std::pair<HloInstruction*, Shape>> result;
  // We want to visit instructions in order from producers to consumers: we
  // hoist the bitcast/reshape downwards and having a valid HLO at every rewrite
  // step helps a lot. A simple DFS or BFS over results will not work in
  // non-tree situations when there are multiple producers of the same consumer.
  // Instead of writing a custom traversal we can simply walk the post-order
  // (producers before consumers) list and only update the instructions
  // affected.
  // TODO(b/393299275): use MakeInstructionPostOrderFrom(bitcast) - that should
  // be slightly more efficient.
  auto def_before_use = bitcast->parent()->MakeInstructionPostOrder();
  for (HloInstruction* instruction : def_before_use) {
    auto it = to_update.find(instruction);
    if (it == to_update.end()) {
      continue;  // Not affected.
    }
    Shape& shape = it->second;
    result.emplace_back(instruction, shape);
    VLOG(2) << absl::StrCat(
        "updating the operand shape of ", instruction->ToString(), " from ",
        ShapeUtil::HumanStringWithLayout(instruction->shape()), " to ",
        ShapeUtil::HumanStringWithLayout(shape));
    switch (instruction->opcode()) {
      case HloOpcode::kBitcast:
      case HloOpcode::kReshape:
        // Other bitcast/reshape will be hoisted separately so we don't need to
        // update its operand.
        break;
      case HloOpcode::kBroadcast: {
        TF_ASSIGN_OR_RETURN(ReshapeParams params,
                            CalculateBroadcastInputReshape(
                                Cast<HloBroadcastInstruction>(instruction),
                                shape.dimensions()));
        TF_RETURN_IF_ERROR(set_shape(instruction->users(), params.new_shape));
        break;
      }
      case HloOpcode::kTranspose: {
        TF_ASSIGN_OR_RETURN(ReshapeParams params,
                            CalculateTransposeInputReshape(
                                Cast<HloTransposeInstruction>(instruction),
                                shape.dimensions()));
        TF_RETURN_IF_ERROR(set_shape(instruction->users(), params.new_shape));
        break;
      }
      default:
        if (!instruction->IsElementwise()) {
          return absl::FailedPreconditionError(absl::StrCat(
              "Cannot hoist bitcast/reshape past ", instruction->ToString()));
        }
        TF_RETURN_IF_ERROR(set_shape(instruction->users(), shape));
        break;
    }
  }
  return result;
}

// Hoists the given 'bitcast' or 'reshape' upwards out of its computation, to
// the parent of each caller.
absl::Status HoistBitcastUpwardsToCallers(
    HloInstruction* bitcast, const std::vector<HloInstruction*>& callers) {
  TF_ASSIGN_OR_RETURN(auto rewrite_plan,
                      PlanHoistBitcastUpwardsToCallers(bitcast));
  for (auto [instruction, shape] : rewrite_plan) {
    VLOG(2) << absl::StrCat(
        "rewriting result shape of ", instruction->ToString(), " from ",
        ShapeUtil::HumanStringWithLayout(instruction->shape()), " to ",
        ShapeUtil::HumanStringWithLayout(shape));
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        // Create a new bitcast in callers.
        int64_t number = instruction->parameter_number();
        for (HloInstruction* caller : callers) {
          // Create a `bitcast` regardless of whether we started with a
          // `bitcast` or a `reshape`. `reshape` is strictly a special case of
          // `bitcast` anyway.
          HloInstruction* new_bitcast =
              caller->AddInstruction(HloInstruction::CreateBitcast(
                  shape, caller->mutable_operand(number)));
          TF_RETURN_IF_ERROR(
              caller->ReplaceOperandWithDifferentShape(number, new_bitcast));
        }
        break;
      }
      case HloOpcode::kBroadcast: {
        auto* broadcast = Cast<HloBroadcastInstruction>(instruction);
        auto params =
            CalculateBroadcastOutputReshape(broadcast, shape.dimensions());
        // Must be OK, already succeeded in PlanHoistBitcastUpwardsToCallers.
        QCHECK_OK(params);
        *broadcast->mutable_dimensions() = params->new_dims;
        break;
      }
      case HloOpcode::kTranspose: {
        auto* transpose = Cast<HloTransposeInstruction>(instruction);
        auto params =
            CalculateTransposeOutputReshape(transpose, shape.dimensions());
        // Must be OK, already succeeded in PlanHoistBitcastUpwardsToCallers.
        QCHECK_OK(params);
        *transpose->mutable_dimensions() = params->new_dims;
        break;
      }
      default:
        break;
    }
    *instruction->mutable_shape() = shape;
  }
  TF_RETURN_IF_ERROR(bitcast->ReplaceAllUsesWith(bitcast->mutable_operand(0)));
  TF_RETURN_IF_ERROR(bitcast->parent()->RemoveInstruction(bitcast));
  return absl::OkStatus();
}

// Hoists the given `bitcast` or `reshape` downwards out of its computation, to
// the parent of each caller.
absl::Status HoistBitcastDownwardsToCallers(
    HloInstruction* bitcast, const std::vector<HloInstruction*>& callers) {
  TF_ASSIGN_OR_RETURN(auto rewrite_plan,
                      PlanHoistBitcastDownwardsToCallers(bitcast));
  for (auto [instruction, shape] : rewrite_plan) {
    VLOG(2) << absl::StrCat(
        "rewriting operand shape of ", instruction->ToString(), " from ",
        ShapeUtil::HumanStringWithLayout(instruction->shape()), " to ",
        ShapeUtil::HumanStringWithLayout(shape));
    switch (instruction->opcode()) {
      case HloOpcode::kBroadcast: {
        auto* broadcast = Cast<HloBroadcastInstruction>(instruction);
        auto params =
            CalculateBroadcastInputReshape(broadcast, shape.dimensions());
        // Must be OK, already succeeded in PlanHoistBitcastDownwardsToCallers.
        QCHECK_OK(params);
        *broadcast->mutable_dimensions() = params->new_dims;
        shape = params->new_shape;
        break;
      }
      case HloOpcode::kTranspose: {
        auto* transpose = Cast<HloTransposeInstruction>(instruction);
        auto params =
            CalculateTransposeInputReshape(transpose, shape.dimensions());
        // Must be OK, already succeeded in PlanHoistBitcastDownwardsToCallers.
        QCHECK_OK(params);
        *transpose->mutable_dimensions() = params->new_dims;
        shape = params->new_shape;
        break;
      }
      default:
        break;
    }
    *instruction->mutable_shape() = shape;
  }

  // Insert new bitcast for each caller's result. We create a `bitcast`
  // regardless of whether we started with a `bitcast` or a a `reshape`.
  // `reshape` is strictly a special case of `bitcast` anyway.
  for (HloInstruction* caller : callers) {
    HloInstruction* new_bitcast = caller->AddInstruction(
        HloInstruction::CreateBitcast(caller->shape(), caller));
    TF_RETURN_IF_ERROR(caller->ReplaceAllUsesWith(new_bitcast));
    *caller->mutable_shape() = rewrite_plan.empty()
                                   ? bitcast->operand(0)->shape()
                                   : rewrite_plan.back().first->shape();
  }

  TF_RETURN_IF_ERROR(
      bitcast->ReplaceAllUsesWithDifferentShape(bitcast->mutable_operand(0)));
  TF_RETURN_IF_ERROR(bitcast->parent()->RemoveInstruction(bitcast));
  return absl::OkStatus();
}

// Try hoisting bitcasts and reshapes in the computation away from 'dot' to the
// callers of the computation. Some bitcasts/reshapes may remain in the
// computation, because they cannot be hoisted across all ops, e.g. across some
// transposes and broadcasts. This is not reported as an error.
absl::Status TryHoistBitcastsInComputationToCallers(HloInstruction* dot,
                                                    CallGraph* call_graph) {
  auto callers = call_graph->GetComputationCallers(dot->parent());
  for (HloInstruction* instruction : GetProducerSet(dot)) {
    if (!HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kReshape>(
            instruction)) {
      continue;
    }
    VLOG(2) << "Hoisting bitcast upwards " << instruction->ToString();
    auto status = HoistBitcastUpwardsToCallers(instruction, callers);
    if (!status.ok()) {
      VLOG(2) << "Failed to hoist bitcast or reshape upwards: " << status;
    }
  }
  for (HloInstruction* instruction : GetConsumerSet(dot)) {
    if (!HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kReshape>(
            instruction)) {
      continue;
    }
    VLOG(2) << "Hoisting bitcast downwards " << instruction->ToString();
    auto status = HoistBitcastDownwardsToCallers(instruction, callers);
    if (!status.ok()) {
      VLOG(2) << "Failed to hoist bitcast or reshape downwards: " << status;
    }
  }
  return absl::OkStatus();
}

class NestGemmFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit NestGemmFusionVisitor(
      mlir::MLIRContext* ctx, CallGraph* call_graph,
      const se::GpuComputeCapability compute_capability)
      : ctx_(ctx),
        call_graph_(call_graph),
        compute_capability_(compute_capability) {}

  absl::Status HandleFusion(HloInstruction* instruction) override {
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instruction);

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
    DCHECK_EQ(GetDotCount(computation), 1) << "Fusion has more than one dot.";
    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
    TF_RETURN_IF_ERROR(
        TryHoistBitcastsInComputationToCallers(instr, call_graph_));
    VLOG(2) << "After hoisting bitcasts and reshapes: "
            << computation->ToString();

    TF_RETURN_IF_ERROR(
        MakeNestedFusionFromGemmFusion(fusion, config.value(), dot, ctx_));

    this->MarkAsChanged();
    // TODO(b/393299275): support checks should be run *before* the fusion is
    // constructed and this pass should only be applied to the known supported
    // HLO. Currently though, we are at mercy of what GemmFusion pass thinks
    // legacy emitter can handle. We change the kind of the fusion here and
    // switch the track. Thus it is on us to make sure that the generic
    // emitter will be able to handle the result. That is an early check to
    // make sure that that nesting did not produce an unsupported HLO.
    CodegenDecision can_codegen_computation =
        IsTritonSupportedComputation(*computation, compute_capability_);
    if (!can_codegen_computation) {
      return absl::InternalError(absl::StrCat(
          "Computation of fusion ", fusion->ToString(),
          " is not supported by Triton: ", can_codegen_computation.Explain()));
    }
    return absl::OkStatus();
  }

 private:
  mlir::MLIRContext* ctx_;
  CallGraph* call_graph_;
  const se::GpuComputeCapability compute_capability_;
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
    NestGemmFusionVisitor visitor(&ctx, call_graph.get(), compute_capability_);
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
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
  auto output_tile_sizes = get_tile_sizes(out_rank);
  std::sort(output_tile_sizes.begin(), output_tile_sizes.end());

  const TilingSpecification& tiling_specification =
      analysis.GetTilingSpecification();

  do {
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
      continue;
    }
    TF_ASSIGN_OR_RETURN(FlatTiling flat_tiling_parameters,
                        tiling.Flatten(tiling_specification));
    auto mapped_dot_tile_sizes =
        EvaluateTileSizes(tiled_dot.symbolic_tile(), flat_tiling_parameters);
    if (mapped_dot_tile_sizes == expected_dot_tile_sizes) {
      BlockLevelParameters params;
      params.output_tile_sizes = {std::vector<int64_t>(
          output_tile_sizes.begin(), output_tile_sizes.end())};
      params.num_warps = config.num_warps;
      params.num_ctas = config.num_ctas;
      params.num_stages = config.num_stages;
      return params;
    }
  } while (std::next_permutation(output_tile_sizes.begin(),
                                 output_tile_sizes.end()));

  return absl::InternalError(absl::StrCat(
      "Couldn't find output tile sizes that satisfy ", tiled_dot.ToString()));
}

}  // namespace detail
}  // namespace xla::gpu

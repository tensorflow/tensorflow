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

#include "xla/backends/gpu/transforms/nest_gemm_fusion.h"

#include <cstdint>
#include <memory>
#include <utility>
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
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/transforms/convert_triton_gemm_config.h"
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
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

using ::mlir::MLIRContext;

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

  ASSIGN_OR_RETURN(HloInstruction * fusion, FuseInstructionsFromRoot(root));

  ASSIGN_OR_RETURN(auto gpu_config, fusion->backend_config<GpuBackendConfig>());
  gpu_config.mutable_fusion_backend_config()->set_kind(
      kTritonNestedGemmFusionKind);
  RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  for (int64_t operand_index : consumer.OperandIndices(&root)) {
    RETURN_IF_ERROR(consumer.ReplaceOperandWith(operand_index, fusion));
  }

  return absl::OkStatus();
}

absl::Status IsDot(const HloInstruction& dot) {
  if (dot.opcode() != HloOpcode::kDot &&
      dot.opcode() != HloOpcode::kScaledDot) {
    return absl::InternalError(
        absl::StrCat("Expected a dot instruction but got ", dot.ToString()));
  }
  return absl::OkStatus();
}

// Annotates the given nested fusion with the given tile sizes.
// Implementation for AnnotateDotLhs/RhsNestedFusion().
absl::Status AnnotateDotOperandNestedFusionImpl(
    HloFusionInstruction& nested_fusion, const HloInstruction& dot,
    const TritonGemmConfig& config,
    absl::Span<const int64_t> contracting_dimensions,  // Must be single element
    absl::Span<const int64_t> batch_dimensions, int64_t contracting_dim_size,
    int64_t non_contracting_dim_size) {
  RETURN_IF_ERROR(IsDot(dot));
  if (contracting_dimensions.size() != 1) {
    return absl::InternalError(
        absl::StrCat("Expected a single lhs contracting dimension but got ",
                     contracting_dimensions.size()));
  }

  ASSIGN_OR_RETURN(
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
  block_level_parameters.is_warp_specialization_allowed =
      config.is_warp_specialization_allowed;

  ASSIGN_OR_RETURN(auto gpu_config,
                   nested_fusion.backend_config<GpuBackendConfig>());
  *gpu_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  RETURN_IF_ERROR(nested_fusion.set_backend_config(gpu_config));

  return absl::OkStatus();
}

absl::Status AnnotateDotLhsNestedFusion(HloFusionInstruction& nested_fusion,
                                        const HloInstruction& dot,
                                        const TritonGemmConfig& config) {
  RETURN_IF_ERROR(IsDot(dot));
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  return AnnotateDotOperandNestedFusionImpl(
      nested_fusion, dot, config,
      dimension_numbers.lhs_contracting_dimensions(),
      dimension_numbers.lhs_batch_dimensions(), config.block_k, config.block_m);
}

absl::Status AnnotateDotRhsNestedFusion(HloFusionInstruction& nested_fusion,
                                        const HloInstruction& dot,
                                        const TritonGemmConfig& config) {
  RETURN_IF_ERROR(IsDot(dot));
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  return AnnotateDotOperandNestedFusionImpl(
      nested_fusion, dot, config,
      dimension_numbers.rhs_contracting_dimensions(),
      dimension_numbers.rhs_batch_dimensions(), config.block_k, config.block_n);
}

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

// Constructs nested fusion nodes for the operands of `concatenate` instructions
// and annotates them with `kTritonNestedGemmFusionKind`.
absl::Status FuseAndAnnotateConcatOperands(HloComputation* computation) {
  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() != HloOpcode::kConcatenate) {
      continue;
    }
    for (HloInstruction* operand : instr->mutable_operands()) {
      RETURN_IF_ERROR(FuseInstructionsForConsumer(*operand, *instr));
    }
  }
  return absl::OkStatus();
}

// Transforms a fusion into an equivalent nested fusion if it has a single dot.
// Returns ok if the transformation was successful.
absl::Status MakeNestedFusionFromGemmFusion(
    HloFusionInstruction* fusion, HloInstruction* dot, MLIRContext* ctx,
    const se::DeviceDescription& device_description) {
  RETURN_IF_ERROR(IsDot(*dot));
  const bool is_scaled_dot = dot->opcode() == HloOpcode::kScaledDot;
  constexpr int lhs = 0;
  constexpr int rhs = 1;
  ASSIGN_OR_RETURN(TritonGemmConfig config, GetTritonGemmConfig(*fusion));
  HloComputation* computation = fusion->called_computation();

  // First, create nested fusions for the operands of `concatenate` instructions
  // if they exist.
  RETURN_IF_ERROR(FuseAndAnnotateConcatOperands(computation));

  // Left-hand side of the dot.
  RETURN_IF_ERROR(
      FuseInstructionsForConsumer(*dot->mutable_operand(lhs), *dot));
  RETURN_IF_ERROR(AnnotateDotLhsNestedFusion(
      *::xla::Cast<HloFusionInstruction>(dot->mutable_operand(lhs)), *dot,
      config));

  // Right-hand side of the dot.
  RETURN_IF_ERROR(
      FuseInstructionsForConsumer(*dot->mutable_operand(rhs), *dot));
  RETURN_IF_ERROR(AnnotateDotRhsNestedFusion(
      *::xla::Cast<HloFusionInstruction>(dot->mutable_operand(rhs)), *dot,
      config));

  if (is_scaled_dot) {
    constexpr int kLhsScale = 2;
    constexpr int kRhsScale = 3;
    constexpr int kContractingScaleFactor = 32;
    auto scale_config = config;
    scale_config.block_k /= kContractingScaleFactor;
    RETURN_IF_ERROR(
        FuseInstructionsForConsumer(*dot->mutable_operand(kLhsScale), *dot));
    RETURN_IF_ERROR(AnnotateDotLhsNestedFusion(
        *::xla::Cast<HloFusionInstruction>(dot->mutable_operand(kLhsScale)),
        *dot, scale_config));
    RETURN_IF_ERROR(
        FuseInstructionsForConsumer(*dot->mutable_operand(kRhsScale), *dot));
    RETURN_IF_ERROR(AnnotateDotRhsNestedFusion(
        *::xla::Cast<HloFusionInstruction>(dot->mutable_operand(kRhsScale)),
        *dot, scale_config));
  }
  // Delete newly unused instructions, if any.
  ASSIGN_OR_RETURN([[maybe_unused]] bool changed,
                   HloDCE::RunOnComputation(
                       computation,
                       /*remove_cross_partition_collective_ops=*/false));

  // Annotate the fusion itself.
  ASSIGN_OR_RETURN(auto gpu_config, fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.clear_triton_gemm_config();
  backend_config.set_kind(kTritonNestedGemmFusionKind);

  ASSIGN_OR_RETURN(
      BlockLevelParameters block_level_parameters,
      FindBlockLevelParameters(dot, config, ctx, device_description));

  *backend_config.mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  return absl::OkStatus();
}

class NestGemmFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit NestGemmFusionVisitor(
      MLIRContext* mlir_context, CallGraph* call_graph,
      const se::DeviceDescription& device_description)
      : mlir_context_(mlir_context),
        call_graph_(call_graph),
        device_description_(device_description) {}

 private:
  absl::Status AcceptNestedInstruction(const HloInstruction* instruction) {
    if (instruction->IsElementwise()) {
      return absl::OkStatus();
    }
    const DebugOptions& debug_options =
        instruction->GetModule()->config().debug_options();
    if (instruction->opcode() == HloOpcode::kScaledDot &&
        !debug_options.xla_gpu_experimental_scaled_dot_with_triton()) {
      return absl::InternalError("Scaled dot with Triton is not enabled.");
    }

    if (instruction->opcode() == HloOpcode::kFusion) {
      return AcceptResultingFusion(Cast<HloFusionInstruction>(instruction));
    }

    return absl::OkStatus();
  }

  // Checks whether all operations are from the "tested" set that we confirmed
  // to not cause regressions.
  // That enables a progressive rollout of the new emitter. Eventually we should
  // remove this check completely as all computations will be supported by the
  // generic emitter and performance regressions will be addressed.
  absl::Status AcceptResultingFusion(const HloFusionInstruction* fusion) {
    const HloComputation* computation = fusion->called_computation();
    for (const HloInstruction* instruction : computation->instructions()) {
      RETURN_IF_ERROR(AcceptNestedInstruction(instruction));
    }
    return absl::OkStatus();
  }

  absl::Status RewriteFusion(HloFusionInstruction* fusion,
                             CallGraph* call_graph) {
    HloComputation* computation = fusion->called_computation();
    HloInstruction* instr =
        hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
    if (instr == nullptr) {
      instr = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                       HloOpcode::kScaledDot);
      if (instr == nullptr) {
        return absl::InternalError(absl::StrCat("Computation of fusion ",
                                                fusion->ToString(),
                                                " has no dot instruction"));
      }
    }

    RETURN_IF_ERROR(MakeNestedFusionFromGemmFusion(fusion, instr, mlir_context_,
                                                   device_description_));

    MarkAsChanged();
    bool scaled_dot_enabled =
        fusion->GetModule()
            ->config()
            .debug_options()
            .xla_gpu_experimental_scaled_dot_with_triton();
    if (CodegenDecision can_codegen_computation = IsTritonSupportedComputation(
            *fusion->called_computation(),
            device_description_.gpu_compute_capability());
        !scaled_dot_enabled && !can_codegen_computation) {
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
      instr = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                       HloOpcode::kScaledDot);
      if (instr == nullptr) {
        VLOG(2) << "Skipping fusion as it has no dot instruction";
        return absl::OkStatus();
      }
    }
    return RewriteFusion(fusion, call_graph_);
  }

 private:
  MLIRContext* mlir_context_;
  CallGraph* call_graph_;
  const se::DeviceDescription& device_description_;
};

}  // namespace

absl::StatusOr<bool> NestGemmFusion::RunOnModule(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  auto call_graph = CallGraph::Build(module, execution_threads);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    NestGemmFusionVisitor visitor(mlir_context_, call_graph.get(),
                                  device_description_);
    RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

absl::StatusOr<bool> NestGemmFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->config()
          .debug_options()
          .xla_gpu_unsupported_disable_nested_gemm_fusions()) {
    return false;
  }
  return RunOnModule(module, execution_threads);
}

}  // namespace xla::gpu

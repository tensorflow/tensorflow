/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/softmax_rewriter_triton.h"

#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/fusions/triton/triton_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/gpu/transforms/reduction_dimension_grouper.h"
#include "xla/service/gpu/transforms/reduction_splitter.h"
#include "xla/service/gpu/transforms/tree_reduction_rewriter.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using hlo_query::IsBroadcastOfParameter;
using hlo_query::IsBroadcastOfScalarConstant;

bool HasDefaultLayout(const Shape& shape) {
  return shape.has_layout() &&
         LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

inline bool HasOneUse(const HloInstruction* instr) {
  return instr->user_count() == 1;
}

// Chooses which operand to use for fusion processing. Taking in a unary or
// binary instruction, returns the first non-splat operand. If none is
// present, returns any operand.
HloInstruction* ChooseOperandForFusionProcessing(HloInstruction* instr) {
  CHECK_GT(instr->operand_count(), 0);
  CHECK_LE(instr->operand_count(), 2);

  if (instr->operand_count() > 1 &&
      (IsBroadcastOfScalarConstant(*instr->operand(0)) ||
       IsBroadcastOfParameter(*instr->operand(0)))) {
    return instr->mutable_operand(1);
  }
  return instr->mutable_operand(0);
}

bool IsTriviallyFusible(HloInstruction* instr,
                        const se::GpuComputeCapability& gpu_version,
                        int num_allowed_users = 1) {
  // Checks whether an op is trivially fusible. An op is said to be trivially
  // fusible if it does not increase the amount of memory read/written by the
  // resulting fusion, is compatible with any chosen tiling, and can be
  // codegen'd using Triton. The op is allowed to have up to num_allowed_users
  // users.
  if (instr->user_count() > num_allowed_users ||
      !HasDefaultLayout(instr->shape())) {
    return false;
  }

  if (HloPredicateIsOp<HloOpcode::kBitcast>(instr)) {
    return true;
  }

  if (instr->IsElementwise() && instr->operand_count() == 1) {
    return static_cast<bool>(IsTritonSupportedInstruction(*instr, gpu_version));
  }

  // Elementwise binary ops are trivially fusible if the operands are the same,
  // or if exactly one of the operands is a splat constant.
  if (instr->IsElementwiseBinary()) {
    const HloInstruction* operand_0 = instr->operand(0);
    const HloInstruction* operand_1 = instr->operand(1);

    // Elementwise binary ops should be fused if both operands are the same and
    // if the operand is triton supported.
    if (operand_0 == operand_1) {
      return static_cast<bool>(
          IsTritonSupportedInstruction(*instr, gpu_version));
    }

    // For simplicity we only fuse elementwise binary ops with splat operands
    // if they contain one non-splat operand.
    if ((IsBroadcastOfScalarConstant(*operand_0) ||
         IsBroadcastOfParameter(*operand_0)) ^
        (IsBroadcastOfScalarConstant(*operand_1) ||
         IsBroadcastOfParameter(*operand_1))) {
      return static_cast<bool>(
          IsTritonSupportedInstruction(*instr, gpu_version));
    }
  }

  return false;
}

// Returns true if a trivially connected producer of 'consumer' with opcode
// 'opcode' exists. If such an instruction is found, the value of 'producer' is
// set to it. The definition of "trivial" operations is as given in
// 'IsTriviallyFusible'.
bool TrivialEdge(HloInstruction** producer, HloInstruction* consumer,
                 HloOpcode opcode,
                 const se::GpuComputeCapability& gpu_version) {
  while (consumer->opcode() != opcode) {
    if (IsTriviallyFusible(consumer, gpu_version)) {
      consumer = ChooseOperandForFusionProcessing(consumer);
    } else {
      return false;
    }
  }

  *producer = consumer;
  return true;
}

bool IsTriviallyConnectedProducerOf(
    HloInstruction* producer, HloInstruction* consumer,
    const se::GpuComputeCapability& gpu_version) {
  if (producer == consumer) {
    return true;
  }

  HloInstruction* found_producer = consumer;
  while (
      TrivialEdge(&found_producer, consumer, producer->opcode(), gpu_version)) {
    if (found_producer == producer) {
      return true;
    }

    if (!IsTriviallyFusible(found_producer, gpu_version)) {
      return false;
    }

    consumer = found_producer->mutable_operand(0);
  }

  return false;
}

// Creates a fusion corresponding to the input diamond. The resulting
// fusion instruction is added to the module, but is not yet inserted into the
// graph as a replacement of the original instructions.
//
// TODO(b/347956491): this awkward abstraction is needed to work around
// limitations of HloFusionAdaptor, which underpins the implementation of
// SymbolicTileAnalysis. We need to come up with a better solution.
absl::StatusOr<HloFusionInstruction*> MakeFusionForDiamond(
    const DiamondDescriptor& diamond) {
  auto [root, producer] = diamond;

  std::string suggested_name = "triton_softmax";
  HloComputation::Builder builder(absl::StrCat(suggested_name, "_computation"));
  // Original instruction -> fused one.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;

  int param = 0;
  old_to_new_mapping[producer] =
      builder.AddInstruction(HloInstruction::CreateParameter(
          param, producer->shape(), absl::StrCat("parameter_", param)));
  param++;

  std::vector<HloInstruction*> parameters = {producer};

  std::function<void(HloInstruction*)> create_computation =
      [&](HloInstruction* instr) -> void {
    if (old_to_new_mapping.contains(instr)) {
      return;
    }
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : instr->mutable_operands()) {
      create_computation(operand);
      new_operands.push_back(old_to_new_mapping[operand]);
    }
    if (HloPredicateIsOp<HloOpcode::kParameter>(instr)) {
      old_to_new_mapping[instr] =
          builder.AddInstruction(HloInstruction::CreateParameter(
              param, instr->shape(), absl::StrCat("parameter_", param)));
      parameters.push_back(instr);
      param++;
    } else {
      old_to_new_mapping[instr] = builder.AddInstruction(
          instr->CloneWithNewOperands(instr->shape(), new_operands));
    }
  };
  create_computation(root);

  HloComputation* computation =
      root->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                           /*is_entry=*/false);

  HloInstruction* normalization_fusion =
      root->parent()->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), HloInstruction::FusionKind::kCustom, parameters,
          computation));

  normalization_fusion->GetModule()->SetAndUniquifyInstrName(
      normalization_fusion, "triton_softmax");
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      normalization_fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kTritonFusionKind));
  TF_RETURN_IF_ERROR(normalization_fusion->set_backend_config(gpu_config));
  return xla::Cast<HloFusionInstruction>(normalization_fusion);
}

// Runs an HLO pipeline to convert the `module` to the stage as it would look
// like in the main XLA:GPU compilation pipeline if the normalization diamond
// were not fused by SoftmaxRewriterTriton.
//
// Before the FusionPipeline, this function runs passes that are relevant to the
// instructions in the normalization diamond and are placed between
// SoftmaxRewriterTriton and PriorityFusion in the main compilation pipeline:
// passes that rewrite and split reductions.
absl::Status RunFusionPipeline(
    HloModule* module, const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size) {
  HloPassPipeline reduction_pipeline("reduction_pipeline");
  // Passes that run after SoftmaxRewriterTriton and before PriorityFusion and
  // transform reductions.
  reduction_pipeline.AddPass<ReductionDimensionGrouper>();
  reduction_pipeline.AddPass<HloPassFix<ReductionSplitter>>(
      device_info,
      /*ignore_small_reduce_dims=*/false);
  reduction_pipeline.AddPass<HloPassFix<TreeReductionRewriter>>(device_info);

  TF_RETURN_IF_ERROR(reduction_pipeline.Run(module).status());

  return FusionPipeline(module->config().debug_options(), shape_size,
                        /*thread_pool=*/nullptr, device_info)
      .Run(module)
      .status();
}

// Returns a run time estimate for instructions in the `fusion` if they were
// fused without SoftmaxRewriterTriton.
//
// This can help us understand how effective `ReductionSplitter` and
// `PriorityFusion` are for this fusion.
//
// In the bigger module, the instructions in the normalization diamond will be
// fused with other instructions around it, so it's not an exact estimate, but
// should be a good enough approximation.
absl::StatusOr<absl::Duration>
EstimateOptimizedHloRunTimeWithoutSoftMaxRewriterTriton(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size) {
  auto new_module = ExtractComputationIntoNewModule(
      *fusion->fused_instructions_computation());

  // After this call, the `new_module` will have instruction fused without
  // SoftmaxRewriterTriton.
  TF_RETURN_IF_ERROR(
      RunFusionPipeline(new_module.get(), device_info, shape_size));

  VLOG(3) << "priority fusion module: " << new_module->ToString();

  HloComputation* entry_computation = new_module->entry_computation();
  GpuHloCostAnalysis::Options cost_analysis_options{
      shape_size,
      /*per_second_rates=*/{},
      /*min_latencies_seconds=*/{},
      /*count_multiple_input_accesses=*/true};
  GpuHloCostAnalysis cost_analysis(cost_analysis_options, device_info);
  TF_RETURN_IF_ERROR(entry_computation->Accept(&cost_analysis));

  absl::Duration total_run_time = absl::ZeroDuration();

  for (const HloInstruction* instr : entry_computation->instructions()) {
    total_run_time += GpuPerformanceModel::EstimateRunTimeForInstruction(
                          instr, device_info, &cost_analysis,
                          GpuPerformanceModelOptions::Default())
                          .exec_time;
  }

  return total_run_time;
}

// Returns an empty `FusionDecision` if the normalization diamond should be
// fused together. In that case, also chooses and tests the best block level
// parameters. Otherwise, returns a `FusionDecision` with an explanation why the
// normalization diamond should not be fused.
//
// If `use_cost_model_to_evaluate_fusions` is set to `true`, the function will
// also use the Cost Model to estimate the run time of the fused and unfused
// versions of the normalization diamond. If the fused version is slower,
// returns a `FusionDecision` to indicate that the function should not happen.
absl::StatusOr<FusionDecision>
DecideIfShouldFuseAndMaybeSetBlockLevelParameters(
    HloFusionInstruction* normalization_fusion,
    GpuPerformanceModelWithIndexingAnalysis& indexing_performance_model,
    const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size,
    bool use_cost_model_to_evaluate_fusions) {
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(normalization_fusion);

  TF_ASSIGN_OR_RETURN(
      TiledRunTimeDataOrError tiled_runtime_data_or,
      indexing_performance_model.TryFindBestTilingForFusion(*fusion_adaptor));

  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&tiled_runtime_data_or)) {
    return FusionDecision::Forbid(absl::StrCat("SymbolicTileAnalysis failed: ",
                                               fusion_decision->Explain()));
  }

  TiledRunTimeData tiled_runtime_data =
      std::get<TiledRunTimeData>(std::move(tiled_runtime_data_or));

  if (use_cost_model_to_evaluate_fusions) {
    TF_ASSIGN_OR_RETURN(absl::Duration run_time_without_softmax_rewriter,
                        EstimateOptimizedHloRunTimeWithoutSoftMaxRewriterTriton(
                            normalization_fusion, device_info, shape_size));

    VLOG(2) << "run time estimate if normalization diamond fused together: "
            << tiled_runtime_data.runtime_data.exec_time;
    VLOG(2)
        << "run time estimate if normalization diamond is not fused together: "
        << run_time_without_softmax_rewriter;

    if (run_time_without_softmax_rewriter <
        tiled_runtime_data.runtime_data.exec_time) {
      return FusionDecision::Forbid(
          "Run time estimate for without applying the custom normalization "
          "rewrite is faster.");
    }
  }

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      normalization_fusion->backend_config<GpuBackendConfig>());
  *backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      tiled_runtime_data.block_level_parameters.ToBlockLevelFusionConfig();
  TF_RETURN_IF_ERROR(normalization_fusion->set_backend_config(backend_config));
  VLOG(2) << "Fusing with backend config: " << backend_config.DebugString();

  return FusionDecision::Allow();
}

absl::StatusOr<bool> MaybeFuseDiamondImpl(
    const DiamondDescriptor& diamond,
    GpuPerformanceModelWithIndexingAnalysis& indexing_performance_model,
    const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size,
    bool use_cost_model_to_evaluate_fusions) {
  TF_ASSIGN_OR_RETURN(HloFusionInstruction * normalization_fusion,
                      MakeFusionForDiamond(diamond));
  HloInstruction* root = diamond.root;

  VLOG(2) << "MaybeFuseDiamondImpl: " << normalization_fusion->ToString();

  TF_ASSIGN_OR_RETURN(
      FusionDecision fusion_decision,
      DecideIfShouldFuseAndMaybeSetBlockLevelParameters(
          normalization_fusion, indexing_performance_model, device_info,
          shape_size, use_cost_model_to_evaluate_fusions));

  if (!fusion_decision.CanFuse()) {
    VLOG(2) << "Not fusing: " << fusion_decision.Explain();
    normalization_fusion->DetachFromOperandsAndUsers();
    TF_RETURN_IF_ERROR(normalization_fusion->parent()->RemoveInstruction(
        normalization_fusion));
    return false;
  }

  if (root->IsRoot()) {
    root->parent()->set_root_instruction(normalization_fusion);
    TF_RETURN_IF_ERROR(
        root->parent()->RemoveInstructionAndUnusedOperands(root));
  } else {
    TF_RETURN_IF_ERROR(
        root->parent()->ReplaceInstruction(root, normalization_fusion));
  }
  return true;
}

// Returns `true` if the diamond passed as a parameter can be tiled correctly
// using `SymbolicTileAnalysis`.
absl::StatusOr<bool> CanSymbolicTileAnalysisTileDiamond(
    const DiamondDescriptor& diamond,
    const se::DeviceDescription& device_info) {
  TF_ASSIGN_OR_RETURN(HloFusionInstruction * normalization_fusion,
                      MakeFusionForDiamond(diamond));
  mlir::MLIRContext context;
  SymbolicTileAnalysisOrError symbolic_tile_analysis_or_error =
      SymbolicTileAnalysis::AnalyzeComputation(
          *normalization_fusion->called_computation(), &context,
          TritonEmitterConstraints::GetBuilder(device_info));

  bool can_tile = std::holds_alternative<SymbolicTileAnalysis>(
      symbolic_tile_analysis_or_error);

  TF_RETURN_IF_ERROR(diamond.root->GetModule()->RemoveEmbeddedComputation(
      normalization_fusion->called_computation()));
  TF_RETURN_IF_ERROR(
      diamond.root->parent()->RemoveInstruction(normalization_fusion));

  return can_tile;
}

FusionDecision ShouldFuseReduction(const HloInstruction& reduce,
                                   const se::GpuComputeCapability& cc) {
  if (CodegenDecision is_supported = IsTritonSupportedInstruction(reduce, cc);
      !is_supported) {
    return FusionDecision::Forbid(is_supported.Explain());
  }

  if (reduce.dimensions().size() != 1 ||
      reduce.dimensions(0) != reduce.operand(0)->shape().rank() - 1) {
    return FusionDecision::Forbid(
        "The reductions in the diamond must reduce 1 dimension and that "
        "dimension must be the last dimension of the operand.");
  }

  // Ensure that the reduction's identity is either a constant or a supported
  // convert of a constant.
  const HloInstruction* identity = reduce.operand(1);
  bool should_fuse_identity =
      HloPredicateIsOp<HloOpcode::kConstant>(identity) ||
      (HloPredicateIsOp<HloOpcode::kConvert>(identity) &&
       identity->operand(0)->opcode() == HloOpcode::kConstant &&
       IsTritonSupportedInstruction(*identity, cc));
  if (!should_fuse_identity) {
    return FusionDecision::Forbid(
        "Reduction identity is not a constant or a supported convert of a "
        "constant.");
  }

  return FusionDecision::Allow();
}

DiamondMatchingDecision MatchesTritonCompatibleClosedReductionDiamondImpl(
    HloInstruction* instr, const se::GpuComputeCapability& cc) {
  if (!instr->IsElementwiseBinary()) {
    return FusionDecision::Forbid("Root is not elementwise binary.");
  }

  if (!IsTritonSupportedInstruction(*instr, cc)) {
    return FusionDecision::Forbid(
        "Root is not supported for Triton instruction.");
  }

  HloInstruction* producer;
  HloInstruction* broadcast;
  HloInstruction* reduce;

  if (!TrivialEdge(&broadcast, instr->mutable_operand(1), HloOpcode::kBroadcast,
                   cc)) {
    return FusionDecision::Forbid(
        "Could not find a trivial connection from root to a broadcast.");
  }

  if (!TrivialEdge(&reduce, broadcast->mutable_operand(0), HloOpcode::kReduce,
                   cc)) {
    return FusionDecision::Forbid(
        "Could not find a trivial connection from matched broadcast to a "
        "reduction.");
  }

  if (!(HasDefaultLayout(broadcast->shape()) &&
        HasDefaultLayout(reduce->shape()))) {
    return FusionDecision::Forbid(
        "Broadcast or reduce have non-default layouts.");
  }

  if (FusionDecision should_fuse_reduction = ShouldFuseReduction(*reduce, cc);
      !should_fuse_reduction) {
    VLOG(2) << should_fuse_reduction.Explain();
    return should_fuse_reduction;
  }

  // Ensure that the reduction's identity is either a constant or a supported
  // convert of a constant.
  const HloInstruction* identity = reduce->operand(1);
  bool should_fuse_identity =
      HloPredicateIsOp<HloOpcode::kConstant>(identity) ||
      (HloPredicateIsOp<HloOpcode::kConvert>(identity) &&
       identity->operand(0)->opcode() == HloOpcode::kConstant &&
       IsTritonSupportedInstruction(*identity, cc));
  if (!should_fuse_identity) {
    return FusionDecision::Forbid(
        "Reduction identity is not a constant or a supported convert of a "
        "constant.");
  }

  if (!HasOneUse(broadcast) || !HasOneUse(reduce)) {
    return FusionDecision::Forbid("More than one use of broadcast or reduce.");
  }

  producer = reduce->mutable_operand(0);

  if (absl::c_linear_search(broadcast->dimensions(),
                            broadcast->shape().rank() - 1)) {
    return FusionDecision::Forbid(
        "Broadcast is not along the reduction dimension.");
  }

  while (IsTriviallyFusible(producer, cc)) {
    producer = ChooseOperandForFusionProcessing(producer);
  }

  if (!HasDefaultLayout(producer->shape())) {
    return FusionDecision::Forbid("Producer has non-default layout.");
  }

  if (!IsTriviallyConnectedProducerOf(producer, instr->mutable_operand(0),
                                      cc)) {
    return FusionDecision::Forbid("Producer is not trivially connected.");
  }

  if (producer != instr->operand(0) && instr->operand(0)->user_count() != 1) {
    return FusionDecision::Forbid("Unsupported root-producer connection.");
  }

  VLOG(2) << "Matched Softmax diamond with: ";
  VLOG(2) << "root: " << instr->ToString();
  VLOG(2) << "producer: " << producer->ToString();
  VLOG(2) << "broadcast: " << broadcast->ToString();
  VLOG(2) << "reduce: " << reduce->ToString();

  return producer;
}

}  // anonymous namespace

DiamondMatchingDecision
SoftmaxRewriterTriton::MatchesTritonCompatibleClosedReductionDiamond(
    HloInstruction* instr) const {
  return MatchesTritonCompatibleClosedReductionDiamondImpl(
      instr, device_info_.gpu_compute_capability());
}

absl::StatusOr<std::vector<DiamondDescriptor>>
SoftmaxRewriterTriton::FindAllFusibleNormalizationDiamonds(
    HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  const se::GpuComputeCapability& cc = device_info_.gpu_compute_capability();
  std::vector<DiamondDescriptor> matched_diamonds;

  for (HloComputation* comp :
       module.MakeNonfusionComputations(execution_threads)) {
    if (comp->IsCustomCallComputation()) {
      continue;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      auto producer =
          MatchesTritonCompatibleClosedReductionDiamondImpl(instr, cc);
      if (std::holds_alternative<HloInstruction*>(producer)) {
        DiamondDescriptor diamond{
            /*root=*/instr, /*producer=*/std::get<HloInstruction*>(producer)};
        // We filter out the diamonds that cannot be tiled correctly using
        // `SymbolicTileAnalysis`.
        TF_ASSIGN_OR_RETURN(
            bool can_tile_diamond,
            CanSymbolicTileAnalysisTileDiamond(diamond, device_info_));
        if (can_tile_diamond) {
          matched_diamonds.push_back(diamond);
        } else {
          VLOG(2) << "Cannot tile the diamond pattern described by "
                  << "instructions " << instr->ToString() << " and "
                  << std::get<HloInstruction*>(producer)->ToString() << ".";
          continue;
        }

      } else {
        VLOG(2) << "Cannot match the diamond pattern for instruction "
                << instr->ToString()
                << ". Reason: " << std::get<FusionDecision>(producer).Explain();
      }
    }
  }

  return matched_diamonds;
}

absl::StatusOr<bool> SoftmaxRewriterTriton::MaybeFuseNormalizationDiamond(
    const DiamondDescriptor& diamond) {
  HloFusionAnalysisCache fusion_analysis_cache(device_info_);
  GpuPerformanceModelWithIndexingAnalysis indexing_performance_model(
      &device_info_, &fusion_analysis_cache, shape_size_, &mlir_context_);

  return MaybeFuseDiamondImpl(diamond, indexing_performance_model, device_info_,
                              shape_size_, use_cost_model_to_evaluate_fusions_);
}

absl::StatusOr<bool> SoftmaxRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(EnsureTritonSupportsComputeCapability(
      device_info_.gpu_compute_capability()));

  TF_ASSIGN_OR_RETURN(
      std::vector<DiamondDescriptor> diamonds,
      FindAllFusibleNormalizationDiamonds(*module, execution_threads));

  bool changed = false;
  // The diamonds must be emitted in reverse order, to make sure that producer
  // instructions are emitted correctly when the root of diamond n is exactly
  // the producer of diamond n+1.
  for (auto diamond = diamonds.rbegin(); diamond != diamonds.rend();
       ++diamond) {
    TF_ASSIGN_OR_RETURN(bool fused, MaybeFuseNormalizationDiamond(*diamond));
    changed |= fused;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla

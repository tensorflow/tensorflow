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

#include <cstdint>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/MathExtras.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/cost_model/fusion_analysis_cache.h"
#include "xla/backends/gpu/cost_model/gpu_indexing_performance_model.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

// Pattern-matches slow loop fusions that can likely be handled better by
// Triton than by other emitters.
// TODO(b/370690811,b/372187266): generalize this to other slow transposes.
bool ShouldRewriteLoopTransposeFusion(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_description) {
  const HloInstruction* root =
      fusion->fused_instructions_computation()->root_instruction();

  bool is_loop_transpose_fusion =
      fusion->fusion_kind() == HloInstruction::FusionKind::kLoop &&
      root->opcode() == HloOpcode::kTranspose;

  if (!is_loop_transpose_fusion) {
    return false;
  }

  // The slow transposes are those when the minormost dimension in the input
  // is neither the minormost nor the second minormost dimension in the output,
  // and the output minormost dimension is swapped with the new minormost
  // dimension.

  // We use the normalized logical transpose shape so it should be enough to
  // check that the minormost dimension's index within the result is smaller
  // than rank - 2, and that the new minormost dimension is swapped with it.
  absl::InlinedVector<int64_t, 3> permutation;
  auto normalized_dims_or = ShapeUtil::GetNormalizedLogicalTransposeShape(
      root->operand(0)->shape(), root->shape(), root->dimensions(),
      permutation);
  if (!normalized_dims_or.ok()) {
    return false;
  }
  auto normalized_dims = normalized_dims_or.value();
  int64_t rank = normalized_dims.size();

  // This only triggers for transposes with major-to-minor layout.
  bool has_major_to_minor_layout =
      LayoutUtil::IsMonotonicWithDim0Major(root->shape().layout());
  int64_t result_minormost_dim_in_operand = permutation.back();

  if (!(has_major_to_minor_layout &&
        permutation[result_minormost_dim_in_operand] == rank - 1 &&
        permutation[rank - 1] < rank - 2)) {
    return false;
  }

  // Because of Triton's power-of-two restriction, we're only guaranteed to
  // handle the bitcast case when the bitcast's minor dimension is a power of
  // two. This ensures that we can tile it reasonably even if the bitcast's
  // input has that dimension collapsed. (See comments in `symbolic_tile.cc`
  // around destructuring summations to understand why this is important.)
  auto can_bitcast_input_be_tiled_efficiently =
      [](const HloInstruction* bitcast) {
        return llvm::isPowerOf2_64(bitcast->shape().dimensions_minor(0));
      };

  bool is_pure_transpose = ::xla::Match(root, m::Transpose(m::Parameter()));
  bool is_bitcasted_transpose_with_power_of_two_minor_dim = ::xla::Match(
      root,
      m::Transpose(m::Bitcast(m::Parameter())
                       .WithPredicate(can_bitcast_input_be_tiled_efficiently)));
  return is_pure_transpose ||
         is_bitcasted_transpose_with_power_of_two_minor_dim;
}

// Pattern matches reduction fusions that can likely be handled better by
// Triton than by other emitters.
// At present we try to match closely for s32 dots that have been rewritten as
// reductions.
bool ShouldRewriteReductionFusion(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_description) {
  if (fusion->IsMultiOutputFusion()) {
    return false;
  }
  const HloInstruction* root =
      fusion->fused_instructions_computation()->root_instruction();
  if (const bool is_reduce_fusion = root->opcode() == HloOpcode::kReduce;
      !is_reduce_fusion) {
    return false;
  }
  // All inputs are s32.
  for (const auto* operand : root->operands()) {
    if (operand->shape().element_type() != S32) {
      return false;
    }
  }
  if (const bool is_output_s32 = root->shape().element_type() == S32;
      !is_output_s32) {
    return false;
  }
  return true;
}

absl::StatusOr<bool> ShouldTryRewriteFusion(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_description) {
  if (fusion->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_enable_fusion_block_level_rewriter()) {
    return true;
  }

  // TODO(b/370690811): ShouldRewriteLoopTransposeFusion rewrite may no longer
  // be necessary once MLIR emitters transposes are faster.
  return ShouldRewriteLoopTransposeFusion(fusion, device_description) ||
         ShouldRewriteReductionFusion(fusion, device_description);
}

absl::StatusOr<bool> ProcessFusionInstruction(
    HloFusionInstruction* fusion_instruction,
    const se::DeviceDescription& device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size,
    mlir::MLIRContext* mlir_context) {
  TF_ASSIGN_OR_RETURN(bool should_try_rewrite,
                      ShouldTryRewriteFusion(fusion_instruction, device_info));
  if (!should_try_rewrite) {
    VLOG(2) << "Not rewriting fusion " << fusion_instruction->ToString()
            << " because it is not supported.";
    return false;
  }

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
      &device_info, &fusion_analysis_cache, shape_size, mlir_context);

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
                 tiled_runtime_data.block_level_parameters.output_tile_sizes[0],
                 ", "),
             "], num_warps=",
             tiled_runtime_data.block_level_parameters.num_warps)
      << " for fusion computation " << fusion_computation->ToString();

  *backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      tiled_runtime_data.block_level_parameters.ToBlockLevelFusionConfig();
  backend_config.mutable_fusion_backend_config()->set_kind(kTritonFusionKind);
  TF_RETURN_IF_ERROR(fusion_instruction->set_backend_config(backend_config));
  fusion_instruction->set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return true;
}

}  // anonymous namespace

absl::StatusOr<bool> FusionBlockLevelRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(EnsureTritonSupportsComputeCapability(
      device_info_.gpu_compute_capability()));

  bool has_changed = false;

  for (HloComputation* computation :
       module->MakeComputationSorted(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }
    HloFusionInstruction* fusion_instruction =
        ::xla::Cast<HloFusionInstruction>(computation->FusionInstruction());
    TF_ASSIGN_OR_RETURN(
        bool changed, ProcessFusionInstruction(fusion_instruction, device_info_,
                                               shape_size_, mlir_context_));

    has_changed |= changed;
  }

  return has_changed;
}

}  // namespace gpu
}  // namespace xla

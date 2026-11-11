/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/gpu_default_backend_policy.h"

#include <cstdint>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "llvm/Support/MathExtras.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/codegen/tiling/tiling_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

// Pattern-matches slow loop fusions that can likely be handled better by
// Triton than by other emitters.
// TODO(b/370690811,b/372187266): generalize this to other slow transposes.
bool IsSlowLoopTransposeFusion(const HloFusionInstruction& fusion) {
  const HloInstruction* root = fusion.fused_expression_root();

  bool is_loop_transpose_fusion =
      fusion.fusion_kind() == HloInstruction::FusionKind::kLoop &&
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
  // This fails for transposes without major-to-minor layout.
  absl::StatusOr<absl::InlinedVector<int64_t, 3>> normalized_dims =
      ShapeUtil::GetNormalizedLogicalTransposeShape(
          root->operand(0)->shape(), root->shape(), root->dimensions(),
          permutation);
  if (!normalized_dims.ok()) {
    return false;
  }
  int64_t rank = normalized_dims->size();
  int64_t result_minormost_dim_in_operand = permutation.back();
  if (permutation[result_minormost_dim_in_operand] != rank - 1 ||
      permutation[rank - 1] >= rank - 2) {
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

  bool is_pure_transpose = Match(root, m::Transpose(m::Parameter()));
  bool is_bitcasted_transpose_with_power_of_two_minor_dim = Match(
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
bool IsS32ReductionFusion(const HloFusionInstruction& fusion) {
  const HloInstruction* root = fusion.fused_expression_root();
  if (root->opcode() != HloOpcode::kReduce) {
    return false;
  }
  return absl::c_all_of(root->operands(),
                        [](const HloInstruction* operand) {
                          return operand->shape().element_type() == S32;
                        }) &&
         root->shape().element_type() == S32;
}

// Matches fusions whose root is the first output of a scan.
bool IsScanFusion(const HloFusionInstruction& fusion) {
  const HloInstruction* root = fusion.fused_expression_root();
  return root->opcode() == HloOpcode::kGetTupleElement &&
         root->operand(0)->opcode() == HloOpcode::kScan &&
         root->tuple_index() == 0;
}

bool ShouldDefaultToBlockLevelEmitter(const HloFusionInstruction& fusion) {
  // Custom fusions other than generic Triton fusions are assigned to other
  // backends.
  if (fusion.fusion_kind() == HloInstruction::FusionKind::kCustom &&
      !IsGenericTritonFusion(fusion)) {
    return false;
  }
  // If a dot fusion can be handled by Triton, GemmRewriter would have already
  // taken care of it.
  if (absl::c_any_of(
          fusion.fused_instructions(),
          HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kScaledDot>)) {
    return false;
  }

  const DebugOptions& debug_options =
      fusion.GetModule()->config().debug_options();
  if (IsSameShapeMultiOutputFusion(fusion,
                                   Shape::Equal().IgnoreElementType()) &&
      debug_options
          .xla_gpu_experimental_enable_same_shape_multi_output_fusion()) {
    return true;
  }
  // TODO(b/370690811): IsSlowLoopTransposeFusion may no longer be necessary
  // once MLIR emitters transposes are faster.
  return IsSlowLoopTransposeFusion(fusion) || IsS32ReductionFusion(fusion) ||
         IsScanFusion(fusion);
}

}  // namespace

std::optional<autotuner::Backend> GetDefaultBackend(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return std::nullopt;
  }
  if (ShouldDefaultToBlockLevelEmitter(*Cast<HloFusionInstruction>(&instr))) {
    return autotuner::Backend::BLOCK_LEVEL_EMITTER;
  }
  return std::nullopt;
}

}  // namespace xla::gpu

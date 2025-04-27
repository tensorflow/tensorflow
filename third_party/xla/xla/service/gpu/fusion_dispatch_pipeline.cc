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

#include "xla/service/gpu/fusion_dispatch_pipeline.h"

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/MathExtras.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/transforms/fusion_block_level_rewriter.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

bool IsSlowLoopTransposeFusion(const HloFusionInstruction* fusion) {
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
  int64_t rank = root->shape().dimensions().size();

  // The transpose dimension grouper has run, so it should be enough to check
  // that the minormost dimension's index within the result is smaller than
  // rank - 2, and that the new minormost dimension is swapped with it.
  // This only triggers for transposes with major-to-minor layout.
  bool has_major_to_minor_layout =
      LayoutUtil::IsMonotonicWithDim0Major(root->shape().layout());
  absl::Span<int64_t const> transpose_dimensions = root->dimensions();
  int64_t result_minormost_dim_in_operand = transpose_dimensions.back();

  return has_major_to_minor_layout &&
         transpose_dimensions[result_minormost_dim_in_operand] == rank - 1 &&
         transpose_dimensions[rank - 1] < rank - 2;
}

// Pattern-matches slow loop fusions that can likely be handled better by
// Triton than by other emitters.
// TODO(b/370690811,b/372187266): generalize this to other slow transposes.
bool FusionWillBeHandledBetterByTriton(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_description) {
  if (!IsSlowLoopTransposeFusion(fusion)) {
    return false;
  }

  const HloInstruction* root =
      fusion->fused_instructions_computation()->root_instruction();

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

}  // anonymous namespace

HloPassPipeline FusionDispatchPipeline(
    const se::DeviceDescription& device_description,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  std::function<absl::StatusOr<bool>(const HloFusionInstruction*)>
      try_rewrite_fusion_if =
          [&device_description](
              const HloFusionInstruction* fusion) -> absl::StatusOr<bool> {
    bool should_always_rewrite_to_block_level =
        fusion->GetModule()
            ->config()
            .debug_options()
            .xla_gpu_experimental_enable_fusion_block_level_rewriter();

    // TODO(b/370690811): this rewrite may no longer be necessary once MLIR
    // emitters transposes are faster.
    return should_always_rewrite_to_block_level ||
           FusionWillBeHandledBetterByTriton(fusion, device_description);
  };

  // Even though this is a single pass, we need to create a pipeline in order
  // to make sure the pass's run is recorded in the `HloModuleMetadata`.
  HloPassPipeline pipeline("fusion-dispatch-pipeline");
  pipeline.AddPass<FusionBlockLevelRewriter>(device_description, shape_size_fn,
                                             std::move(try_rewrite_fusion_if));
  return pipeline;
}

}  // namespace gpu
}  // namespace xla

/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/coalescing_analysis.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_analysis.h"

namespace xla {
namespace gpu {

using mlir::AffineMap;

bool IsReadCoalescedHeuristic(
    const std::optional<HloFusionAnalysis>& fusion_analysis,
    const HloInstruction* producer, const HloInstruction* consumer) {
  auto analyzed_kind_or_reduction =
      fusion_analysis ? fusion_analysis->GetEmitterFusionKind()
                      : HloFusionAnalysis::EmitterFusionKind::kReduction;

  // Transposing minor dimension breaks coalescing.
  if (analyzed_kind_or_reduction !=
      HloFusionAnalysis::EmitterFusionKind::kTranspose) {
    auto is_broadcast = [&](const HloInstruction* instr) {
      while (true) {
        if (instr->opcode() == HloOpcode::kBroadcast) return true;
        if (instr->operand_count() != 1) return false;
        if (instr->opcode() != HloOpcode::kBitcast && !instr->IsElementwise()) {
          return false;
        }
        instr = instr->operand(0);
      }
    };

    auto is_bad_transpose = [&](const HloInstruction* instr) {
      if (instr->opcode() == HloOpcode::kFusion) {
        for (auto* instr : instr->fused_instructions()) {
          // Hack: we allow transposes of broadcasts.
          if (TransposesMinorDimension(instr) &&
              !is_broadcast(instr->operand(0))) {
            return true;
          }
        }
        return false;
      }
      return TransposesMinorDimension(instr);
    };

    if (is_bad_transpose(producer)) return false;
    if (consumer && is_bad_transpose(consumer)) return false;
  }

  // Fusing two row reductions breaks coalescing.
  if (analyzed_kind_or_reduction ==
          HloFusionAnalysis::EmitterFusionKind::kReduction &&
      IsInputFusibleReduction(*producer) && consumer &&
      IsInputFusibleReduction(*consumer)) {
    return false;
  }

  return true;
}

namespace {

// Converts a layout to a dimensions transposition necessary to get to that
// layout from identity.
std::vector<int64_t> ToTransposeDimensions(const Layout& l) {
  std::vector<int64_t> out(l.minor_to_major().begin(),
                           l.minor_to_major().end());
  absl::c_reverse(out);
  return out;
}

}  // namespace

bool IsReadCoalesced(const HloInstruction* operand, const HloInstruction* instr,
                     const absl::flat_hash_map<const HloInstruction*,
                                               IndexingMapSet>& indexing_maps,
                     mlir::MLIRContext* mlir_context) {
  const Shape& output_shape = instr->shape();
  const Shape& operand_shape = operand->shape();

  AffineMap output_transpose = ComputeTransposeIndexingMap(
      ToTransposeDimensions(output_shape.layout()), mlir_context);
  AffineMap operand_transpose = ComputeTransposeIndexingMap(
      InversePermutation(ToTransposeDimensions(operand_shape.layout())),
      mlir_context);

  bool is_coalesced = true;
  for (const auto& indexing_map : indexing_maps.at(operand)) {
    if (!indexing_map.has_value()) return false;

    AffineMap normalized_indexing_map = operand_transpose.compose(
        indexing_map->affine_map.compose(output_transpose));
    // First version is naive, we just check that the affine maps of input and
    // output have the same minor dimension.
    is_coalesced &= normalized_indexing_map.isMinorIdentityWithBroadcasting();
  }
  return is_coalesced;
}

}  // namespace gpu
}  // namespace xla

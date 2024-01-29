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

#include "xla/service/gpu/model/coalescing_analysis.h"

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

bool IsReadCoalescedHeuristic(const HloFusionAnalysis& fusion_analysis,
                              const HloInstruction* producer,
                              const HloInstruction* consumer) {
  auto fusion_kind = fusion_analysis.GetEmitterFusionKind();

  // Transposing minor dimension breaks coalescing.
  if (fusion_kind != HloFusionAnalysis::EmitterFusionKind::kTranspose) {
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
  if (fusion_kind == HloFusionAnalysis::EmitterFusionKind::kReduction &&
      IsInputFusibleReduction(*producer) && consumer &&
      IsInputFusibleReduction(*consumer)) {
    return false;
  }

  return true;
}

bool IsReadCoalesced(const HloInstruction* operand, const HloInstruction* instr,
                     const absl::flat_hash_map<const HloInstruction*,
                                               IndexingMapSet>& indexing_maps,
                     mlir::MLIRContext* mlir_context) {
  bool is_coalesced = true;
  const Shape& output_shape = instr->shape();
  const Shape& operand_shape = operand->shape();
  auto output_physical_to_logical_map =
      GetIndexingMapFromPhysicalLayoutToLogical(output_shape, mlir_context);
  auto input_logical_to_physical_map =
      GetIndexingMapFromLogicalToPhysicalLayout(operand_shape, mlir_context);
  for (const auto& indexing_map : indexing_maps.at(operand)) {
    if (!indexing_map.has_value()) return false;

    auto normalized_indexing_map = indexing_map;
    if (output_physical_to_logical_map.has_value()) {
      normalized_indexing_map = ComposeIndexingMaps(
          normalized_indexing_map, output_physical_to_logical_map);
    }
    if (input_logical_to_physical_map.has_value()) {
      normalized_indexing_map = ComposeIndexingMaps(
          input_logical_to_physical_map, normalized_indexing_map);
    }
    // First version is naive, we just check that the affine maps of input and
    // output have the same minor dimension.
    is_coalesced &=
        normalized_indexing_map->affine_map.isMinorIdentityWithBroadcasting();
  }
  return is_coalesced;
}

}  // namespace gpu
}  // namespace xla

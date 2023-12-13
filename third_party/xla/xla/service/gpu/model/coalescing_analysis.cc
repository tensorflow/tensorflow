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

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla {
namespace gpu {

bool IsReadCoalesced(const std::optional<HloFusionAnalysis>& fusion_analysis,
                     const HloInstruction* producer,
                     const HloInstruction* consumer) {
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

}  // namespace gpu
}  // namespace xla

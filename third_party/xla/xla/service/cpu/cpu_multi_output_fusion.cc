/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/cpu/cpu_multi_output_fusion.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/multi_output_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::cpu {

// Huristic to limit the number of operands to avoid excessive compile time.
static constexpr int64_t kMaxOperands = 20;

static bool CheckInsideFusion(const HloInstruction* instr) {
  const HloComputation* parent = instr->parent();
  for (const auto* caller_instr : parent->caller_instructions()) {
    if (CheckInsideFusion(caller_instr)) {
      return true;
    }
  }

  if (parent->IsFusionComputation()) {
    return true;
  }
  return false;
}

static bool IsSupportedOp(HloOpcode opcode) {
  return emitters::IsSupportedElementalOp(opcode) &&
         opcode != HloOpcode::kDot && opcode != HloOpcode::kDynamicUpdateSlice;
}

bool CpuMultiOutputFusion::ShapesCompatibleForFusion(HloInstruction* instr1,
                                                     HloInstruction* instr2) {
  auto get_element_shape = [&](HloInstruction* instr) {
    const HloInstruction* element_instr = instr;
    if (instr->opcode() == HloOpcode::kFusion) {
      auto* fused_expression_root = instr->fused_expression_root();
      if (instr->IsMultiOutputFusion()) {
        // The shapes in all tuple operands should agree. Just pick the first
        // one.
        element_instr = fused_expression_root->operands()[0];
      } else {
        element_instr = fused_expression_root;
      }
    }
    return element_instr->shape();
  };

  // The elementwise output shapes must be the same (including layout)
  return ShapeUtil::ShapeUtil::Equal(get_element_shape(instr1),
                                     get_element_shape(instr2));
}

bool CpuMultiOutputFusion::IsFusible(HloInstruction* instr) {
  if (CheckInsideFusion(instr)) {
    return false;
  }

  // Constants to the extent they can be are fused in input<->output fusion.
  if (instr->opcode() == HloOpcode::kConstant) {
    return false;
  }

  if (instr->IsLoopFusion()) {
    return IsSupportedOp(instr->fused_expression_root()->opcode());
  }

  return IsSupportedOp(instr->opcode());
}

bool CpuMultiOutputFusion::LegalToFuse(HloInstruction* instr1,
                                       HloInstruction* instr2) {
  if (instr1->operand_count() >= kMaxOperands ||
      instr2->operand_count() >= kMaxOperands) {
    return false;
  }

  return MultiOutputFusion::LegalToFuse(instr1, instr2);
}

int64_t CpuMultiOutputFusion::GetProfit(HloInstruction* instr1,
                                        HloInstruction* instr2) {
  absl::flat_hash_set<HloInstruction*> candidate_operands;
  for (HloInstruction* operand : instr1->operands()) {
    if (IsProfitableOperand(operand)) {
      candidate_operands.insert(operand);
    }
  }

  int64_t profit = 0;
  for (HloInstruction* operand : instr2->operands()) {
    if (candidate_operands.contains(operand)) {
      profit += ShapeUtil::ByteSizeOf(operand->shape(), sizeof(void*));
    }
  }

  return profit;
}

}  // namespace xla::cpu

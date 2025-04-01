/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/change_op_data_type.h"

#include <optional>

#include "xla/service/hlo_creation_utils.h"
#if defined(INTEL_MKL)
#include "xla/service/cpu/onednn_contraction_rewriter.h"
#endif  // INTEL_MKL

namespace xla {
namespace {
std::optional<PrimitiveType> GetUniformOperandType(
    const HloInstruction* instr) {
  std::optional<PrimitiveType> type;
  for (const HloInstruction* operand : instr->operands()) {
    if (!type.has_value()) {
      type = operand->shape().element_type();
    } else if (operand->shape().element_type() != type.value()) {
      return std::nullopt;
    }
  }
  return type;
}
}  // namespace

absl::StatusOr<bool> ChangeOpDataType::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  HloCloner default_cloner = [](const HloInstruction* inst, const Shape& shape,
                                absl::Span<HloInstruction* const> operands) {
    return inst->CloneWithNewOperands(shape, operands);
  };
  HloCloner cloner = cloner_ ? cloner_ : default_cloner;

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      std::optional<PrimitiveType> operand_type = GetUniformOperandType(instr);
      if (!op_matcher_(instr) || !operand_type.has_value() ||
          !instr->shape().IsArray() ||
          instr->opcode() == HloOpcode::kParameter) {
        continue;
      }
      const PrimitiveType from_type = *operand_type;
      auto it = to_type_map_.find(from_type);
      if (it == to_type_map_.end()) {
        continue;
      }
#if defined(INTEL_MKL)
      if (cpu::OneDnnContractionRewriter::ShouldRewriteInstr(instr, true)) {
        continue;
      }
#endif  // INTEL_MKL
      const PrimitiveType to_type = it->second;
      absl::InlinedVector<HloInstruction*, 8> new_operands;
      for (HloInstruction* operand : instr->mutable_operands()) {
        new_operands.push_back(MakeConvertToHlo(operand, to_type));
      }

      Shape new_shape = instr->shape();
      new_shape.set_element_type(to_type);

      HloInstruction* new_instr =
          comp->AddInstruction(cloner(instr, new_shape, new_operands));
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(
          instr, MakeConvertToHlo(new_instr, from_type)));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/change_op_data_type.h"

#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"

namespace xla {

StatusOr<bool> ChangeOpDataType::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (!op_matcher_(instr) ||
          !absl::c_all_of(instr->operands(),
                          [&](const HloInstruction* operand) {
                            return operand->shape().element_type() == from_ty_;
                          }) ||
          !instr->shape().IsArray() ||
          instr->opcode() == HloOpcode::kParameter) {
        continue;
      }
      absl::InlinedVector<HloInstruction*, 8> new_operands;
      for (HloInstruction* operand : instr->mutable_operands()) {
        new_operands.push_back(MakeConvertToHlo(operand, to_ty_));
      }

      Shape new_shape = instr->shape();
      new_shape.set_element_type(to_ty_);

      HloInstruction* new_instr = comp->AddInstruction(
          instr->CloneWithNewOperands(new_shape, new_operands));
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(
          instr, MakeConvertToHlo(new_instr, from_ty_)));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla

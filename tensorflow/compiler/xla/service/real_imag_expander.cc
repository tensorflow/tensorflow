/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/real_imag_expander.h"

#include "tensorflow/compiler/xla/literal_util.h"

namespace xla {

bool RealImagExpander::InstructionMatchesPattern(HloInstruction* inst) {
  return (inst->opcode() == HloOpcode::kReal ||
          inst->opcode() == HloOpcode::kImag) &&
         !ShapeUtil::ElementIsComplex(inst->operand(0)->shape());
}

StatusOr<HloInstruction*> RealImagExpander::ExpandInstruction(
    HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kReal) {
    // real with a non-complex input is just a copy.
    return inst->mutable_operand(0);
  } else {
    // Imag with a non-complex input is just a 0. Construct this 0 using
    // scalar 0 of the element type and an appropriate number of broadcasts.
    HloComputation* comp = inst->parent();
    auto zero = comp->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(inst->operand(0)->shape().element_type())));
    zero = comp->AddInstruction(
        HloInstruction::CreateBroadcast(inst->shape(), zero, {}));
    return zero;
  }
}

}  // namespace xla

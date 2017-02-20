/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_query.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace hlo_query {

bool IsConstantR0F32(HloInstruction* instruction, float* out) {
  if (instruction->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsScalarF32(instruction->shape())) {
    *out = LiteralUtil::Get<float>(instruction->literal(), {});
    return true;
  }

  return false;
}

bool AllOperandsAreParameters(const HloInstruction& instruction) {
  for (const auto& operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter) {
      return false;
    }
  }
  return true;
}

HloInstruction* GetMatchingOperand(
    std::function<bool(const HloInstruction*)> matcher,
    HloInstruction* instruction) {
  for (HloInstruction* op : instruction->operands()) {
    if (matcher(op)) {
      return op;
    }
  }
  return nullptr;
}

bool MatchBinaryInstructionOperand(
    std::function<bool(const HloInstruction*)> matcher,
    HloInstruction* instruction, HloInstruction** matching_operand,
    HloInstruction** other_operand) {
  CHECK_EQ(instruction->operand_count(), 2);
  if (matcher(instruction->operand(0))) {
    *matching_operand = instruction->mutable_operand(0);
    *other_operand = instruction->mutable_operand(1);
    return true;
  }
  if (matcher(instruction->operand(1))) {
    *matching_operand = instruction->mutable_operand(1);
    *other_operand = instruction->mutable_operand(0);
    return true;
  }
  return false;
}

bool MatchBinaryInstructionOperandOpcode(HloOpcode opcode,
                                         HloInstruction* instruction,
                                         HloInstruction** matching_operand,
                                         HloInstruction** other_operand) {
  return MatchBinaryInstructionOperand(
      [opcode](const HloInstruction* instruction) {
        return instruction->opcode() == opcode;
      },
      instruction, matching_operand, other_operand);
}

bool IsScalarConstant(const HloInstruction* instruction) {
  return instruction->IsConstant() && ShapeUtil::IsScalar(instruction->shape());
}

}  // namespace hlo_query
}  // namespace xla

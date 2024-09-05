/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/logistic_expander.h"

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {

bool LogisticExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kLogistic;
}

absl::StatusOr<HloInstruction*> LogisticExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloInstruction* operand = instruction->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  // Computing 1.0 / (1.0 - exp(-x))
  HloInstruction* one_constant = MakeScalarLike(operand, 1.0f);
  HloInstruction* exp_instr =
      MakeUnaryHlo(HloOpcode::kExp,
                   MakeUnaryHlo(HloOpcode::kNegate, operand).value())
          .value();
  HloInstruction* denominator =
      MakeBinaryHlo(HloOpcode::kAdd, one_constant, exp_instr).value();
  return MakeBinaryHlo(HloOpcode::kDivide, one_constant, denominator).value();
}

}  // namespace xla

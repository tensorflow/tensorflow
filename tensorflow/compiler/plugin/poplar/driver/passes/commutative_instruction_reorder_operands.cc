/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/commutative_instruction_reorder_operands.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include <map>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

CommutativeInstructionReorderOperands::CommutativeInstructionReorderOperands() {
}

static bool IsReshaping(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kAddDependency) {
    inst = inst->operand(0);
  }
  switch (inst->opcode()) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kPad:
      return true;
    default:
      return false;
  }
}

static bool IsElementwiseBinaryCommutative(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
      return true;
    default:
      return false;
  }
}

StatusOr<bool> CommutativeInstructionReorderOperands::Run(HloModule* module) {
  bool changed = false;
  for (auto* comp : module->computations()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (IsElementwiseBinaryCommutative(inst) &&
          IsReshaping(inst->operand(0)) && !IsReshaping(inst->operand(1))) {
        auto* op0 = inst->mutable_operand(0);
        auto* op1 = inst->mutable_operand(1);
        inst->ReplaceOperandWith(0, op1);
        inst->ReplaceOperandWith(1, op0);
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla

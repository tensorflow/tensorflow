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

#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

StatusOr<bool> HloConstantFolding::Run(HloModule* module) {
  bool changed = false;
  for (auto& computation : module->computations()) {
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // Skip dead code.
      if (instruction->user_count() == 0 &&
          computation->root_instruction() != instruction) {
        continue;
      }
      // Depending on the opcode, choose how to handle constant operands.
      //
      // TODO(b/35975797): Fold constant computations for more than reshapes and
      // transposes.
      switch (instruction->opcode()) {
        case HloOpcode::kReshape: {
          if (instruction->operand(0)->opcode() == HloOpcode::kConstant) {
            TF_ASSIGN_OR_RETURN(
                auto reshaped_literal,
                LiteralUtil::Reshape(
                    instruction->operand(0)->literal(),
                    AsInt64Slice(instruction->shape().dimensions())));
            TF_CHECK_OK(computation->ReplaceWithNewInstruction(
                instruction,
                HloInstruction::CreateConstant(std::move(reshaped_literal))));
            changed = true;
          }
          break;
        }
        case HloOpcode::kTranspose: {
          if (instruction->operand(0)->opcode() == HloOpcode::kConstant) {
            auto transposed_literal = LiteralUtil::Transpose(
                instruction->operand(0)->literal(), instruction->dimensions());
            TF_CHECK_OK(computation->ReplaceWithNewInstruction(
                instruction,
                HloInstruction::CreateConstant(std::move(transposed_literal))));
            changed = true;
          }
          break;
        }
        default:
          break;
      }
    }
  }
  return changed;
}

}  // namespace xla

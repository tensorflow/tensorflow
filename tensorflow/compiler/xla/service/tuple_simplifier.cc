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

#include "tensorflow/compiler/xla/service/tuple_simplifier.h"

#include <queue>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

TupleSimplifier::TupleSimplifier(bool exclude_entry_computation)
    : exclude_entry_computation_(exclude_entry_computation) {}

StatusOr<bool> TupleSimplifier::Run(HloModule* module) {
  // Initially add all GTE and Tuple instructions to the worklist.
  bool changed = false;
  for (auto* computation : module->computations()) {
    if (exclude_entry_computation_ &&
        computation == module->entry_computation()) {
      continue;
    }
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kTuple) {
        // Collapse the following structure into just 'Tuple-shaped Op':
        //
        //   Tuple-shaped Op
        //         |
        //   +-----+-----+
        //   |     |     |
        //  GTE   GTE   GTE
        //   |     |     |
        //   +-----+-----+
        //         |
        //       Tuple
        //
        HloInstruction* top_tuple = nullptr;
        bool can_simplify = true;
        for (int64 operand_number = 0;
             operand_number < instruction->operand_count(); ++operand_number) {
          HloInstruction* operand =
              instruction->mutable_operand(operand_number);
          if (operand->opcode() != HloOpcode::kGetTupleElement ||
              operand->tuple_index() != operand_number) {
            can_simplify = false;
            break;
          }
          if (top_tuple == nullptr) {
            top_tuple = operand->mutable_operand(0);
            if (!ShapeUtil::Compatible(top_tuple->shape(),
                                       instruction->shape())) {
              can_simplify = false;
              break;
            }
          } else if (top_tuple != operand->operand(0)) {
            can_simplify = false;
            break;
          }
        }
        if (can_simplify && top_tuple != nullptr) {
          changed = true;
          TF_RETURN_IF_ERROR(
              computation->ReplaceInstruction(instruction, top_tuple));
        }
      } else {
        auto ancestor = instruction->LatestNonGteAncestorAndIndex();
        if (ancestor.first == instruction) {
          continue;
        }
        // If possible replace a chain of GTE with the operation which produces
        // the element. For example, replace uses of GTE with below with just
        // 'Op' (assuming 'Op' is at the index of the GTE instruction):
        //
        //     ...  Op ...
        //       \  |   /
        //        Tuple
        //          |
        //         GTE
        //         ...
        //          |
        //         GTE
        //          |
        //         GTE
        if (ShapeUtil::Compatible(ancestor.first->shape(),
                                  instruction->shape())) {
          changed = true;
          TF_RETURN_IF_ERROR(
              computation->ReplaceInstruction(instruction, ancestor.first));
        } else if (ancestor.first->opcode() == HloOpcode::kTuple) {
          changed = true;
          TF_RETURN_IF_ERROR(computation->ReplaceInstruction(
              instruction,
              ancestor.first->mutable_operand(ancestor.second[0])));
        }
      }
    }
  }
  return changed;
}

}  // namespace xla

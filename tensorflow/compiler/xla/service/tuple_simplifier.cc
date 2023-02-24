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

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"

namespace xla {

TupleSimplifier::TupleSimplifier(bool exclude_entry_computation)
    : exclude_entry_computation_(exclude_entry_computation) {}

StatusOr<bool> TupleSimplifier::RemoveWholeTuple(HloInstruction* tuple) {
  HloInstruction* top_tuple = nullptr;
  for (int64_t operand_number = 0; operand_number < tuple->operand_count();
       ++operand_number) {
    HloInstruction* operand = tuple->mutable_operand(operand_number);
    if (operand->opcode() != HloOpcode::kGetTupleElement ||
        operand->tuple_index() != operand_number) {
      return false;
    }
    if (top_tuple == nullptr) {
      top_tuple = operand->mutable_operand(0);
      if (!ShapeUtil::Compatible(top_tuple->shape(), tuple->shape())) {
        return false;
      }
    } else if (top_tuple != operand->operand(0)) {
      return false;
    }
  }
  if (top_tuple == nullptr) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(bool changed,
                      tuple->parent()->ReplaceInstruction(
                          tuple, top_tuple, /*preserve_sharding=*/true));
  return changed;
}

StatusOr<bool> TupleSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Initially add all GTE and Tuple instructions to the worklist.
  bool changed = false;
  for (auto* computation : module->computations(execution_threads)) {
    if (exclude_entry_computation_ &&
        computation == module->entry_computation()) {
      continue;
    }
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kTuple) {
        TF_ASSIGN_OR_RETURN(bool c, RemoveWholeTuple(instruction));
        changed |= c;
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
        //
        // Note that this deletes the Tuple instruction altogether. In addition,
        // if only a subset of tuple's elements are used, this transform
        // optimizes them one at a time, and after the last use is optimized,
        // the Tuple will also be deleted.
        HloInstruction* replacement = ancestor.first;
        for (int i = 0; i < ancestor.second.size(); ++i) {
          if (replacement->opcode() != HloOpcode::kTuple) {
            replacement = nullptr;
            break;
          }
          replacement = replacement->mutable_operand(ancestor.second[i]);
        }

        if (replacement) {
          TF_ASSIGN_OR_RETURN(bool replaced, computation->ReplaceInstruction(
                                                 instruction, replacement,
                                                 /*preserve_sharding=*/true));
          changed |= replaced;
        }
      }
    }
  }
  return changed;
}

}  // namespace xla

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace xla {

// Replaces all uses of old_instr with new_instr except the use at
// `while_body_root` (which must be a tuple instruction) at index `tuple_index`.
// This utility helps us replace an instruction in the while body with a
// constant while still keeping it trivially loop invariant.
static Status ReplaceUsesWhileKeepingLoopInvariance(
    HloInstruction* old_instr, HloInstruction* new_instr,
    HloInstruction* while_body_root, int64 tuple_index) {
  CHECK_EQ(while_body_root->opcode(), HloOpcode::kTuple);

  std::vector<HloInstruction*> users;
  users.reserve(old_instr->user_count());
  c_copy(old_instr->users(), std::back_inserter(users));

  for (auto* user : users) {
    for (int64 i = 0, e = user->operand_count(); i < e; i++) {
      if (user->operand(i) == old_instr &&
          !(user == while_body_root && i == tuple_index)) {
        TF_RETURN_IF_ERROR(user->ReplaceOperandWith(i, new_instr));
      }
    }
  }

  return Status::OK();
}

StatusOr<bool> WhileLoopConstantSinking::TrySinkingConstantsIntoWhileBody(
    HloInstruction* while_instr) {
  HloComputation* while_body = while_instr->while_body();

  const HloInstruction& init_value = *while_instr->operand(0);
  if (init_value.opcode() != HloOpcode::kTuple) {
    return false;
  }

  bool changed = false;

  for (HloInstruction* invariant_gte :
       WhileUtil::GetInvariantGTEsForWhileBody(*while_body)) {
    int64 index = invariant_gte->tuple_index();
    const HloInstruction& invariant_value = *init_value.operand(index);
    if (invariant_value.opcode() == HloOpcode::kConstant) {
      auto* constant_instr =
          while_body->AddInstruction(invariant_value.Clone(/*suffix=*/".sunk"));
      TF_RETURN_IF_ERROR(ReplaceUsesWhileKeepingLoopInvariance(
          invariant_gte, constant_instr, while_body->root_instruction(),
          index));
      changed = true;
    }
  }

  return changed;
}

StatusOr<bool> WhileLoopConstantSinking::Run(HloModule* module) {
  VLOG(2) << "HLO module before WhileLoopConstantSinking:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeNonfusionComputations()) {
    // Right now we don't particulary care about optimizing while-of-while
    // patterns.  If/When we do, we'll want to visit the outer while (while_0)
    // before we visit the inner while (while_1):
    //
    // while_1_body(state) {
    //   val = gte(state, 0) // Loop invariant
    //   use(val)
    // }
    //
    // while_0_body(state) {
    //   val = gte(state, 0) // Loop invariant
    //   while_1 = while(init=tuple(val, ...), body=while_1_body, ...)
    //   ...
    // }
    //
    // main {
    //   while_0 = while(init=(constant, ...), body=while_0_body, ...)
    // }
    //
    // This will let us sink the constant into the outer while first and then
    // into the inner while in a single run of this pass.
    c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
              [](const HloInstruction* instr) {
                return instr->opcode() == HloOpcode::kWhile;
              });
  }

  for (HloInstruction* while_instr : while_instrs) {
    // We only sink into while loop bodies, but this can be extended to
    // transform conditions as well.
    TF_ASSIGN_OR_RETURN(bool result,
                        TrySinkingConstantsIntoWhileBody(while_instr));
    changed |= result;
  }

  if (changed) {
    VLOG(2) << "HLO module after WhileLoopConstantSinking:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after WhileLoopConstantSinking";
  }

  return changed;
}
}  // namespace xla

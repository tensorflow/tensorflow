/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/while_loop_constant_sinking.h"

#include <cstdint>
#include <iterator>
#include <stack>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/while_util.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Replaces all uses of old_instr with new_instr except the use at
// `while_body_root` (which must be a tuple instruction) at index `tuple_index`.
// This utility helps us replace an instruction in the while body with a
// constant while still keeping it trivially loop invariant.
absl::Status ReplaceUsesWhileKeepingLoopInvariance(
    HloInstruction* old_instr, HloInstruction* new_instr,
    HloInstruction* while_body_root, int64_t tuple_index) {
  CHECK_EQ(while_body_root->opcode(), HloOpcode::kTuple);

  std::vector<HloInstruction*> users;
  users.reserve(old_instr->user_count());
  absl::c_copy(old_instr->users(), std::back_inserter(users));

  for (auto* user : users) {
    for (int64_t i = 0, e = user->operand_count(); i < e; i++) {
      if (user->operand(i) == old_instr &&
          !(user == while_body_root && i == tuple_index)) {
        TF_RETURN_IF_ERROR(user->ReplaceOperandWith(i, new_instr));
      }
    }
  }

  return absl::OkStatus();
}

HloInstruction* CloneHelper(const HloInstruction* instruction,
                            HloComputation* computation) {
  if (instruction->opcode() == HloOpcode::kConstant) {
    return computation->AddInstruction(instruction->Clone(/*suffix=*/".sunk"));
  }
  if (instruction->opcode() == HloOpcode::kBroadcast) {
    return computation->AddInstruction(instruction->CloneWithNewOperands(
        instruction->shape(),
        {CloneHelper(instruction->operand(0), computation)}));
  }
  LOG(FATAL) << "Unexpected instruction.";
}

}  // namespace

absl::StatusOr<bool> WhileLoopConstantSinking::TrySinkingConstantsIntoWhileLoop(
    HloModule* module, HloInstruction* while_instr) {
  HloComputation* while_cond = while_instr->while_condition();
  HloComputation* while_body = while_instr->while_body();

  const HloInstruction& init_value = *while_instr->operand(0);
  if (init_value.opcode() != HloOpcode::kTuple) {
    return false;
  }

  absl::flat_hash_map<int64_t, absl::InlinedVector<HloInstruction*, 1>>
      conditional_gte_index_to_insts =
          WhileUtil::GetGTEsMapForWhileConditional(*while_cond);
  std::vector<HloInstruction*> invariant_body_gtes =
      WhileUtil::GetInvariantGTEsForWhileBody(*while_body);

  HloCloneContext body_clone_context(module);
  HloCloneContext cond_clone_context(module);
  HloComputation* body_clone = nullptr;
  HloComputation* cond_clone = nullptr;
  for (HloInstruction* invariant_body_gte : invariant_body_gtes) {
    int64_t index = invariant_body_gte->tuple_index();
    const HloInstruction& invariant_value = *init_value.operand(index);

    // Original value should be a constant or broadcast of constant.
    if (invariant_value.opcode() != HloOpcode::kConstant &&
        (!sink_broadcast_of_constants_ ||
         invariant_value.opcode() != HloOpcode::kBroadcast ||
         invariant_value.operand(0)->opcode() != HloOpcode::kConstant)) {
      continue;
    }

    if (sink_only_scalar_constants_) {
      if (!ShapeUtil::IsScalar(init_value.operand(index)->shape())) {
        continue;
      }
    }

    // Sink into the while_body.
    // Should have at least one user that's not while_body_root.
    if (invariant_body_gte->user_count() > 1) {
      if (!body_clone) {
        body_clone = module->AddEmbeddedComputation(
            while_body->Clone("sunk", &body_clone_context));
        while_instr->set_while_body(body_clone);
      }
      HloInstruction* constant_instr =
          CloneHelper(&invariant_value, body_clone);
      TF_RETURN_IF_ERROR(ReplaceUsesWhileKeepingLoopInvariance(
          body_clone_context.FindInstruction(invariant_body_gte),
          constant_instr,
          body_clone_context.FindInstruction(while_body->root_instruction()),
          index));
    }

    // Check if there is a corresponding GTE in while_conditional.
    auto it = conditional_gte_index_to_insts.find(index);
    if (it == conditional_gte_index_to_insts.end()) {
      continue;
    }

    for (HloInstruction* invariant_cond_gte : it->second) {
      // Should have at least one user.
      if (invariant_cond_gte->user_count() > 0) {
        if (!cond_clone) {
          cond_clone = module->AddEmbeddedComputation(
              while_cond->Clone("sunk", &cond_clone_context));
          while_instr->set_while_condition(cond_clone);
        }
        HloInstruction* constant_instr =
            CloneHelper(&invariant_value, cond_clone);
        HloInstruction* cond_gte =
            cond_clone_context.FindInstruction(invariant_cond_gte);
        TF_RETURN_IF_ERROR(cond_gte->ReplaceAllUsesWith(constant_instr));
        TF_RETURN_IF_ERROR(cond_clone->RemoveInstruction(cond_gte));
      }
    }
  }

  return body_clone || cond_clone;
}

absl::StatusOr<bool> WhileLoopConstantSinking::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before WhileLoopConstantSinking:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> while_instrs;

  // Visit computations in order, from outermost to innermost.
  // We want to visit the outer while (while_0) before we visit the inner
  // while (while_1):
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
  std::stack<HloComputation*> agenda;
  agenda.push(module->entry_computation());
  absl::flat_hash_set<HloComputation*> visited;
  while (!agenda.empty()) {
    HloComputation* comp = agenda.top();
    agenda.pop();
    if (!visited.insert(comp).second) {
      continue;
    }
    for (auto* instr : comp->instructions()) {
      // Sinking constants may change the called computations, so do that first
      // if this is a while instruction.
      if (instr->opcode() == HloOpcode::kWhile) {
        TF_ASSIGN_OR_RETURN(bool result,
                            TrySinkingConstantsIntoWhileLoop(module, instr));
        changed |= result;
      }
      for (HloComputation* child : instr->called_computations()) {
        agenda.push(child);
      }
    }
  }
  TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());

  if (changed) {
    VLOG(2) << "HLO module after WhileLoopConstantSinking:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after WhileLoopConstantSinking";
  }

  return changed;
}
}  // namespace xla

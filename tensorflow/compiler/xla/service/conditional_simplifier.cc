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

#include "tensorflow/compiler/xla/service/conditional_simplifier.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

// Tries to replace a conditional with a call operation of the corresponding
// computation. If the given conditional has a constant predicate, tries to
// replace it with a call to its true/false computation as appropriate and then
// inline that computation.
//
// Returns true if it made a change to the graph.
static StatusOr<bool> TryRemoveConditional(HloInstruction* conditional) {
  CHECK_EQ(conditional->opcode(), HloOpcode::kConditional);
  // Do not remove conditionals that contain side-effecting instructions or
  // have control predecessors/successors in either true/false computation.
  if (!conditional->parent()->IsRemovable(conditional) ||
      conditional->HasSideEffect()) {
    VLOG(2) << "Not attempting to remove conditional as it is not removable or "
               "has side effect: "
            << conditional->ToShortString();
    return false;
  }

  if (conditional->operand(0)->opcode() != HloOpcode::kConstant) {
    VLOG(2) << "Not attempting to remove conditional as its predicate is not a "
               "compile-time constant: "
            << conditional->ToShortString();
    return false;
  }

  auto computation = conditional->parent();
  HloInstruction* call_op;
  if (conditional->operand(0)->literal().Get<bool>({})) {
    call_op = computation->AddInstruction(HloInstruction::CreateCall(
        conditional->shape(), {conditional->mutable_operand(1)},
        conditional->true_computation()));
  } else {
    call_op = computation->AddInstruction(HloInstruction::CreateCall(
        conditional->shape(), {conditional->mutable_operand(2)},
        conditional->false_computation()));
  }
  conditional->SetupDerivedInstruction(call_op);
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(conditional, call_op));
  TF_RETURN_IF_ERROR(CallInliner::Inline(call_op).status());

  return true;
}

StatusOr<bool> ConditionalSimplifier::Run(HloModule* module) {
  XLA_VLOG_LINES(
      3, "ConditionalSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;

  // Gather all the conditional ops in our module. We do this ahead of time so
  // we don't have to worry about mutating the lists of computations or
  // instructions as we iterate.
  std::vector<HloInstruction*> conditional_ops;
  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kConditional) {
        conditional_ops.push_back(instr);
      }
    }
  }

  for (HloInstruction* conditional_op : conditional_ops) {
    TF_ASSIGN_OR_RETURN(bool result, TryRemoveConditional(conditional_op));
    changed |= result;
  }

  XLA_VLOG_LINES(3,
                 "ConditionalSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla

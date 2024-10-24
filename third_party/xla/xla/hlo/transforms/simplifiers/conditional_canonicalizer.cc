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

#include "xla/hlo/transforms/simplifiers/conditional_canonicalizer.h"

#include <stack>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

absl::StatusOr<HloInstruction*> CanonicalizeNonTupleConditional(
    HloInstruction* conditional) {
  HloModule* module = conditional->parent()->parent();
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);
  auto parent = conditional->parent();
  const Shape& root_shape = conditional->shape();
  auto new_shape = ShapeUtil::MakeTupleShape(absl::MakeSpan(&root_shape, 1));
  auto new_conditional =
      parent->AddInstruction(conditional->CloneWithNewShape(new_shape));
  new_conditional->ReplaceCalledComputations([&](HloComputation* comp) {
    HloComputation* branch_clone =
        module->AddEmbeddedComputation(comp->Clone());
    HloInstruction* root = branch_clone->root_instruction();
    HloInstruction* tuple =
        branch_clone->AddInstruction(HloInstruction::CreateTuple({root}));
    branch_clone->set_root_instruction(tuple, /*accept_different_shape=*/true);
    return branch_clone;
  });
  auto gte = parent->AddInstruction(
      HloInstruction::CreateGetTupleElement(root_shape, new_conditional, 0));
  TF_RETURN_IF_ERROR(parent->ReplaceInstruction(conditional, gte));
  return new_conditional;
}
}  // namespace

absl::StatusOr<bool> ConditionalCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "ConditionalCanonicalizer::Run(), before:\n" + module->ToString());
  bool changed = false;
  std::stack<HloComputation*> agenda;
  agenda.push(module->entry_computation());
  absl::flat_hash_set<HloComputation*> visited;
  while (!agenda.empty()) {
    HloComputation* comp = agenda.top();
    agenda.pop();
    if (!visited.insert(comp).second) {
      continue;
    }
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      // Conditional canonicalization may change the called computations, so do
      // that first.
      if (inst->opcode() == HloOpcode::kConditional &&
          !inst->shape().IsTuple()) {
        TF_ASSIGN_OR_RETURN(inst, CanonicalizeNonTupleConditional(inst));
        changed = true;
      }
      for (HloComputation* child : inst->called_computations()) {
        agenda.push(child);
      }
    }
  }
  TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  XLA_VLOG_LINES(
      2, "ConditionalCanonicalizer::Run(), after:\n" + module->ToString());
  return changed;
}
}  // namespace xla

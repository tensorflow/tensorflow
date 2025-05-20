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

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace {

absl::StatusOr<bool> CanonicalizeNonTupleConditional(
    HloInstruction* conditional) {
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);
  bool changed = false;

  for (int i = 0; i < conditional->branch_count(); ++i) {
    auto* branch = conditional->branch_computation(i);
    TF_RET_CHECK(branch->num_parameters() == 1) << branch->num_parameters();

    // If this is used by something that's not a conditional, we need to make
    // a copy of the computation before we modify it. Note that if it's used
    // by multiple conditionals, that's ok, since the same rewrite is needed,
    // and it'll only get applied once.
    if (!branch->parameter_instruction(0)->shape().IsTuple() ||
        !branch->root_instruction()->shape().IsTuple()) {
      if (branch->caller_instructions().size() > 1) {
        bool copy_needed = false;
        for (HloInstruction* caller : branch->caller_instructions()) {
          if (ABSL_PREDICT_FALSE(caller->opcode() != HloOpcode::kConditional)) {
            copy_needed = true;
            break;
          }
        }

        if (copy_needed) {
          branch = branch->parent()->AddEmbeddedComputation(branch->Clone());
          conditional->set_branch_computation(i, branch);
          changed = true;
        }
      }
    }

    HloInstruction* const param = branch->parameter_instruction(0);
    // Canonicalize branch inputs to tuples.
    if (!param->shape().IsTuple()) {
      Shape shape = ShapeUtil::MakeTupleShape({param->shape()});
      HloInstruction* const new_param = branch->ReplaceParameter(
          0, HloInstruction::CreateParameter(0, shape, param->name()));
      HloInstruction* const gte = branch->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_param, 0));
      TF_RETURN_IF_ERROR(new_param->ReplaceAllUsesWithDifferentShape(gte));
      changed = true;
    }

    // Canonicalize branch output to tuples.
    HloInstruction* const root = branch->root_instruction();
    if (!root->shape().IsTuple()) {
      HloInstruction* tuple =
          branch->AddInstruction(HloInstruction::CreateTuple({root}));
      branch->set_root_instruction(tuple, /*accept_different_shape=*/true);
      changed = true;
    }
  }

  auto parent = conditional->parent();

  // Canonicalize conditional operands except the predicate.
  for (int i = 1; i < conditional->operand_count(); ++i) {
    auto* const operand = conditional->mutable_operand(i);
    if (!operand->shape().IsTuple()) {
      auto tuple =
          parent->AddInstruction(HloInstruction::CreateTuple({operand}));
      TF_RETURN_IF_ERROR(
          conditional->ReplaceOperandWithDifferentShape(i, tuple));
      changed = true;
    }
  }

  // Canonicalize conditional outputs to tuples.
  const Shape& root_shape = conditional->shape();
  if (!root_shape.IsTuple()) {
    auto new_shape = ShapeUtil::MakeTupleShape(absl::MakeSpan(&root_shape, 1));
    auto new_conditional =
        parent->AddInstruction(conditional->CloneWithNewShape(new_shape));
    auto gte = parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(root_shape, new_conditional, 0));
    TF_RETURN_IF_ERROR(parent->ReplaceInstruction(conditional, gte));
    changed = true;
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> ConditionalCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "ConditionalCanonicalizer::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kConditional) {
        TF_ASSIGN_OR_RETURN(changed, CanonicalizeNonTupleConditional(inst));
      }
    }
  }
  XLA_VLOG_LINES(
      2, "ConditionalCanonicalizer::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla

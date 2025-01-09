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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"

namespace xla {
namespace {
absl::Status CanonicalizeNonTupleConditional(HloInstruction* conditional) {
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);
  for (auto* branch : conditional->called_computations()) {
    HloInstruction* root = branch->root_instruction();
    TF_RET_CHECK(!root->shape().IsTuple());

    HloInstruction* tuple =
        branch->AddInstruction(HloInstruction::CreateTuple({root}));
    branch->set_root_instruction(tuple, /*accept_different_shape=*/true);
  }
  auto parent = conditional->parent();
  const Shape& root_shape = conditional->shape();
  auto new_shape = ShapeUtil::MakeTupleShape(absl::MakeSpan(&root_shape, 1));
  auto new_conditional =
      parent->AddInstruction(conditional->CloneWithNewShape(new_shape));
  auto gte = parent->AddInstruction(
      HloInstruction::CreateGetTupleElement(root_shape, new_conditional, 0));
  TF_RETURN_IF_ERROR(parent->ReplaceInstruction(conditional, gte));
  return absl::OkStatus();
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
      if (inst->opcode() == HloOpcode::kConditional &&
          !inst->shape().IsTuple()) {
        TF_RETURN_IF_ERROR(CanonicalizeNonTupleConditional(inst));
        changed = true;
      }
    }
  }
  XLA_VLOG_LINES(
      2, "ConditionalCanonicalizer::Run(), after:\n" + module->ToString());
  return changed;
}
}  // namespace xla

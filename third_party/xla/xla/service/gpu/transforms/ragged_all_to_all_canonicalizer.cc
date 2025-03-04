/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/ragged_all_to_all_canonicalizer.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> CanonicalizeRaggedAllToAll(
    HloInstruction* ragged_all_to_all, HloComputation* computation,
    HloModule* module) {
  if (HloPredicateIsNotOp<HloOpcode::kRaggedAllToAll>(ragged_all_to_all)) {
    return false;
  }

  // HLO verifier ensures that all offset and size operands have the same type,
  // but it's doesn't check what that type is. It is convenient for downstream
  // passes and emitters to upcast the offsets to S64.
  if (ragged_all_to_all->operand(2)->shape().element_type() == S64) {
    // The ragged-all-to-all is already canonicalized.
    return false;
  }

  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(ragged_all_to_all->operand_count());
  new_operands.push_back(ragged_all_to_all->mutable_operand(0));
  new_operands.push_back(ragged_all_to_all->mutable_operand(1));

  for (int i = 2; i < ragged_all_to_all->operand_count(); ++i) {
    HloInstruction* operand = ragged_all_to_all->mutable_operand(i);
    operand = computation->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::ChangeElementType(operand->shape(), S64), operand));
    new_operands.push_back(operand);
  }

  HloInstruction* new_ragged_all_to_all =
      computation->AddInstruction(HloInstruction::CreateRaggedAllToAll(
          ragged_all_to_all->shape(), new_operands,
          ragged_all_to_all->device_list(),
          /*channel_id=*/ragged_all_to_all->channel_id()));
  TF_RETURN_IF_ERROR(
      ragged_all_to_all->ReplaceAllUsesWith(new_ragged_all_to_all));
  TF_RETURN_IF_ERROR(
      computation->RemoveInstructionAndUnusedOperands(ragged_all_to_all));
  return true;
}

absl::StatusOr<bool> RaggedAllToAllCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (auto computation : module->computations(execution_threads)) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool canonicalized,
                          CanonicalizeRaggedAllToAll(hlo, computation, module));
      changed |= canonicalized;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla

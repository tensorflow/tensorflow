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

#include "xla/hlo/transforms/simplifiers/slice_mover.h"

#include <cstdint>
#include <map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {

// If an instruction has multiple slice users with the same slice parameters,
// remove all but one of the slice users.
absl::StatusOr<bool> CheckAndRemoveRedundantSlices(
    HloComputation* computation, HloInstruction* instruction) {
  bool changed = false;
  std::map<std::vector<int64_t>, std::vector<HloInstruction*>> slice_map;
  for (HloInstruction* user : instruction->users()) {
    if (user->opcode() == HloOpcode::kSlice) {
      std::vector<int64_t> slice_key;
      slice_key.insert(slice_key.end(), user->slice_starts().begin(),
                       user->slice_starts().end());
      slice_key.insert(slice_key.end(), user->slice_limits().begin(),
                       user->slice_limits().end());
      slice_key.insert(slice_key.end(), user->slice_strides().begin(),
                       user->slice_strides().end());
      slice_map[slice_key].push_back(user);
    }
  }

  for (const auto& [slice_key, slices] : slice_map) {
    if (slices.size() > 1) {
      changed = true;
      for (int i = 1; i < slices.size(); ++i) {
        TF_RETURN_IF_ERROR(
            computation->ReplaceInstruction(slices[i], slices[0]));
      }
    }
  }
  return changed;
}

// Move slice operations through add operations.
// Note that algebraic simplifier also has `HandleSlice` function.
absl::StatusOr<bool> MoveSliceOperations(HloComputation* computation) {
  bool changed = false;
  int change_count = 0;
  // TODO(ramzym): Generalize to element-wise operations.
  // TODO(ramzym): Consider also other operations like broadcast, reduce,
  // transpose, etc.
  // TODO(ramzym): Optimize the complexity.
  do {
    changed = false;
    std::vector<HloInstruction*> instructions =
        computation->MakeInstructionPostOrder();
    for (HloInstruction* instruction : instructions) {
      if (CheckAndRemoveRedundantSlices(computation, instruction).value()) {
        changed = true;
        change_count++;
        break;
      }
      if (instruction->opcode() == HloOpcode::kSlice &&
          instruction->operand(0)->opcode() == HloOpcode::kAdd) {
        HloInstruction* add = instruction->mutable_operand(0);
        HloInstruction* lhs = add->mutable_operand(0);
        HloInstruction* rhs = add->mutable_operand(1);
        // TODO(ramzym): Handle the case where the add operands can have
        // different types/layouts (e.g. f32 and f16).
        HloInstruction* lhs_slice =
            computation->AddInstruction(HloInstruction::CreateSlice(
                instruction->shape(), lhs, instruction->slice_starts(),
                instruction->slice_limits(), instruction->slice_strides()));
        HloInstruction* rhs_slice =
            computation->AddInstruction(HloInstruction::CreateSlice(
                instruction->shape(), rhs, instruction->slice_starts(),
                instruction->slice_limits(), instruction->slice_strides()));
        TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
            instruction,
            HloInstruction::CreateBinary(instruction->shape(), HloOpcode::kAdd,
                                         lhs_slice, rhs_slice)));
        changed = true;
        change_count++;
        break;
      }
    }
  } while (changed);

  return change_count > 0;
}
}  // anonymous namespace

absl::StatusOr<bool> SliceMover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool changed_computation,
                        MoveSliceOperations(computation));
    changed |= changed_computation;
  }
  return changed;
  return false;
}

}  // namespace xla

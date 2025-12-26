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

#include "xla/hlo/transforms/simplifiers/computation_canonicalizers.h"

#include <cstddef>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

namespace {
static bool IsConstant(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kConstant>(hlo);
}

static bool IsParameter(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kParameter>(hlo);
}

static bool IsGetTupleElement(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kGetTupleElement>(hlo);
}
}  // namespace

absl::StatusOr<bool> MoveGTEsRightAfterTupleDefinition(
    HloComputation& computation) {
  HloInstructionSequence new_sequence;
  HloSchedule& schedule = computation.parent()->schedule();
  const HloInstructionSequence sequence =
      schedule.GetOrCreateSequence(&computation);

  absl::flat_hash_set<HloInstruction*> moved_gtes;

  for (HloInstruction* inst : sequence.instructions()) {
    if (!moved_gtes.contains(inst)) {
      new_sequence.push_back(inst);
    }
    if (!inst->shape().IsTuple()) {
      continue;
    }
    for (HloInstruction* user : inst->users()) {
      if (IsGetTupleElement(user) && !user->HasControlDependencies()) {
        new_sequence.push_back(user);
        moved_gtes.insert(user);
      }
    }
  }

  bool changed = new_sequence != sequence;
  schedule.set_sequence(&computation, std::move(new_sequence));
  return changed;
}

absl::StatusOr<bool> MoveParametersAndConstantsToFront(
    HloComputation& computation) {
  HloInstructionSequence new_sequence;
  HloSchedule& schedule = computation.parent()->schedule();
  HloInstructionSequence& sequence = schedule.GetOrCreateSequence(&computation);

  for (HloInstruction* inst : sequence.instructions()) {
    if (IsParameter(inst) || IsConstant(inst)) {
      new_sequence.push_back(inst);

      // Because we move instruction to the front of the computation we can't
      // have any control predecessors, however silently dropping them is unsafe
      // as we can have transitive dependencies that define schedule order, so
      // we forward control predecessors to all users.
      for (HloInstruction* control_predecessor : inst->control_predecessors()) {
        for (HloInstruction* user : inst->users()) {
          TF_RETURN_IF_ERROR(control_predecessor->AddControlDependencyTo(user));
        }
      }
      TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
    }
  }

  for (HloInstruction* inst : sequence.instructions()) {
    if (!IsParameter(inst) && !IsConstant(inst)) {
      new_sequence.push_back(inst);
    }
  }

  schedule.set_sequence(&computation, new_sequence);
  const auto& old_instructions = sequence.instructions();
  const auto& new_instructions = new_sequence.instructions();
  for (size_t i = 0; i < old_instructions.size(); ++i) {
    if (old_instructions[i] != new_instructions[i]) {
      return true;
    }
  }
  return false;
}

}  // namespace xla

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

#include "xla/hlo/transforms/simplifiers/slice_hoisting.h"

#include <functional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla {

bool IsSlice(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kSlice ||
         instr->opcode() == HloOpcode::kDynamicSlice;
}

bool ReadyToSchedule(
    const HloInstruction* instr,
    const absl::flat_hash_set<HloInstruction*>& already_scheduled) {
  auto is_scheduled = [&](const HloInstruction* instr) {
    return already_scheduled.contains(instr);
  };

  return absl::c_all_of(instr->operands(), is_scheduled) &&
         absl::c_all_of(instr->control_predecessors(), is_scheduled);
}

// Hoists slices as close to the users as possible, this improves the
// HBM usage.
// Very simple algorithm that checks if a slice is ready to be scheduled, and
// schedules it if it is.
HloInstructionSequence HoistSlices(const HloInstructionSequence& sequence) {
  HloInstructionSequence new_sequence;
  new_sequence.reserve(sequence.instructions().size());

  absl::flat_hash_set<HloInstruction*> already_scheduled;
  already_scheduled.reserve(sequence.instructions().size());

  std::function<void(HloInstruction*)> schedule_instruction =
      [&](HloInstruction* instr) {
        if (already_scheduled.contains(instr)) {
          return;
        }
        new_sequence.push_back(instr);
        already_scheduled.insert(instr);
        for (HloInstruction* user : instr->users()) {
          if (IsSlice(user) && ReadyToSchedule(user, already_scheduled)) {
            schedule_instruction(user);
          }
        }
        for (HloInstruction* successor : instr->control_successors()) {
          if (IsSlice(successor) &&
              ReadyToSchedule(successor, already_scheduled)) {
            schedule_instruction(successor);
          }
        }
      };

  for (HloInstruction* instr : sequence.instructions()) {
    schedule_instruction(instr);
  }

  return new_sequence;
}

absl::StatusOr<bool> SliceHoisting::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  HloSchedule& schedule = module->schedule();
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(computation);
    HloInstructionSequence new_sequence = HoistSlices(sequence);
    if (new_sequence != sequence) {
      changed = true;
      sequence = std::move(new_sequence);
    }
  }
  return changed;
}

}  // namespace xla

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

#include "xla/hlo/transforms/simplifiers/constant_deferring.h"

#include <functional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla {

namespace {

bool IsConstant(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kConstant ||
         (instr->opcode() == HloOpcode::kBroadcast &&
          instr->operand(0)->opcode() == HloOpcode::kConstant);
}

}  // namespace

HloInstructionSequence DeferConstants(const HloInstructionSequence& sequence) {
  HloInstructionSequence new_sequence;
  new_sequence.reserve(sequence.instructions().size());
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      inst_to_constant_operands;
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      inst_to_constant_control_predecessors;
  absl::flat_hash_set<HloInstruction*> scheduled_constants;
  // Post-order DFS to adding add instruction and their constant predecessors to
  // the new sequence in topological order.
  std::function<void(HloInstruction*)> schedule_inst =
      [&](HloInstruction* instr) {
        if (scheduled_constants.contains(instr)) {
          return;
        }
        if (IsConstant(instr)) {
          scheduled_constants.insert(instr);
        }
        for (HloInstruction* operand : inst_to_constant_operands[instr]) {
          schedule_inst(operand);
        }
        for (HloInstruction* pred :
             inst_to_constant_control_predecessors[instr]) {
          schedule_inst(pred);
        }
        new_sequence.push_back(instr);
      };
  for (HloInstruction* instr : sequence.instructions()) {
    if (IsConstant(instr) && instr->user_count() > 0) {
      for (HloInstruction* user : instr->users()) {
        inst_to_constant_operands[user].push_back(instr);
      }
      for (HloInstruction* control_successor : instr->control_successors()) {
        inst_to_constant_control_predecessors[control_successor].push_back(
            instr);
      }
    } else if (inst_to_constant_operands.contains(instr) ||
               inst_to_constant_control_predecessors.contains(instr)) {
      schedule_inst(instr);
    } else {
      new_sequence.push_back(instr);
    }
  }
  return new_sequence;
}

absl::StatusOr<bool> ConstantDeferring::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  HloSchedule& schedule = module->schedule();
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(computation);
    HloInstructionSequence new_sequence = DeferConstants(sequence);
    if (new_sequence != sequence) {
      changed = true;
      sequence = std::move(new_sequence);
    }
  }
  return changed;
}

}  // namespace xla

/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/instruction_hoister.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "tsl/platform/status.h"

namespace xla {
namespace {

// Modifies the schedule to hoist parameters in all of the non-fusion
// computations. Hoisting entry parameters increases the opportunities to
// prefetch entry parameters. Hoisting non-entry parameters is required for
// correctness.
bool HoistParameters(
    HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  CHECK(module.has_schedule());
  HloSchedule& schedule = module.schedule();
  bool modified = false;
  for (const HloComputation* computation :
       module.MakeNonfusionComputations(execution_threads)) {
    CHECK(schedule.is_computation_scheduled(computation));
    if (computation->num_parameters() == 0) {
      continue;
    }
    const HloInstructionSequence& sequence = schedule.sequence(computation);
    bool hoisting_needed = false;
    int parameters_found = 0;
    for (HloInstruction* instruction : sequence.instructions()) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        ++parameters_found;
      }
      if (parameters_found == computation->num_parameters()) {
        break;
      }
      if (instruction->opcode() != HloOpcode::kGetTupleElement &&
          instruction->opcode() != HloOpcode::kTuple &&
          instruction->opcode() != HloOpcode::kConstant &&
          instruction->opcode() != HloOpcode::kBitcast &&
          instruction->opcode() != HloOpcode::kParameter) {
        hoisting_needed = true;
        break;
      }
    }

    if (!hoisting_needed) {
      continue;
    }
    modified = true;
    HloInstructionSequence new_sequence;
    for (HloInstruction* parameter : computation->parameter_instructions()) {
      TF_CHECK_OK(parameter->DropAllControlDeps());
      new_sequence.push_back(parameter);
    }
    for (HloInstruction* instruction : sequence.instructions()) {
      if (instruction->opcode() != HloOpcode::kParameter) {
        new_sequence.push_back(instruction);
      }
    }
    CHECK_EQ(new_sequence.size(), sequence.size());
    schedule.set_sequence(computation, new_sequence);
  }
  return modified;
}

// Modifies the schedules in the given module to hoist (move earlier) constant
// operations. This increases the opportunities to prefetch constant ops.
bool HoistConstantOperations(
    HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  CHECK(module.has_schedule());
  HloSchedule& schedule = module.schedule();
  bool modified = false;
  for (const HloComputation* computation :
       module.MakeNonfusionComputations(execution_threads)) {
    CHECK(schedule.is_computation_scheduled(computation));
    const HloInstructionSequence& sequence = schedule.sequence(computation);
    // Conservatively don't modify the schedule if any instruction has a control
    // successor or predecessor on a constant op. Computations with these
    // dependencies should be very rare anyway.
    bool contains_constant_successor_or_predecessors = false;
    for (HloInstruction* instruction : sequence.instructions()) {
      if (instruction->opcode() == HloOpcode::kConstant) {
        contains_constant_successor_or_predecessors |=
            !instruction->control_predecessors().empty();
        contains_constant_successor_or_predecessors |=
            !instruction->control_successors().empty();
      } else {
        auto is_constant = HloPredicateIsOp<HloOpcode::kConstant>;
        contains_constant_successor_or_predecessors |=
            absl::c_find_if(instruction->control_predecessors(), is_constant) !=
            instruction->control_predecessors().end();
        contains_constant_successor_or_predecessors |=
            absl::c_find_if(instruction->control_successors(), is_constant) !=
            instruction->control_successors().end();
      }
    }
    if (contains_constant_successor_or_predecessors) {
      continue;
    }
    modified = true;
    HloInstructionSequence new_sequence;

    for (HloInstruction* instruction : sequence.instructions()) {
      if (instruction->opcode() == HloOpcode::kConstant) {
        new_sequence.push_back(instruction);
      }
    }
    for (HloInstruction* instruction : sequence.instructions()) {
      if (instruction->opcode() != HloOpcode::kConstant) {
        new_sequence.push_back(instruction);
      }
    }
    CHECK_EQ(new_sequence.size(), sequence.size());
    schedule.set_sequence(computation, new_sequence);
  }
  return modified;
}
}  // namespace

absl::StatusOr<bool> InstructionHoister::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  if (hoist_parameters_) {
    modified |= HoistParameters(*module, execution_threads);
  }
  if (host_constants_) {
    modified |= HoistConstantOperations(*module, execution_threads);
  }
  return modified;
}

}  // namespace xla

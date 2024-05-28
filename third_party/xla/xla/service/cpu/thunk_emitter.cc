/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/thunk_emitter.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/copy_thunk.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

ThunkEmitter::ThunkEmitter(const BufferAssignment* buffer_assignment)
    : buffer_assignment_(buffer_assignment) {}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitEntryComputation(
    const HloModule& module) {
  if (!module.has_schedule()) {
    return absl::InternalError("HLO module must be scheduled to emit thunks");
  }
  return EmitHloComputation(module.entry_computation());
}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSlice(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return buffer_assignment_->GetUniqueSlice(instruction, index);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloComputation(
    const HloComputation* computation) {
  ThunkSequence thunks;

  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation)) {
    return absl::InternalError(
        absl::StrCat("Computation ", computation->name(),
                     " must be scheduled to emit thunks"));
  }

  const HloInstructionSequence& sequence = schedule.sequence(computation);
  for (HloInstruction* instr : sequence.instructions()) {
    TF_ASSIGN_OR_RETURN(ThunkSequence instr_thunks, EmitHloInstruction(instr));
    thunks.Append(std::move(instr_thunks));
  }

  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    // Instructions that do not have a thunk implementation and instead fully
    // defined by the corresponding buffer assignment.
    case HloOpcode::kBitcast:
    case HloOpcode::kParameter:
      return ThunkSequence::Empty();

    // Allocations for constants owned by the executable, and resolved at run
    // time according to the buffer assignment (using allocation index). We do
    // not need to emit any thunks for constant instructions.
    case HloOpcode::kConstant:
      return ThunkSequence::Empty();

    case HloOpcode::kCopy:
      return EmitCopyThunk(instruction);

    default:
      return absl::UnimplementedError(
          absl::StrCat("HLO opcode `", HloOpcodeString(instruction->opcode()),
                       "` is not supported by XLA:CPU ThunkEmitter"));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyThunk(
    const HloInstruction* copy) {
  TF_ASSIGN_OR_RETURN(auto source_buffer, GetAllocationSlice(copy->operand(0)));
  TF_ASSIGN_OR_RETURN(auto destination_buffer, GetAllocationSlice(copy));
  return ThunkSequence::Of<CopyThunk>(source_buffer, destination_buffer,
                                      ShapeUtil::ByteSizeOf(copy->shape()));
}

}  // namespace xla::cpu

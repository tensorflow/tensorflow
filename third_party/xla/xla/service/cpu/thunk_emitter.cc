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

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/cpu/runtime/call_thunk.h"
#include "xla/service/cpu/runtime/copy_thunk.h"
#include "xla/service/cpu/runtime/kernel_thunk.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime/while_thunk.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

ThunkEmitter::ThunkEmitter(IrEmitter2* ir_emitter,
                           const BufferAssignment* buffer_assignment)
    : ir_emitter_(ir_emitter), buffer_assignment_(buffer_assignment) {}

static Thunk::Info ThunkInfo(const HloInstruction* instruction) {
  const HloModule* module = instruction->GetModule();
  return Thunk::Info{std::string(instruction->name()),
                     std::string(module->name()), module->unique_id()};
}

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
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return ThunkSequence::Empty();

    // No-op operations that are used only to define an execution order for the
    // HLO dataflow graph.
    case HloOpcode::kAfterAll:
      return ThunkSequence::Empty();

    // Control flow thunks check predicates on the host and launch nested thunk
    // sequences for branches and loops.
    case HloOpcode::kWhile:
      return EmitWhileThunk(instruction);

    // Allocations for constants owned by the executable, and resolved at run
    // time according to the buffer assignment (using allocation index). We do
    // not need to emit any thunks for constant instructions.
    case HloOpcode::kConstant:
      return ThunkSequence::Empty();

    // Simple HLO instructions lowered to elemental host kernels (plain loops
    // behind the HostKernel API).
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog1p:
    case HloOpcode::kLog:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kRemainder:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kXor:
      return EmitElementalKernelThunk(instruction);

    case HloOpcode::kFusion:
      return EmitFusionKernelThunk(instruction);

    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return EmitReductionKernelThunk(instruction);

    case HloOpcode::kCall:
      return EmitCallThunk(instruction);

    case HloOpcode::kCopy:
      return EmitCopyThunk(instruction);

    default:
      return absl::UnimplementedError(
          absl::StrCat("HLO opcode `", HloOpcodeString(instruction->opcode()),
                       "` is not supported by XLA:CPU ThunkEmitter"));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCallThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(
      ThunkSequence called_sequence,
      EmitHloComputation(instruction->called_computations().front()));
  return ThunkSequence::Of<CallThunk>(ThunkInfo(instruction),
                                      std::move(called_sequence));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyThunk(
    const HloInstruction* instruction) {
  const HloInstruction* source = instruction->operand(0);
  TF_ASSIGN_OR_RETURN(auto source_buffer, GetAllocationSlice(source));
  TF_ASSIGN_OR_RETURN(auto destination_buffer, GetAllocationSlice(instruction));
  return ThunkSequence::Of<CopyThunk>(ThunkInfo(instruction), source_buffer,
                                      source->shape(), destination_buffer,
                                      instruction->shape());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitElementalKernelThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(auto kernel,
                      ir_emitter_->EmitElementalHostKernel(instruction));
  TF_ASSIGN_OR_RETURN(auto buffers, GetLeafAllocationSlices(instruction));

  // TODO(ezhulenev): IrEmitter should return requested ThreadDim for a kernel
  // invocation, for now we assume that we always emit a full loop.
  return ThunkSequence::Of<KernelThunk>(ThunkInfo(instruction), buffers,
                                        kernel.name, se::ThreadDim());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFusionKernelThunk(
    const HloInstruction* instruction) {
  auto* fusion = Cast<HloFusionInstruction>(instruction);
  TF_ASSIGN_OR_RETURN(auto kernel, ir_emitter_->EmitFusionHostKernel(fusion));
  TF_ASSIGN_OR_RETURN(auto buffers, GetLeafAllocationSlices(instruction));

  // TODO(ezhulenev): IrEmitter should return requested ThreadDim for a kernel
  // invocation, for now we assume that we always emit a full loop.
  return ThunkSequence::Of<KernelThunk>(ThunkInfo(instruction), buffers,
                                        kernel.name, se::ThreadDim());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReductionKernelThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(auto kernel,
                      ir_emitter_->EmitReductionHostKernel(instruction));
  TF_ASSIGN_OR_RETURN(auto buffers, GetLeafAllocationSlices(instruction));

  // TODO(ezhulenev): IrEmitter should return requested ThreadDim for a kernel
  // invocation, for now we assume that we always emit a full loop.
  return ThunkSequence::Of<KernelThunk>(ThunkInfo(instruction), buffers,
                                        kernel.name, se::ThreadDim());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitWhileThunk(
    const HloInstruction* instruction) {
  HloInstruction* cond = instruction->while_condition()->root_instruction();
  TF_ASSIGN_OR_RETURN(auto cond_buffer, GetAllocationSlice(cond));

  TF_ASSIGN_OR_RETURN(ThunkSequence cond_thunk,
                      EmitHloComputation(instruction->while_condition()));
  TF_ASSIGN_OR_RETURN(ThunkSequence body_thunk,
                      EmitHloComputation(instruction->while_body()));

  return ThunkSequence::Of<WhileThunk>(ThunkInfo(instruction), cond_buffer,
                                       std::move(cond_thunk),
                                       std::move(body_thunk));
}

absl::StatusOr<std::vector<BufferAllocation::Slice>>
ThunkEmitter::GetLeafAllocationSlices(const HloInstruction* instruction) {
  std::vector<BufferAllocation::Slice> buffers;
  auto add_buffers = [&](const HloInstruction* instr) -> absl::Status {
    for (const auto& indexed : ShapeUtil::GetLeafShapes(instr->shape())) {
      TF_ASSIGN_OR_RETURN(buffers.emplace_back(),
                          GetAllocationSlice(instr, indexed.index));
    }
    return absl::OkStatus();
  };
  for (HloInstruction* operand : instruction->operands()) {
    TF_RETURN_IF_ERROR(add_buffers(operand));
  }
  TF_RETURN_IF_ERROR(add_buffers(instruction));
  return buffers;
}

}  // namespace xla::cpu

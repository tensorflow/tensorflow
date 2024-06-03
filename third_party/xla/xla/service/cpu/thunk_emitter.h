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

#ifndef XLA_SERVICE_CPU_THUNK_EMITTER_H_
#define XLA_SERVICE_CPU_THUNK_EMITTER_H_

#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape_util.h"

namespace xla::cpu {

// ThunkEmitter is responsible for converting optimized HLO module into a
// sequence of thunks that will launch "work" on the CPU: launch host kernels,
// call into the libraries (oneDNN, Eigen, etc.).
//
// During the thunk emission it emits IR (LLVM IR) for the host kernels via the
// IrEmitter that later will be compiled into the executable binary (one or
// multiple LLVM modules compiled to object files).
class ThunkEmitter {
 public:
  ThunkEmitter(IrEmitter2* ir_emitter,
               const BufferAssignment* buffer_assignment);

  // Emits HLO module entry computation as a sequence of thunks.
  absl::StatusOr<ThunkSequence> EmitEntryComputation(const HloModule& module);

 private:
  // Returns the buffer allocation slice assigned to the given instruction at
  // the given shape index. Instruction must have a unique slice assigned to it!
  absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
      const HloInstruction* instruction, const ShapeIndex& index = {});

  absl::StatusOr<ThunkSequence> EmitHloComputation(
      const HloComputation* computation);

  absl::StatusOr<ThunkSequence> EmitHloInstruction(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitCallThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitCopyThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitElementalKernelThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitFusionKernelThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitReductionKernelThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitRngGetAndUpdateStateThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitInfeedThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitWhileThunk(
      const HloInstruction* instruction);

  // Return the list of buffer allocation slices assigned to the given
  // instruction that will be passed to the host kernel as arguments: a
  // flattened list of all the leaf buffers for all operands and result. We do
  // not materialize tuples at run time and only read and write from buffers
  // corresponding to arrays.
  absl::StatusOr<std::vector<BufferAllocation::Slice>>
  GetHostKernelAllocationSlices(const HloInstruction* instruction);

  IrEmitter2* ir_emitter_;
  const BufferAssignment* buffer_assignment_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_THUNK_EMITTER_H_

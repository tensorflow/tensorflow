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

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/cpu/runtime/resource_use.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/target_machine_features.h"
#include "xla/service/hlo_module_config.h"
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
  ThunkEmitter(IrEmitter2& ir_emitter,
               const BufferAssignment& buffer_assignment,
               const TargetMachineFeatures& target_machine_features,
               const HloModuleConfig& hlo_module_config);

  // Emits HLO module entry computation as a sequence of thunks.
  absl::StatusOr<ThunkSequence> EmitEntryComputation(const HloModule& module);

 private:
  struct HostKernelAllocationSlices {
    std::vector<BufferAllocation::Slice> arguments;
    std::vector<BufferAllocation::Slice> results;
  };

  // Returns the buffer allocation slice assigned to the given instruction at
  // the given shape index. Instruction must have a unique slice assigned to it!
  absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
      const HloInstruction* instruction, const ShapeIndex& index = {});

  // Returns a token resource corresponding to the given instruction result.
  absl::StatusOr<std::shared_ptr<Resource>> GetTokenResource(
      const HloInstruction* instruction, const ShapeIndex& index = {});

  absl::StatusOr<ThunkSequence> EmitHloComputation(
      const HloComputation* computation);

  absl::StatusOr<ThunkSequence> EmitHloInstruction(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitCallThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitConcatenateKernelThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitConvolutionThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitCopyThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitElementalKernelThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitFftThunk(const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitFusionKernelThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitReductionKernelThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitRngThunk(const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitRngGetAndUpdateStateThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitInfeedThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitOutfeedThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitConditionThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitWhileThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitDotThunk(const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitReplicaIdThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitPartitionIdThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitAllGatherThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitAllReduceThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitAllToAllThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitCollectivePermuteThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitReduceScatterThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitCustomCallThunk(
      const HloInstruction* instruction);

  absl::StatusOr<ThunkSequence> EmitSelectAndScatterThunk(
      const HloInstruction* instruction);

  // Returns the list of buffer allocation slices assigned to the given
  // instruction that will be passed to the host kernel as arguments: a
  // flattened list of all the leaf buffers for all operands and result. We do
  // not materialize tuples at run time and only read and write from buffers
  // corresponding to arrays.
  absl::StatusOr<HostKernelAllocationSlices> GetHostKernelAllocationSlices(
      const HloInstruction* instruction);

  // Verifies that the element types of all of the given operand instructions
  // match and are of one of the given supported types.
  absl::Status ElementTypesSameAndSupported(
      const HloInstruction& instruction,
      absl::Span<const HloInstruction* const> operands,
      absl::Span<const PrimitiveType> supported_types);

  IrEmitter2& ir_emitter_;
  const BufferAssignment& buffer_assignment_;

  const TargetMachineFeatures& target_machine_features_;
  const HloModuleConfig& hlo_module_config_;

  // A global resource that is used to order all collective operations.
  std::shared_ptr<Resource> communicator_resource_;

  // Token resources that correspond to the token buffer allocation slices. We
  // rely on buffer assignment to assign unique "identity" to each token, and
  // create a separate resource for each unique allocation slice.
  absl::flat_hash_map<BufferAllocation::Slice, std::shared_ptr<Resource>>
      token_resources_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_THUNK_EMITTER_H_

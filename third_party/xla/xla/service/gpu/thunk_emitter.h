/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_THUNK_EMITTER_H_
#define XLA_SERVICE_GPU_THUNK_EMITTER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"

namespace xla::gpu {

// Emits Thunks for the given HLO module.
class ThunkEmitter {
 public:
  absl::string_view platform_name() const {
    return ir_emitter_context_->platform_name();
  }

  explicit ThunkEmitter(
      IrEmitterContext* absl_nonnull ir_emitter_context,
      llvm_ir::LLVMCommandLineOptionsReleasableLock* absl_nonnull
          llvm_options_lock);
  ThunkEmitter(const ThunkEmitter&) = delete;
  ThunkEmitter& operator=(const ThunkEmitter&) = delete;

  absl::StatusOr<std::unique_ptr<SequentialThunk>> EmitHloEntryComputation(
      const HloModule* module);

  llvm::Module* constants_module() { return constants_module_.get(); }
  std::unique_ptr<llvm::Module> ConsumeConstantsModule() {
    return std::move(constants_module_);
  }
  std::vector<std::unique_ptr<llvm::Module>> ConsumeKernelModules() {
    return std::move(kernel_modules_);
  }

 private:
  // Emits code for the given HLO computation.
  //
  // Also populates related information to 'ir_emitter_context_' for
  // large-constant initializations. Large constants don't get initializers in
  // the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  absl::StatusOr<ThunkSequence> EmitHloComputation(
      const HloComputation* computation);

  absl::StatusOr<ThunkSequence> EmitHloInstruction(
      const HloInstruction* hlo, bool emit_group_thunks = false);

  absl::StatusOr<ThunkSequence> EmitAsyncStart(const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitAsyncComputation(const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitAsyncCustomCallStart(
      const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitAsyncDone(const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCommandBufferThunk(
      const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCollectiveAsyncDone(
      Thunk::Kind kind, const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCollectiveGroupStartThunk(
      const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCollectiveMetadata(
      const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCollectivePermute(
      const HloCollectivePermuteInstruction* hlo);

  template <typename CollectiveThunkType, typename HloInstType>
  absl::StatusOr<ThunkSequence> EmitCollectiveThunk(
      Thunk::Kind kind, const HloInstruction* async_start,
      const HloInstType* inst, std::optional<bool> use_global_device_ids);

  absl::StatusOr<ThunkSequence> EmitConditional(const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitConstant(const HloConstantInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitConvolutionReorderThunk(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitConvolutionThunk(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCopy(const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCopyStartThunk(
      const HloCopyStartInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCopyDoneThunk(const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCuDnnThunk(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCubDeviceRadixSort(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulThunk(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulThunkF8(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitCustomCallThunk(
      const HloCustomCallInstruction* hlo);

  template <typename HloInstType>
  absl::StatusOr<ThunkSequence> EmitDegeneratedCollectiveThunk(
      std::vector<CollectiveThunk::Buffer>& buffers,
      const HloInstruction* async_start, const HloInstType* inst);

  absl::StatusOr<ThunkSequence> EmitFusion(const HloFusionInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitFftThunk(const HloFftInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitGemmThunk(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitInfeed(const HloInfeedInstruction* hlo);

  template <typename NvshmemAllReduceThunkType,
            typename HloAllReduceInstruction>
  absl::StatusOr<ThunkSequence> EmitNvshmemThunk(
      Thunk::Kind kind, const HloInstruction* async_start,
      const HloAllReduceInstruction* inst,
      std::optional<bool> use_global_device_ids);

  absl::StatusOr<ThunkSequence> EmitNvshmemAsyncDone(Thunk::Kind kind,
                                                     const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitNormThunk(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitOutfeed(const HloOutfeedInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitPadToStatic(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitPtxCustomCall(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitRecvDoneThunk(
      const HloRecvDoneInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitRecvThunk(const HloRecvInstruction* hlo,
                                              bool emit_group_thunks);

  template <typename ThunkType>
  absl::StatusOr<ThunkSequence> EmitReplicaOrPartitionId(
      const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitRngGetAndUpdateState(
      const HloRngGetAndUpdateStateInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitSliceToDynamic(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitSendDoneThunk(
      const HloSendDoneInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitSendThunk(const HloSendInstruction* hlo,
                                              bool emit_group_thunks);

  absl::StatusOr<ThunkSequence> EmitSort(const HloSortInstruction* sort);

  absl::StatusOr<ThunkSequence> EmitTopKCustomCall(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitTriangularSolveCustomCall(
      const HloInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitTritonCustomCall(
      const HloCustomCallInstruction* hlo);

  absl::StatusOr<ThunkSequence> EmitWhile(const HloInstruction* hlo);

  absl::Status AssertNonDeterminismIsOkay(const std::string& op_name);

  absl::StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;
  absl::StatusOr<ShapedSlice> GetShapedSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;

  CollectivesAsyncEvents& GetCollectivesAsyncEvents() {
    return ir_emitter_context_->collectives_async_events();
  }

  InstructionToHostExecuteAsyncEvents&
  GetInstructionToHostExecuteAsyncEvents() {
    return ir_emitter_context_->instruction_to_host_execute_async_events();
  }
  IrEmitterContext* ir_emitter_context_;

  // Container for async host send/recv events shared by host send/recv thunks.
  std::shared_ptr<HostSendRecvAsyncEvents> send_recv_events_;

  // Container for async copy-start/copy-done events.
  std::shared_ptr<CopyThunk::AsyncEvents> copy_events_;

  // Shared buffer addresses registry for NVSHMEM put/get operations.
  std::shared_ptr<NvshmemBufferAddresses> nvshmem_buffer_addresses_;

  // Cache to store the call_graph.
  std::unique_ptr<CallGraph> call_graph_;

  // Module with constants.
  std::unique_ptr<llvm::Module> constants_module_;

  // Modules for each emitted kernel.
  std::vector<std::unique_ptr<llvm::Module>> kernel_modules_;

  // Releasable lock for LLVM options. Most of the thunks are emitted under the
  // lock, however some thunks (e.g. custom calls) temporarily release the lock
  // to avoid deadlocks when foreign code calls into LLVM with a different
  // set of options.
  llvm_ir::LLVMCommandLineOptionsReleasableLock* llvm_options_lock_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_THUNK_EMITTER_H_

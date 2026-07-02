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
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/gpu_hlo_ordering.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"

namespace xla::gpu {

struct DynamicSliceCopyFusion;
struct StaticSliceCopyFusion;

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

  xla::Future<std::unique_ptr<SequentialThunk>> EmitHloEntryComputation(
      const HloModule* module);

  llvm::Module* constants_module() { return constants_module_.get(); }
  LlvmKernelSource ConsumeConstantsModule() {
    return LlvmKernelSource{std::move(constants_module_context_),
                            std::move(constants_module_)};
  }

 private:
  // Emits code for the given HLO computation.
  //
  // Also populates related information to 'ir_emitter_context_' for
  // large-constant initializations. Large constants don't get initializers in
  // the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  AsyncThunkSequence EmitHloComputation(const HloComputation* computation);

  AsyncThunkSequence EmitHloInstruction(const HloInstruction* hlo,
                                        bool emit_group_thunks = false);

  // Calls the right function to emit the custom call thunk for `hlo`.
  AsyncThunkSequence EmitCustomCallSwitch(const HloInstruction* hlo);

  AsyncThunkSequence EmitAsyncStart(const HloInstruction* instr);

  AsyncThunkSequence EmitCallComputation(const HloInstruction* instr);

  AsyncThunkSequence EmitAsyncComputation(const HloInstruction* instr);

  AsyncThunkSequence EmitAsyncCustomCallStart(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitAsyncDone(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCollectiveAsyncDone(
      const HloInstruction* inst);

  AsyncThunkSequence EmitCollectiveGroupStartThunk(const HloInstruction* instr);

  // Async start is either an AsyncStart instruction or a
  // CollectivePermuteStart.
  absl::StatusOr<ThunkSequence> EmitCollectivePermute(
      const HloCollectivePermuteInstruction* instr,
      const HloInstruction* absl_nonnull async_start);

  template <typename CollectiveThunkType, typename HloInstType>
  AsyncThunkSequence EmitCollectiveThunk(
      Thunk::Kind kind, const HloInstruction* async_start,
      const HloInstType* inst, std::optional<bool> use_global_device_ids);

  AsyncThunkSequence EmitConditional(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitConstant(
      const HloConstantInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitConvolutionReorderThunk(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitConvolutionThunk(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCopy(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCopyStartThunk(
      const HloCopyStartInstruction* copy_start_instr);

  absl::StatusOr<ThunkSequence> EmitCopyDoneThunk(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCuDnnThunk(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulThunk(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulThunkF8(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCublasLtGroupedMatmulThunk(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulThunkMx(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCustomCallThunk(
      const HloCustomCallInstruction* instr);

  template <typename HloInstType>
  absl::StatusOr<ThunkSequence> EmitDegeneratedCollectiveThunk(
      std::vector<CollectiveThunk::Buffer>& buffers,
      const HloInstruction* async_start, const HloInstType* inst);

  AsyncThunkSequence EmitFusion(const HloFusionInstruction* instr);

  AsyncThunkSequence EmitDynamicSliceCopyFusion(
      const HloFusionInstruction* instr, DynamicSliceCopyFusion copy);

  AsyncThunkSequence EmitStaticSliceCopyFusion(
      const HloFusionInstruction* instr, const StaticSliceCopyFusion& copy);

  absl::StatusOr<ThunkSequence> EmitFftThunk(const HloFftInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitInfeed(const HloInfeedInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitNormThunk(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitOutfeed(const HloOutfeedInstruction* instr);

  AsyncThunkSequence EmitPadToStatic(const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitPtxCustomCall(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitRecvDoneThunk(
      const HloRecvDoneInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitRecvThunk(const HloRecvInstruction* instr,
                                              bool emit_group_thunks);

  template <typename ThunkType>
  absl::StatusOr<ThunkSequence> EmitReplicaOrPartitionId(
      const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitRngSeedThunk(const HloInstruction* instr);

  AsyncThunkSequence EmitRngGetAndUpdateState(
      const HloRngGetAndUpdateStateInstruction* instr);

  AsyncThunkSequence EmitSliceToDynamic(const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitSendDoneThunk(
      const HloSendDoneInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitSendThunk(const HloSendInstruction* instr,
                                              bool emit_group_thunks);

  AsyncThunkSequence EmitSort(const HloSortInstruction* sort);

  absl::StatusOr<ThunkSequence> EmitTopKCustomCall(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitTriangularSolveCustomCall(
      const HloInstruction* instr);

  AsyncThunkSequence EmitTritonCustomCall(
      const HloCustomCallInstruction* instr);

  AsyncThunkSequence EmitWhile(const HloInstruction* instr);

  absl::Status AssertNonDeterminismIsOkay(const std::string& op_name);

  AsyncThunkSequence EmitDynamicSliceFusionV2(
      const HloFusionInstruction* instr);

  std::optional<BufferAllocation::Slice> GetAllocationOverride(
      const HloInstruction* instr, const ShapeIndex& index) const;
  absl::StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;
  absl::StatusOr<ShapedSlice> GetShapedSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;

  InstructionToHostExecuteAsyncEvents&
  GetInstructionToHostExecuteAsyncEvents() {
    return ir_emitter_context_->instruction_to_host_execute_async_events();
  }
  IrEmitterContext* ir_emitter_context_;

  // Container for async host send/recv events shared by host send/recv thunks.
  std::shared_ptr<HostSendRecvAsyncEvents> send_recv_events_;

  // Maps async-start instructions to their AsyncExecution so that the
  // corresponding async-done can emit an AsyncDoneThunk sharing the same
  // AsyncExecution.
  absl::flat_hash_map<const HloInstruction*, std::shared_ptr<AsyncExecution>>
      hlo_async_executions_;

  // Cache to store the call_graph.
  std::unique_ptr<CallGraph> call_graph_;

  std::unique_ptr<llvm::LLVMContext> constants_module_context_;
  std::unique_ptr<llvm::Module> constants_module_;

  // TODO(tjoerg): Attach the HloOrdering to the HloSchedule instead of
  // re-creating it here.
  absl::flat_hash_map<const HloModule*,
                      std::unique_ptr<ConcurrentRegionsHloOrdering>>
      concurrent_regions_ordering_;

  // Releasable lock for LLVM options. Most of the thunks are emitted under the
  // lock, however some thunks (e.g. custom calls) temporarily release the lock
  // to avoid deadlocks when foreign code calls into LLVM with a different
  // set of options.
  llvm_ir::LLVMCommandLineOptionsReleasableLock* llvm_options_lock_;

  // AllocationOverrides lets EmitDynamicSliceFusionV2 redirect buffer lookups
  // for specific HLO instructions. When emitting embedded thunks for a
  // dynamic-slice fusion, the hero's operands and results must map to
  // synthetic BufferAllocation::Slices (the "embedded_allocations") rather
  // than the real buffer assignment. InstallAllocationOverrides sets the map;
  // GetAllocationSliceForHlo checks it before falling through to the normal
  // buffer assignment. The returned cleanup object restores the empty state.
  using AllocationOverrides =
      absl::flat_hash_map<const HloInstruction*,
                          std::vector<BufferAllocation::Slice>>;
  auto InstallAllocationOverrides(AllocationOverrides overrides) {
    allocation_overrides_ = std::move(overrides);
    return absl::MakeCleanup([this] { allocation_overrides_.clear(); });
  }
  AllocationOverrides allocation_overrides_;

  // Stores HloFusionAnalysis objects to ensure they outlive any asynchronous
  // operations that may hold references to them.
  std::vector<std::unique_ptr<HloFusionAnalysis>> analysis_garbage_collector_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_THUNK_EMITTER_H_

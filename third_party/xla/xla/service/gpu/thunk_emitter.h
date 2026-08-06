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
#include "xla/future.h"
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

// Lowers a scheduled HLO module to a sequence of GPU runtime thunks.
//
// ThunkEmitter follows the structure of the HLO program: HLO emitters create
// thunks implementing their operation at run time. Control flow operations are
// implemented as a composition of nested thunk sequences.
//
// ThunkEmitter keeps HLO semantic dispatch separate from thunk emission.
// Dispatch* methods select a specific emitter when an HLO construct has
// different runtime semantics based on its properties or location; Emit*
// methods implement the selected semantics. Send/recv semantics depend on two
// things. `is_host_transfer()` selects host-transfer or device-transfer
// semantics; host transfers synchronize with handler completion events. For
// device transfers, the parent computation determines whether the operation is
// an implicit async start or a synchronous operation inside a generic async
// computation.
//
// Thunk emission also records large-constant initializations in
// `ir_emitter_context_`. Large constants do not get initializers in generated
// code and must be initialized by XLA from the stored content. Constants with
// initializers in generated code have empty content.
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

  Future<ThunkSequence> EmitHloEntryComputation(const HloModule* module);

  llvm::Module* constants_module() { return constants_module_.get(); }
  LlvmKernelSource ConsumeConstantsModule() {
    return LlvmKernelSource{std::move(constants_module_context_),
                            std::move(constants_module_)};
  }

 private:
  //===--------------------------------------------------------------------===//
  // Context-dependent HLO dispatch
  //===--------------------------------------------------------------------===//

  // Dispatch* methods inspect HLO properties and location and select the
  // specific Emit* method implementing the resulting runtime semantics. They
  // do not construct thunks directly.

  // Dispatches a generic async-start based on its wrapped computation and
  // runtime behavior. Host operations use their dedicated completion paths,
  // synchronous collectives emit bare collective thunks, and all other
  // operations use generic async execution.
  Future<ThunkSequence> DispatchAsyncStart(const HloInstruction* instr);

  // Dispatches legacy and generic async-done HLOs to the completion path
  // selected by their start or wrapped operation.
  absl::StatusOr<ThunkSequence> DispatchAsyncDone(const HloInstruction* instr);

  // DispatchSend/DispatchRecv select host-transfer or device-transfer
  // semantics using `is_host_transfer()`. Device send/recv lowering also
  // depends on the parent computation:
  //
  //  - Outside an async computation, send/recv implicitly starts an async
  //    execution that is completed by the matching send/recv-done instruction.
  //  - Inside an async computation, send/recv lowers directly to
  //    `SendThunk`/`RecvThunk`. The enclosing generic async-start/async-done
  //    pair owns execution and completion.
  //
  absl::StatusOr<ThunkSequence> DispatchSend(const HloSendInstruction* instr);
  absl::StatusOr<ThunkSequence> DispatchSendDone(
      const HloSendDoneInstruction* instr);

  absl::StatusOr<ThunkSequence> DispatchRecv(const HloRecvInstruction* instr);
  absl::StatusOr<ThunkSequence> DispatchRecvDone(
      const HloRecvDoneInstruction* instr);

  // Dispatches a custom call to specialized thunk emission when supported and
  // falls back to generic custom-call emission.
  Future<ThunkSequence> DispatchCustomCall(const HloInstruction* hlo);

  // Dispatches a legacy collective-start to synchronous collective emission or
  // generic async execution.
  Future<ThunkSequence> DispatchLegacyCollectiveStart(
      const HloInstruction* instr);

  //===--------------------------------------------------------------------===//
  // Structural HLO traversal
  //===--------------------------------------------------------------------===//

  // HLO traversal follows a depth-first traversal of the HLO program. Each
  // Emit* method lowers a single HLO operation and recursively composes
  // Dispatch* and Emit* methods for nested HLO computations. Methods return
  // Future<ThunkSequence> when nested emission can complete asynchronously (if
  // thunk emission is expensive, i.e. requires LLVM compilation).

  // Traverses `computation` in schedule order and recursively emits each
  // instruction. This is the computation traversal primitive and is not an HLO
  // opcode handler.
  Future<ThunkSequence> EmitHloComputation(const HloComputation* computation);

  // Emits thunks for `hlo` by dispatching to the operation-specific Dispatch*
  // or Emit* method. Opcode dispatch is intentionally shallow;
  // operation-specific control flow belongs in the handler.
  Future<ThunkSequence> EmitHloInstruction(const HloInstruction* hlo);

  //===--------------------------------------------------------------------===//
  // HLO-specific thunk emission
  //===--------------------------------------------------------------------===//

  // Registers and returns the AsyncExecution shared by an async-start/done
  // pair.
  absl::StatusOr<std::shared_ptr<AsyncExecution>> RegisterAsyncExecution(
      const HloInstruction* async_start);

  // Emits a generic async start by recursively emitting its wrapped
  // computation and wrapping the resulting thunks in an AsyncStartThunk.
  Future<ThunkSequence> EmitAsyncStart(const HloInstruction* instr);

  // Emits only an AsyncStartThunk for the already emitted `thunks`, using the
  // registered `execution`.
  absl::StatusOr<ThunkSequence> EmitAsyncStart(
      std::shared_ptr<AsyncExecution> execution,
      const HloInstruction* async_start, ThunkSequence thunks);

  // Emits an AsyncDoneThunk that waits for the execution started by `start`.
  absl::StatusOr<ThunkSequence> EmitAsyncDone(const HloInstruction* done,
                                              const HloInstruction* start);

  // Emits an async start for a device send/recv outside an async computation.
  // Pipelined starts create or reuse the execution owned by their canonical
  // start. This is called synchronously during HLO traversal.
  absl::StatusOr<ThunkSequence> EmitAsyncSendRecvStart(
      const HloSendRecvInstruction* async_start, ThunkSequence thunks);

  // Emits a kCall by invoking EmitHloComputation for its called computation.
  Future<ThunkSequence> EmitCall(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitHostExecuteStart(
      const HloInstruction* async_start,
      const HloCustomCallInstruction* host_execute);

  absl::StatusOr<ThunkSequence> EmitHostExecuteDone(
      const HloInstruction* async_done,
      const HloCustomCallInstruction* host_execute);

  Future<ThunkSequence> EmitCollective(const HloInstruction* collective);

  Future<ThunkSequence> EmitCollectiveGroup(const HloInstruction* instr);

  Future<ThunkSequence> EmitCollectiveKernel(
      Thunk::ThunkInfo thunk_info, std::vector<CollectiveThunk::Buffer> buffers,
      const HloInstruction* instr, const CollectiveConfig& config);

  template <typename CollectiveThunkType, typename HloInstType>
  Future<ThunkSequence> EmitCollective(
      Thunk::Kind kind, const HloInstType* inst,
      std::optional<bool> use_global_device_ids);

  template <typename HloInstType>
  absl::StatusOr<ThunkSequence> EmitDegeneratedCollective(
      std::vector<CollectiveThunk::Buffer>& buffers, const HloInstType* inst);

  Future<ThunkSequence> EmitConditional(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitConstant(
      const HloConstantInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitConvolutionReorder(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitConvolution(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCopy(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCopyStart(
      const HloCopyStartInstruction* copy_start_instr);

  absl::StatusOr<ThunkSequence> EmitCopyDone(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCuDnn(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCublasLtMatmul(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulF8(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCublasLtGroupedMatmul(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitCublasLtMatmulMx(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitGenericCustomCall(
      const HloCustomCallInstruction* instr);

  Future<ThunkSequence> EmitFusion(const HloFusionInstruction* instr);

  Future<ThunkSequence> EmitDynamicSliceCopyFusion(
      const HloFusionInstruction* instr, DynamicSliceCopyFusion copy);

  Future<ThunkSequence> EmitStaticSliceCopyFusion(
      const HloFusionInstruction* instr, const StaticSliceCopyFusion& copy);

  absl::StatusOr<ThunkSequence> EmitFft(const HloFftInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitInfeed(const HloInfeedInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitNorm(const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitOutfeed(const HloOutfeedInstruction* instr);

  Future<ThunkSequence> EmitPadToStatic(const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitPtxCustomCall(
      const HloCustomCallInstruction* instr);

  // Emits device send/recv as synchronous operations without async wrapping.
  // These instructions have `is_host_transfer() == false`;
  // DispatchSend/DispatchRecv use their parent computation to decide whether
  // to wrap the emitted thunks in an implicit async start.
  absl::StatusOr<ThunkSequence> EmitSend(const HloSendInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitRecv(const HloRecvInstruction* instr);

  // Completes an implicit async device send/recv outside an async computation.
  absl::StatusOr<ThunkSequence> EmitSendDone(
      const HloSendDoneInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitRecvDone(
      const HloRecvDoneInstruction* instr);

  // Emits host-transfer send/recv (`is_host_transfer() == true`). Host
  // transfers use handler completion events and are never wrapped in generic
  // async execution.
  absl::StatusOr<ThunkSequence> EmitHostSend(const HloSendInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitHostRecv(const HloRecvInstruction* instr);
  absl::StatusOr<ThunkSequence> EmitHostSendDone(
      const HloInstruction* done, const HloSendRecvInstruction* host_transfer);
  absl::StatusOr<ThunkSequence> EmitHostRecvDone(
      const HloInstruction* done, const HloSendRecvInstruction* host_transfer);

  template <typename ThunkType>
  absl::StatusOr<ThunkSequence> EmitReplicaOrPartitionId(
      const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitRngSeed(const HloInstruction* instr);

  Future<ThunkSequence> EmitRngGetAndUpdateState(
      const HloRngGetAndUpdateStateInstruction* instr);

  Future<ThunkSequence> EmitSliceToDynamic(
      const HloCustomCallInstruction* instr);

  Future<ThunkSequence> EmitSort(const HloSortInstruction* sort);

  absl::StatusOr<ThunkSequence> EmitTopKCustomCall(
      const HloCustomCallInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitTriangularSolveCustomCall(
      const HloInstruction* instr);

  Future<ThunkSequence> EmitTritonCustomCall(
      const HloCustomCallInstruction* instr);

  Future<ThunkSequence> EmitWhile(const HloInstruction* instr);

  absl::Status AssertNonDeterminismIsOkay(const std::string& op_name);

  Future<ThunkSequence> EmitDynamicSliceFusionV2(
      const HloFusionInstruction* instr);

  std::optional<BufferAllocation::Slice> GetAllocationOverride(
      const HloInstruction* instr, const ShapeIndex& index) const;
  absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;
  absl::StatusOr<ShapedSlice> GetShapedSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;

  InstructionToHostExecuteAsyncEvents&
  GetInstructionToHostExecuteAsyncEvents() {
    return ir_emitter_context_->instruction_to_host_execute_async_events();
  }
  IrEmitterContext* ir_emitter_context_;

  // Completion events shared by host-transfer send/recv start and done thunks.
  std::shared_ptr<HostSendRecvAsyncEvents> send_recv_events_;

  // Maps async-start instructions to their AsyncExecution so that the
  // corresponding async-done can emit an AsyncDoneThunk sharing the same
  // AsyncExecution. Registry access is confined to synchronous HLO traversal;
  // asynchronous emission callbacks capture the registered shared pointer.
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
  // GetAllocationSlice checks it before falling through to the normal
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

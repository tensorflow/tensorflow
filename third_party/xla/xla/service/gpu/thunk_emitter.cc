/*Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/thunk_emitter.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/llvm/llvm_emitter.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"
#include "xla/backends/gpu/runtime/convolution_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/device_to_host_copy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/host_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/infeed_thunk.h"
#include "xla/backends/gpu/runtime/legacy_custom_call_thunk.h"
#include "xla/backends/gpu/runtime/norm_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/rng_seed_thunk.h"
#include "xla/backends/gpu/runtime/select_k_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/topk.h"
#include "xla/backends/gpu/runtime/triangular_solve_thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/backends/gpu/transforms/dynamic_slice_copy.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/ffi/attribute_map.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/transforms/gpu_passes.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/custom_kernel_emitter.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_hlo_ordering.h"
#include "xla/service/gpu/gpu_norm_runner.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/triton_call.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/human_readable_json.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla::gpu {
namespace {

absl::StatusOr<TritonKernelSource> EmitTritonFrom(
    const TritonCall& call, const std::string& kernel_name,
    mlir::MLIRContext& mlir_context) {
  VLOG(3) << "Generating: " << kernel_name;

  mlir::OwningOpRef<mlir::ModuleOp> triton_module;
  {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&mlir_context);
    triton_module =
        mlir::parseSourceString<mlir::ModuleOp>(call.ir, &mlir_context);
    if (!triton_module) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse Triton module: ",
                       diagnostic_handler.ConsumeStatus().message(),
                       "\ninput ir: \"", absl::CHexEscape(call.ir), "\""));
    }
  }

  auto triton_fn = triton_module->lookupSymbol<mlir::triton::FuncOp>(call.name);
  TF_RET_CHECK(triton_fn) << "Call name not found in the Triton module: "
                          << call.name;
  triton_fn.setName(kernel_name);

  return TritonKernelSource(std::move(triton_module));
}

// TODO: move into a host_execute specific file.
bool IsHostExecuteCustomCall(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() ==
             "HostExecute";  // TODO: this constant string should be shared with
                             // the TPU one
}

bool IsImplicitAsyncSendRecvStart(const HloInstruction* instr) {
  // A device send/recv outside an async computation implicitly acts as an
  // async-start even though its HLO opcode does not spell out "start". Inside
  // an async computation it is emitted as a synchronous operation; the
  // enclosing generic async-start/done pair owns asynchronous execution and
  // completion.
  return !instr->parent()->IsAsyncComputation();
}

bool HasCollectivesGroupAttribute(const HloInstruction* instr) {
  return instr->frontend_attributes().map().contains(
      kCollectiveGroupMarkerAttr);
}

bool ShouldEmitCollectiveSynchronously(const HloInstruction* instr,
                                       const DebugOptions& debug_options) {
  // With an overlap limit greater than one, the scheduler can keep multiple
  // collectives in flight on different communication streams. Preserve their
  // async execution scopes even when the collective is marked `is_sync`.
  return IsGPUSyncCollective(*instr) &&
         debug_options
                 .xla_gpu_experimental_parallel_collective_overlap_limit() <= 1;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Context-dependent HLO dispatch
//===----------------------------------------------------------------------===//

Future<ThunkSequence> ThunkEmitter::DispatchAsyncStart(
    const HloInstruction* instr) {
  if (instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    const HloInstruction* wrapped = instr->async_wrapped_instruction();

    // Host send/recv are selected by `is_host_transfer()` and use handler
    // completion events. HostExecute also has its own completion semantics.
    // Neither is nested in generic AsyncStartThunk/AsyncDoneThunk.
    if (auto* send = DynCast<HloSendInstruction>(wrapped);
        send != nullptr && send->is_host_transfer()) {
      return EmitHostSend(send);
    }
    if (auto* recv = DynCast<HloRecvInstruction>(wrapped);
        recv != nullptr && recv->is_host_transfer()) {
      return EmitHostRecv(recv);
    }
    if (auto* call = DynCast<HloCustomCallInstruction>(wrapped);
        call != nullptr && IsHostExecuteCustomCall(*call)) {
      return EmitHostExecuteStart(instr, call);
    }
  }

  if (ShouldEmitCollectiveSynchronously(instr,
                                        ir_emitter_context_->debug_options())) {
    return HasCollectivesGroupAttribute(instr)
               ? EmitCollectiveGroup(instr)
               : EmitCollective(instr->async_wrapped_instruction());
  }
  return EmitAsyncStart(instr);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchAsyncDone(
    const HloInstruction* instr) {
  const bool is_synchronous_collective = ShouldEmitCollectiveSynchronously(
      instr->operand(0), ir_emitter_context_->debug_options());

  // Dispatch legacy typed done instructions first. Generic kAsyncDone
  // instructions are dispatched below according to the wrapped instruction.
  switch (instr->opcode()) {
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectivePermuteDone:
      return is_synchronous_collective
                 ? ThunkSequence::Empty()
                 : EmitAsyncDone(instr, instr->operand(0));
    case HloOpcode::kRecvDone:
      return DispatchRecvDone(Cast<HloRecvDoneInstruction>(instr));
    case HloOpcode::kSendDone:
      return DispatchSendDone(Cast<HloSendDoneInstruction>(instr));
    case HloOpcode::kAsyncDone:
      break;
    default:
      return Internal("Unsupported async done instruction: %s",
                      instr->ToString());
  }

  if (!instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    return is_synchronous_collective ? ThunkSequence::Empty()
                                     : EmitAsyncDone(instr, instr->operand(0));
  }

  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  switch (wrapped->opcode()) {
    // Complete a collective wrapped in generic async start/done. A collective
    // emitted synchronously has no corresponding completion thunk.
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllGather:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllToAll:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
      return is_synchronous_collective
                 ? ThunkSequence::Empty()
                 : EmitAsyncDone(instr, instr->operand(0));

    // Complete a fusion or call wrapped in generic async start/done.
    case HloOpcode::kFusion:
    case HloOpcode::kCall:
      return EmitAsyncDone(instr, instr->operand(0));

    // Select host or device completion for a wrapped recv.
    case HloOpcode::kRecv: {
      auto* recv = Cast<HloRecvInstruction>(wrapped);
      if (recv->is_host_transfer()) {
        return EmitHostRecvDone(instr, recv);
      }
      return EmitAsyncDone(instr, instr->operand(0));
    }

    // Select host or device completion for a wrapped send.
    case HloOpcode::kSend: {
      auto* send = Cast<HloSendInstruction>(wrapped);
      if (send->is_host_transfer()) {
        return EmitHostSendDone(instr, send);
      }
      return EmitAsyncDone(instr, instr->operand(0));
    }

    // Select host-execute or generic async completion for a wrapped custom
    // call.
    case HloOpcode::kCustomCall: {
      auto* custom_call = Cast<HloCustomCallInstruction>(wrapped);
      if (IsHostExecuteCustomCall(*custom_call)) {
        return EmitHostExecuteDone(instr, custom_call);
      }
      return EmitAsyncDone(instr, instr->operand(0));
    }

    default:
      return Internal("Unsupported async done wrapped instruction: %s",
                      HloOpcodeString(wrapped->opcode()));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchSend(
    const HloSendInstruction* instr) {
  if (instr->is_host_transfer()) {
    return EmitHostSend(instr);
  }

  ABSL_ASSIGN_OR_RETURN(ThunkSequence thunks, EmitSend(instr));
  if (IsImplicitAsyncSendRecvStart(instr)) {
    return EmitAsyncSendRecvStart(instr, std::move(thunks));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchSendDone(
    const HloSendDoneInstruction* instr) {
  return instr->is_host_transfer() ? EmitHostSendDone(instr, instr)
                                   : EmitSendDone(instr);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchRecv(
    const HloRecvInstruction* instr) {
  if (instr->is_host_transfer()) {
    return EmitHostRecv(instr);
  }

  ABSL_ASSIGN_OR_RETURN(ThunkSequence thunks, EmitRecv(instr));
  if (IsImplicitAsyncSendRecvStart(instr)) {
    return EmitAsyncSendRecvStart(instr, std::move(thunks));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::DispatchRecvDone(
    const HloRecvDoneInstruction* instr) {
  return instr->is_host_transfer() ? EmitHostRecvDone(instr, instr)
                                   : EmitRecvDone(instr);
}

Future<ThunkSequence> ThunkEmitter::DispatchCustomCall(
    const HloInstruction* hlo) {
  auto* custom_call = Cast<HloCustomCallInstruction>(hlo);

  if (IsCublasLtMatmul(*hlo)) {
    return EmitCublasLtMatmul(custom_call);
  }
  if (IsCublasLtMatmulF8(*hlo)) {
    return EmitCublasLtMatmulF8(custom_call);
  }
  if (IsCublasLtGroupedMatmul(*hlo)) {
    return EmitCublasLtGroupedMatmul(custom_call);
  }
  if (IsCublasLtMatmulMx(*hlo)) {
    return EmitCublasLtMatmulMx(custom_call);
  }
  if (IsCudnnConvolutionReorder(*hlo)) {
    return EmitConvolutionReorder(custom_call);
  }
  if (IsCustomCallToDnnNorm(*hlo)) {
    return EmitNorm(custom_call);
  }
  if (IsCustomCallTofMHA(*hlo) || IsCustomCallTofMHAF8(*hlo) ||
      IsCustomCallToBlockScaledDot(*hlo)) {
    return EmitCuDnn(custom_call);
  }
  if (IsCustomCallToPtxKernel(*hlo)) {
    return EmitPtxCustomCall(custom_call);
  }
  if (IsCustomCallToTopK(*hlo)) {
    return EmitTopKCustomCall(custom_call);
  }
  if (IsCustomCallToDnnConvolution(*hlo)) {
    return EmitConvolution(custom_call);
  }
  if (IsTriangularSolve(*hlo)) {
    return EmitTriangularSolveCustomCall(hlo);
  }
  // CUB sort is handled as a generic FFI custom call via CustomCallThunk.
  // See xla.gpu.ext.cub_sort_keys and xla.gpu.ext.cub_sort_pairs handlers.
  if (hlo->custom_call_target() == "PadToStatic") {
    return EmitPadToStatic(custom_call);
  }
  if (hlo->custom_call_target() == "SliceToDynamic") {
    return EmitSliceToDynamic(custom_call);
  }
  if (hlo->custom_call_target() == "__gpu$xla.gpu.triton") {
    // TODO(slebedev): Remove this after June 15th 2025.
    return EmitTritonCustomCall(custom_call);
  }
  if (hlo->custom_call_target() == kNopCustomCallTarget) {
    return ThunkSequence::Empty();
  }
  if (hlo->custom_call_target() == kPinCustomCallTarget ||
      hlo->custom_call_target() == kUnpinCustomCallTarget ||
      hlo->custom_call_target() == kCreateBufferCustomCallTarget) {
    return ThunkSequence::Empty();
  }
  if (hlo->custom_call_target() == "GetRngSeed") {
    return EmitRngSeed(hlo);
  }
  return EmitGenericCustomCall(custom_call);
}

Future<ThunkSequence> ThunkEmitter::DispatchLegacyCollectiveStart(
    const HloInstruction* instr) {
  const bool is_legacy_collective_start =
      HloPredicateIsOp<HloOpcode::kAllGatherStart, HloOpcode::kAllReduceStart,
                       HloOpcode::kCollectivePermuteStart>(instr);
  TF_RET_CHECK(is_legacy_collective_start);
  if (ShouldEmitCollectiveSynchronously(instr,
                                        ir_emitter_context_->debug_options())) {
    return EmitCollective(instr);
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<AsyncExecution> execution,
                   RegisterAsyncExecution(instr));
  return EmitCollective(instr).Map(
      [this, instr,
       execution = std::move(execution)](ThunkSequence thunks) mutable {
        return EmitAsyncStart(std::move(execution), instr, std::move(thunks));
      });
}

//===----------------------------------------------------------------------===//
// HLO-specific thunk emission
//===----------------------------------------------------------------------===//

absl::StatusOr<std::shared_ptr<AsyncExecution>>
ThunkEmitter::RegisterAsyncExecution(const HloInstruction* async_start) {
  // Register before starting nested thunk emission: instruction futures are
  // resolved concurrently, so the corresponding async-done may be emitted
  // while the nested-emission future is still pending.
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      async_start, ir_emitter_context_->GetNextThunkId());
  auto execution = std::make_shared<AsyncExecution>(std::move(info));
  auto [_, inserted] = hlo_async_executions_.emplace(async_start, execution);
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    async_start->ToString());
  }
  return execution;
}

void AppendThunkSequence(ThunkSequence& thunks,
                         ThunkSequence& additional_thunks) {
  thunks.insert(thunks.end(),
                std::make_move_iterator(additional_thunks.begin()),
                std::make_move_iterator(additional_thunks.end()));
}

ThunkSequence FlattenThunkSequence(std::vector<ThunkSequence>&& sequences) {
  ThunkSequence result;

  int total = 0;
  for (const ThunkSequence& seq : sequences) {
    total += seq.size();
  }
  result.reserve(total);

  for (ThunkSequence& seq : sequences) {
    AppendThunkSequence(result, seq);
  }
  return result;
}

absl::StatusOr<std::string> CanonicalGemmHlo(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto gpu_config, instr->backend_config<GpuBackendConfig>());

  auto* gemm_config = gpu_config.has_grouped_gemm_backend_config()
                          ? gpu_config.mutable_grouped_gemm_backend_config()
                                ->mutable_gemm_backend_config()
                          : gpu_config.mutable_gemm_backend_config();

  // Clear algorithm-specific fields from the cache key
  gemm_config->clear_selected_algorithm();
  gemm_config->set_autotune_workspace_size(0);
  return instr->ToString(HloPrintOptions::Fingerprint()) +
         BackendConfigWrapper(gpu_config).GetRawString();
}

ThunkEmitter::ThunkEmitter(
    IrEmitterContext* absl_nonnull ir_emitter_context,
    llvm_ir::LLVMCommandLineOptionsReleasableLock* absl_nonnull
        llvm_options_lock)
    : ir_emitter_context_(ir_emitter_context),
      send_recv_events_(std::make_shared<HostSendRecvAsyncEvents>()),
      call_graph_(CallGraph::Build(&ir_emitter_context->hlo_module())),
      constants_module_context_(std::make_unique<llvm::LLVMContext>()),
      constants_module_(ir_emitter_context_->CreateLLVMModule(
          absl::StrCat(ir_emitter_context_->hlo_module().name(), "_consts"),
          *constants_module_context_)),
      llvm_options_lock_(llvm_options_lock) {}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConstant(
    const HloConstantInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(DenseDataIntermediate content,
                   LiteralToXlaFormat(instr->literal()));

  int element_bytes =
      primitive_util::ByteWidth(instr->literal().shape().element_type());
  TF_RET_CHECK(content.span().size() % element_bytes == 0);
  // Treat packed constants as a byte constant.
  int num_elements = content.span().size() / element_bytes;

  std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSlice(instr, {}));

  // LLVM and PTXAS don't deal well with large constants, so we only emit very
  // small constants directly in LLVM IR.  Larger constants are emitted with
  // zero initializers in LLVM IR and are later overwritten when the PTX/CUBIN
  // is loaded.
  bool should_emit_initializer = num_elements <= 1;
  AppendGlobalConstant(constants_module_.get(), num_elements, element_bytes,
                       global_name, slice.index(), content,
                       should_emit_initializer);

  GpuExecutable::ConstantInfo info;
  info.symbol_name.assign(global_name);
  info.allocation_index = slice.index();
  if (!should_emit_initializer) {
    info.content = content;
  }

  ir_emitter_context_->constants().push_back(std::move(info));
  return ThunkSequence::Empty();
}

Future<ThunkSequence> ThunkEmitter::EmitConditional(
    const HloInstruction* instr) {
  std::vector<Future<ThunkSequence>> branch_thunks;
  branch_thunks.reserve(instr->branch_count());
  for (HloComputation* comp : instr->branch_computations()) {
    branch_thunks.emplace_back(EmitHloComputation(comp));
  }
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSlice(instr->operand(0), {}));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ShapedSlice shaped_slice{slice, instr->operand(0)->shape()};
  return tsl::JoinFutures(absl::MakeSpan(branch_thunks))
      .Map([info = std::move(info), shaped_slice = std::move(shaped_slice)](
               std::vector<ThunkSequence> branch_thunks) mutable {
        return ThunkSequence::Of<ConditionalThunk>(
            std::move(info), std::move(shaped_slice), std::move(branch_thunks));
      });
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
Future<ThunkSequence> ThunkEmitter::EmitPadToStatic(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  ABSL_ASSIGN_OR_RETURN(
      KernelDefinition<LlvmKernelSource> kernel_def,
      EmitPadToStaticLLVMIR(instr, ir_emitter_context_, kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ABSL_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](auto thunk) { return ThunkSequence::Of(std::move(thunk)); });
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
Future<ThunkSequence> ThunkEmitter::EmitSliceToDynamic(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));
  ABSL_ASSIGN_OR_RETURN(
      KernelDefinition<LlvmKernelSource> kernel_def,
      EmitSliceToDynamicLLVMIR(instr, ir_emitter_context_, kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ABSL_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](auto thunk) { return ThunkSequence::Of(std::move(thunk)); });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolution(
    const HloCustomCallInstruction* instr) {
  std::vector<ShapedSlice> operand_slices;
  operand_slices.reserve(instr->operand_count());
  for (const HloInstruction* operand : instr->operands()) {
    ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(operand, {}));
    operand_slices.push_back(slice);
  }

  // The first and the last element in the result tuple for a convolution are
  // always the result and the scratch buffer. It may have auxiliary results in
  // addition to the main result.
  std::vector<ShapedSlice> result_slices;
  for (int i = 0; i < instr->shape().tuple_shapes().size() - 1; i++) {
    ABSL_ASSIGN_OR_RETURN(ShapedSlice result_slice,
                     GetShapedSliceForHlo(instr, {i}));
    result_slices.push_back(result_slice);
  }

  ABSL_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  ABSL_ASSIGN_OR_RETURN(auto gpu_config, instr->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& backend_config =
      gpu_config.cudnn_conv_backend_config();
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice scratch_slice,
      GetAllocationSlice(
          instr,
          {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  GpuConvDescriptor descriptor = {kind,
                                  backend_config,
                                  instr->operand(0)->shape(),
                                  instr->operand(1)->shape(),
                                  instr->shape().tuple_shapes(0),
                                  static_cast<size_t>(scratch_slice.size()),
                                  instr->window(),
                                  instr->convolution_dimension_numbers(),
                                  instr->feature_group_count()};
  ABSL_ASSIGN_OR_RETURN(auto thunk,
                   ConvolutionThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           instr, ir_emitter_context_->GetNextThunkId()),
                       std::move(descriptor), std::move(operand_slices),
                       std::move(result_slices), scratch_slice));
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmul(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  ABSL_ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));
  bool has_matrix_bias = config.beta() != 0;

  TF_RET_CHECK(instr->operand_count() ==
               2 + int{has_matrix_bias} + int{has_vector_bias});

  ABSL_ASSIGN_OR_RETURN(bool has_aux_output,
                   xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));
  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ShapedSlice c;
  if (has_matrix_bias) {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(2)));
  } else {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ABSL_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    ABSL_ASSIGN_OR_RETURN(
        bias, GetShapedSliceForHlo(instr->operand(has_matrix_bias ? 3 : 2)));
  }

  std::optional<ShapedSlice> aux;
  if (has_aux_output) {
    ABSL_ASSIGN_OR_RETURN(aux, GetShapedSliceForHlo(instr, {1}));
  }

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().IsTuple() &&
      (instr->shape().tuple_shapes().size() - has_aux_output - 1)) {
    TF_RET_CHECK(
        (has_aux_output && instr->shape().tuple_shapes().size() == 3) ||
        (!has_aux_output && instr->shape().tuple_shapes().size() == 2));
    ABSL_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  ABSL_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  ABSL_ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  return ThunkSequence::Of<CublasLtMatmulThunk>(
      std::move(info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt, bias, aux, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, workspace_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulF8(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() > 3 && instr->operand_count() < 8);
  ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  ABSL_ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  ABSL_ASSIGN_OR_RETURN(bool has_aux_output,
                   xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ShapedSlice c;
  bool has_matrix_bias = config.beta() != 0;
  if (has_matrix_bias) {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(2)));
  } else {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ABSL_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  int a_scale_index = has_matrix_bias ? 3 : 2;
  ABSL_ASSIGN_OR_RETURN(ShapedSlice a_scale,
                   GetShapedSliceForHlo(instr->operand(a_scale_index)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b_scale,
                   GetShapedSliceForHlo(instr->operand(a_scale_index + 1)));

  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();
  bool is_fp8 = instr->shape().tuple_shapes(0).element_type() == F8E4M3FN ||
                instr->shape().tuple_shapes(0).element_type() == F8E5M2;
  // cublasLT requires c_scale/d_scale to be null when C/D is not
  // FP8. Currently, C cannot be FP8.
  std::optional<ShapedSlice> d_scale;
  if (is_cuda && is_fp8) {
    ABSL_ASSIGN_OR_RETURN(d_scale, GetShapedSliceForHlo(instr->operands().back()));
  }

  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    ABSL_ASSIGN_OR_RETURN(bias,
                     GetShapedSliceForHlo(instr->operand(a_scale_index + 2)));
  }

  std::optional<ShapedSlice> d_amax;
  if (config.damax_output()) {
    ABSL_ASSIGN_OR_RETURN(d_amax, GetShapedSliceForHlo(instr, {1}));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  TF_RET_CHECK(!has_aux_output);
  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().tuple_shapes().size() - config.damax_output() == 2) {
    ABSL_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ABSL_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ABSL_ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  return ThunkSequence::Of<CublasLtMatmulThunk>(
      std::move(info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt, bias, std::nullopt, a_scale, b_scale,
      std::nullopt, d_scale, d_amax, workspace_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtGroupedMatmul(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config =
      gpu_config.grouped_gemm_backend_config().gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  // Matrix bias and vector bias add extra operands
  bool has_matrix_bias = config.beta() != 0;
  ABSL_ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));
  TF_RET_CHECK(instr->operand_count() ==
               3 + int{has_matrix_bias} + int{has_vector_bias});

  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice group_sizes,
                   GetShapedSliceForHlo(instr->operand(2)));

  // Handle matrix bias if present
  ShapedSlice c;
  if (has_matrix_bias) {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(3)));
  } else {
    ABSL_ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ABSL_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  // Handle vector bias if present
  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    int bias_operand_index = has_matrix_bias ? 4 : 3;
    ABSL_ASSIGN_OR_RETURN(bias,
                     GetShapedSliceForHlo(instr->operand(bias_operand_index)));
  }

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().IsTuple() && (instr->shape().tuple_shapes().size() - 1)) {
    ABSL_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }
  ABSL_ASSIGN_OR_RETURN(
      auto gemm_config,
      GroupedGemmConfig::For(static_cast<const HloInstruction*>(instr),
                             ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  // Extract epilogue from backend config instead of hardcoding to kDefault
  ABSL_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ABSL_ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));

  return ThunkSequence::Of<CublasLtMatmulThunk>(
      std::move(info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      std::move(group_sizes), bias, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, workspace_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulMx(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 4);
  ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice a_scale,
                   GetShapedSliceForHlo(instr->operand(2)));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b_scale,
                   GetShapedSliceForHlo(instr->operand(3)));

  ABSL_ASSIGN_OR_RETURN(ShapedSlice c, GetShapedSliceForHlo(instr, output_index));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  ABSL_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().tuple_shapes().size() == 2) {
    ABSL_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(
            instr,
            {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));
  }

  ABSL_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ABSL_ASSIGN_OR_RETURN(std::string canonical_hlo, CanonicalGemmHlo(instr));
  return ThunkSequence::Of<CublasLtMatmulThunk>(
      std::move(info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*group_sizes=*/std::nullopt,
      /*bias=*/std::nullopt, /*aux=*/std::nullopt, a_scale, b_scale,
      /*c_scale=*/std::nullopt, /*d_scale=*/std::nullopt,
      /*d_amax=*/std::nullopt, workspace_buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolutionReorder(
    const HloCustomCallInstruction* instr) {
  bool has_bias = instr->operand_count() > 1;

  ABSL_ASSIGN_OR_RETURN(ShapedSlice filter_input,
                   GetShapedSliceForHlo(instr->operand(0)));

  ShapedSlice filter_output;
  std::optional<ConvolutionReorderThunk::BiasBuffers> biases;
  if (has_bias) {
    ABSL_ASSIGN_OR_RETURN(filter_output, GetShapedSliceForHlo(instr, {0}));

    ABSL_ASSIGN_OR_RETURN(ShapedSlice bias_input,
                     GetShapedSliceForHlo(instr->operand(1)));
    ABSL_ASSIGN_OR_RETURN(ShapedSlice bias_output, GetShapedSliceForHlo(instr, {1}));
    biases = {{bias_input, bias_output}};
  } else {
    ABSL_ASSIGN_OR_RETURN(filter_output, GetShapedSliceForHlo(instr));
  }

  ABSL_ASSIGN_OR_RETURN(auto thunk,
                   ConvolutionReorderThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           instr, ir_emitter_context_->GetNextThunkId()),
                       filter_input, filter_output, biases));
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNorm(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto const gpu_backend_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::CudnnNormBackendConfig& backend_config =
      gpu_backend_config.cudnn_norm_backend_config();

  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice x_slice,
                   GetAllocationSlice(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice scale_slice,
                   GetAllocationSlice(instr->operand(1)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice y_or_dx_slice,
                   GetAllocationSlice(instr, {0}));

  std::optional<BufferAllocation::Slice> bias_slice, expectation_slice,
      norm_factor_slice, dy_slice, dscale_slice, dbias_slice;

  if (backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER ||
      backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    ABSL_ASSIGN_OR_RETURN(bias_slice, GetAllocationSlice(instr->operand(2)));
  }
  if (backend_config.kind() ==
      xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    ABSL_ASSIGN_OR_RETURN(expectation_slice, GetAllocationSlice(instr, {1}));
    ABSL_ASSIGN_OR_RETURN(norm_factor_slice, GetAllocationSlice(instr, {2}));
  }
  if (backend_config.kind() == xla::gpu::CudnnNormBackendConfig::LAYER_BWD) {
    ABSL_ASSIGN_OR_RETURN(dy_slice, GetAllocationSlice(instr->operand(2)));
    ABSL_ASSIGN_OR_RETURN(expectation_slice, GetAllocationSlice(instr->operand(3)));
    ABSL_ASSIGN_OR_RETURN(norm_factor_slice, GetAllocationSlice(instr->operand(4)));
    ABSL_ASSIGN_OR_RETURN(dscale_slice, GetAllocationSlice(instr, {1}));
    ABSL_ASSIGN_OR_RETURN(dbias_slice, GetAllocationSlice(instr, {2}));
  }
  ABSL_ASSIGN_OR_RETURN(
      ShapedSlice scratch_slice,
      GetShapedSliceForHlo(
          instr,
          {static_cast<int64_t>(instr->shape().tuple_shapes().size()) - 1}));

  GpuNormDescriptor descriptor;
  descriptor.backend_config = backend_config;

  descriptor.x_shape = instr->operand(0)->shape();
  descriptor.scale_shape = instr->operand(1)->shape();
  descriptor.y_or_dx_shape = ShapeUtil::GetSubshape(instr->shape(), {0});
  descriptor.scratch_shape = scratch_slice.shape;

  if (backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER ||
      backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    descriptor.bias_shape = instr->operand(2)->shape();
  }
  if (backend_config.kind() ==
      xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    descriptor.expectation_shape = ShapeUtil::GetSubshape(instr->shape(), {1});
    descriptor.norm_factor_shape = ShapeUtil::GetSubshape(instr->shape(), {2});
  }
  if (backend_config.kind() == xla::gpu::CudnnNormBackendConfig::LAYER_BWD) {
    descriptor.dy_shape = instr->operand(2)->shape();
    descriptor.expectation_shape = instr->operand(3)->shape();
    descriptor.norm_factor_shape = instr->operand(4)->shape();
    descriptor.dscale_shape = ShapeUtil::GetSubshape(instr->shape(), {1});
    descriptor.dbias_shape = ShapeUtil::GetSubshape(instr->shape(), {2});
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<NormThunk> thunk,
      NormThunk::Create(Thunk::ThunkInfo::WithProfileAnnotation(
                            instr, ir_emitter_context_->GetNextThunkId()),
                        std::move(descriptor), x_slice, scale_slice,
                        y_or_dx_slice, bias_slice, expectation_slice,
                        norm_factor_slice, dy_slice, dscale_slice, dbias_slice,
                        scratch_slice.slice));
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCuDnn(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));
  ABSL_ASSIGN_OR_RETURN(const std::string fingerprint,
                   FingerprintWithBackendConfig<GpuBackendConfig>(*instr));
  // check if sdpa dropout is enabled
  std::optional<int64_t> dropout_seed = std::nullopt;
  if (MHACallHasDropout(instr->custom_call_target())) {
    ABSL_ASSIGN_OR_RETURN(const auto gpu_config,
                     instr->backend_config<xla::gpu::GpuBackendConfig>());
    dropout_seed = gpu_config.cudnn_fmha_backend_config().seed();
  }
  return ThunkSequence::Of<CuDnnThunk>(
      fingerprint,
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      kernel_arguments.GetArgumentShapedSlices(),
      kernel_arguments.GetArgumentOutputFlags(),
      /*should_memzero=*/IsCustomCallTofMHA(*instr), dropout_seed);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitPtxCustomCall(
    const HloCustomCallInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto thunk,
                   EmitPtxCustomKernelThunk(instr, ir_emitter_context_));
  return ThunkSequence::Of(std::move(thunk));
}

std::optional<BufferAllocation::Slice> ThunkEmitter::GetAllocationOverride(
    const HloInstruction* instr, const ShapeIndex& index) const {
  auto it = allocation_overrides_.find(instr);
  if (it == allocation_overrides_.end()) {
    return std::nullopt;
  }

  int64_t flat_idx = index.empty() ? 0 : index[0];
  if (flat_idx >= 0 && static_cast<size_t>(flat_idx) < it->second.size()) {
    return it->second[static_cast<size_t>(flat_idx)];
  }

  return std::nullopt;
}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSlice(
    const HloInstruction* instr, const ShapeIndex& index) const {
  if (std::optional<BufferAllocation::Slice> slice =
          GetAllocationOverride(instr, index)) {
    return *slice;
  }

  return ir_emitter_context_->buffer_assignment().GetUniqueSlice(instr, index);
}

absl::StatusOr<ShapedSlice> ThunkEmitter::GetShapedSliceForHlo(
    const HloInstruction* instr, const ShapeIndex& index) const {
  if (std::optional<BufferAllocation::Slice> slice =
          GetAllocationOverride(instr, index)) {
    return ShapedSlice{*slice, ShapeUtil::GetSubshape(instr->shape(), index)};
  }

  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSlice(instr, index));
  ABSL_ASSIGN_OR_RETURN(
      Shape shape,
      ir_emitter_context_->buffer_assignment().GetShapeForUniqueSlice(instr,
                                                                      index));
  return ShapedSlice{slice, shape};
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitGenericCustomCall(
    const HloCustomCallInstruction* instr) {
  const std::string& call_target_name = instr->custom_call_target();

  // Typed FFI custom calls is a replacement for legacy custom calls
  // with a rich type safe API.
  bool is_ffi_custom_call =
      instr->api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  using Slices = std::vector<NullableShapedSlice>;

  Slices operands;
  for (auto* operand : instr->operands()) {
    ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            operands.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          ABSL_ASSIGN_OR_RETURN(auto slice, GetAllocationSlice(operand, index));
          operands.push_back(ShapedSlice{slice, subshape});
          return absl::OkStatus();
        }));
  }

  Slices results;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          results.push_back(std::nullopt);
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        ABSL_ASSIGN_OR_RETURN(auto slice, GetAllocationSlice(instr, index));
        results.push_back(ShapedSlice{slice, subshape});
        return absl::OkStatus();
      }));

  // For XLA FFI handlers we decode opaque backend config into
  // attributes map at IR emission time, so that we do not need to
  // parse MLIR at run time. For FFI handlers backend config must be
  // a compatible MLIR dictionary.
  ffi::AttributesMap attributes;

  auto backend_config = instr->backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    VLOG(3) << "Unable to parse backend config for custom call: "
            << backend_config.status().message() << "\n"
            << "Fall back to parse the raw backend config str.";
  }

  auto ffi_thunk = [&]() -> absl::StatusOr<std::unique_ptr<Thunk>> {
    auto& called_computations = instr->called_computations();
    auto& backend_config_str =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().attributes()
            : instr->raw_backend_config_string();
    if (!backend_config_str.empty()) {
      mlir::Attribute attr = mlir::parseAttribute(
          backend_config_str, ir_emitter_context_->mlir_context());
      auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
      if (dict == nullptr) {
        return absl::InternalError(
            "Unsupported backend config. Expected a string "
            "parsable into "
            "dictionary attribute");
      }
      ABSL_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    }
    bool use_pdl = false;
    static constexpr absl::string_view kUsesPdl = "uses_pdl";
    if (ir_emitter_context_->debug_options().xla_gpu_enable_pdl()) {
      if (auto it = attributes.find(kUsesPdl); it != attributes.end()) {
        if (auto* scalar =
                std::get_if<xla::ffi::Scalar>(&it->second.AsVariant())) {
          if (auto* val = std::get_if<bool>(&scalar->AsVariant())) {
            use_pdl = *val;
          }
        }
      }
    }
    auto released_lock_keeper = llvm_options_lock_->TemporarilyReleaseLock();
    return CustomCallThunk::Create(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        call_target_name, std::move(operands), std::move(results),
        std::move(attributes),
        called_computations.empty() ? nullptr : called_computations[0],
        ir_emitter_context_->platform_name(),
        ir_emitter_context_->gpu_compute_capability(),
        /*execution_state=*/nullptr,
        ir_emitter_context_->cpu_target_machine_options(), use_pdl);
  };

  auto legacy_thunk = [&]() -> absl::StatusOr<std::unique_ptr<Thunk>> {
    std::string opaque =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().opaque()
            : instr->raw_backend_config_string();
    return LegacyCustomCallThunk::Create(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        call_target_name, std::move(operands), std::move(results),
        std::move(opaque), instr->api_version(),
        ir_emitter_context_->platform_name());
  };

  absl::StatusOr<std::unique_ptr<Thunk>> custom_call_thunk =
      is_ffi_custom_call ? ffi_thunk() : legacy_thunk();

  ThunkSequence thunks;
  if (custom_call_thunk.ok()) {
    thunks.push_back(std::move(custom_call_thunk.value()));
  }
  if (ir_emitter_context_->debug_options().xla_gpu_mock_custom_calls()) {
    // xla_gpu_mock_custom_calls=true means we won't emit thunks for all custom
    // call targets that couldn't be found.
    return thunks;
  }
  if (!custom_call_thunk.ok()) {
    return custom_call_thunk.status();
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFft(
    const HloFftInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                   GetAllocationSlice(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                   GetAllocationSlice(instr));
  return ThunkSequence::Of<FftThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr->fft_type(), instr->fft_length(),
      /*input_buffer=*/arg_slice,
      /*output_buffer=*/dest_slice,
      /*input_shape=*/instr->operand(0)->shape(),
      /*output_shape=*/instr->shape());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitTriangularSolveCustomCall(
    const HloInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 2);
  auto operands = instr->operands();
  TF_RET_CHECK(instr->shape().IsTuple() &&
               instr->shape().tuple_shapes().size() == 2);

  // We expect Fortran layout for everything other than the temp
  // buffer (the last operand).  Fortran layout is not XLA default
  // layout with elements 0 and 1 swapped.  For example instead of
  // default layout {3,2,1,0} we'd have Fortran layout {2,3,1,0}.
  auto has_fortran_layout = [](const Layout& layout) {
    int n = layout.minor_to_major().size();
    return layout.minor_to_major(0) == n - 2 &&
           layout.minor_to_major(1) == n - 1;
  };
  TF_RET_CHECK(has_fortran_layout(operands[0]->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(operands[1]->shape().layout()));
  TF_RET_CHECK(has_fortran_layout(instr->shape().tuple_shapes(0).layout()));

  ABSL_ASSIGN_OR_RETURN(ShapedSlice a_slice, GetShapedSliceForHlo(operands[0]));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice b_slice, GetShapedSliceForHlo(operands[1]));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice result_slice, GetShapedSliceForHlo(instr, {0}));
  ABSL_ASSIGN_OR_RETURN(ShapedSlice temp_slice, GetShapedSliceForHlo(instr, {1}));

  TriangularSolveOptions backend_config;
  auto& backend_config_str = instr->raw_backend_config_string();
  if (!backend_config_str.empty()) {
    ABSL_RETURN_IF_ERROR(
        tsl::HumanReadableJsonToProto(backend_config_str, &backend_config));
  }

  ThunkSequence thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output
  // if they aren't the same buffer.
  if (b_slice.slice != result_slice.slice) {
    thunks.Emplace<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        /*source_buffer=*/b_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(b_slice.shape));
  }

  thunks.Emplace<TriangularSolveThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      backend_config, a_slice, result_slice, temp_slice);

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    return thunks;
  }
  auto info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  // Don't repeat the annotation from inside thunks
  info.profile_annotation = {};
  return ThunkSequence::Of<SequentialThunk>(info, std::move(thunks));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitTopKCustomCall(
    const HloCustomCallInstruction* instr) {
  auto operands = instr->operands();
  const auto& shape = instr->shape();
  TF_RET_CHECK(operands.size() == 1)
      << "Expect only 1 operand for TopK custom call.";
  TF_RET_CHECK(shape.IsTuple())
      << "Expect TopK custom call to have tuple shape.";
  TF_RET_CHECK(shape.tuple_shapes().size() == 2)
      << "Expect TopK custom call shape to have exactly 2 "
         "sub-shapes.";

  auto data_shape = operands[0]->shape();
  auto top_elements_shape = shape.tuple_shapes()[0];
  auto indices_shape = shape.tuple_shapes()[1];

  TF_RET_CHECK(data_shape.dimensions().size() <= 2) << "Invalid input shape.";
  TF_RET_CHECK(indices_shape.element_type() == PrimitiveType::S32)
      << "Indices should be S32.";

  bool has_batch = data_shape.dimensions().size() == 2;
  auto [batch_size, n, k] =
      has_batch
          ? std::tuple<size_t, size_t, size_t>{data_shape.dimensions(0),
                                               data_shape.dimensions(1),
                                               top_elements_shape.dimensions(1)}
          : std::tuple<size_t, size_t, size_t>{
                1, data_shape.dimensions(0), top_elements_shape.dimensions(0)};

  // Prepare kernel arguments.
  ABSL_ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  auto dtype = data_shape.element_type();
  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();

  // Enable RAFT if TopK is_stable = false.
  bool use_raft = !hlo_instruction_utils::IsTopKStable(instr);

  if (is_cuda && use_raft) {
    // The heuristic for deciding when to use TopK Custom Kernel versus
    // Raft::matrix::select_k was developed as part of the initial research
    // in b/409009349.
    // CustomCall TopK requires k <= 16 and n >= 1024
    bool use_raft_select_k = false;
    if (dtype == PrimitiveType::F32) {
      use_raft_select_k =
          (n < 1024) || (n == 1024 && k > 12) || (n > 1024 && k >= 8);
    } else if (dtype == PrimitiveType::BF16) {
      use_raft_select_k = n < 1024 || k >= 8;
    }

    VLOG(3) << "EmitTopKCustomCall: dtype=" << dtype << ", n=" << n
            << ", k=" << k << ", use_raft_select_k=" << use_raft_select_k;

    Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
        instr, ir_emitter_context_->GetNextThunkId());
    if (use_raft_select_k) {
      return ThunkSequence::Of<SelectKThunk>(std::move(info), batch_size, n, k,
                                             dtype, kernel_arguments);
    }
  }

  auto wavefront_size =
      ir_emitter_context_->gpu_device_info().threads_per_warp();

  TF_RET_CHECK(k <= 16) << "CustomCall TopK requires k <= 16";
  // Load TopK custom kernel.
  ABSL_ASSIGN_OR_RETURN(CustomKernel kernel, kernel::topk::GetTopKKernel(
                                            "topk", dtype, n, k, batch_size,
                                            platform_name(), wavefront_size));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return ThunkSequence::Of<CustomKernelThunk>(
      std::move(info), std::move(kernel), kernel_arguments);
}

Future<ThunkSequence> ThunkEmitter::EmitTritonCustomCall(
    const HloCustomCallInstruction* instr) {
  BorrowedMlirContext borrowed_context =
      ir_emitter_context_->BorrowMlirContext();
  LoadMlirDialectsForTriton(**borrowed_context);
  TritonCall call = TritonCall::Parse(instr->raw_backend_config_string(),
                                      borrowed_context->get());
  auto call_zeroed_outputs = call.zeroed_outputs;
  auto generate =
      [this, &instr, borrowed_context = std::move(borrowed_context),
       call =
           std::move(call)]() mutable -> xla::Future<KernelReuseCache::Entry> {
    std::string kernel_name =
        ir_emitter_context_->GetSanitizedUniqueName(call.name);

    ABSL_ASSIGN_OR_RETURN(TritonKernelSource triton_source,
                     EmitTritonFrom(call, kernel_name, **borrowed_context));

    HloModule* hlo_module = instr->GetModule();

    BlockLevelParameters block_level_parameters;
    block_level_parameters.num_stages = call.num_stages;
    block_level_parameters.num_warps = call.num_warps;
    block_level_parameters.num_ctas = 1;
    block_level_parameters.global_scratch_memory_size =
        call.global_scratch_memory_size;
    block_level_parameters.is_tma_allowed = call.is_tma_allowed;
    block_level_parameters.waves_per_eu = call.waves_per_eu;

    return ir_emitter_context_->kernel_compiler()
        ->CompileTritonToLlvm(
            kernel_name, *hlo_module, ir_emitter_context_->gpu_device_info(),
            block_level_parameters, ir_emitter_context_->target_triple(),
            ir_emitter_context_->data_layout(), std::move(triton_source),
            std::move(borrowed_context), /*is_xla_fusion=*/false)
        .Map([kernel_name,
              kernel_impl_name = ir_emitter_context_->GetSanitizedUniqueName(
                  kernel_name + "_impl"),
              instr, call = std::move(call),
              kernel_compiler = ir_emitter_context_->kernel_compiler(),
              buffer_assignment = &ir_emitter_context_->buffer_assignment(),
              &gpu_device_info = ir_emitter_context_->gpu_device_info()](
                 TritonWrapperResult result) mutable
                 -> xla::Future<KernelReuseCache::Entry> {
          auto local_module =
              std::move(result.kernel_source).thread_safe_module();

          ABSL_ASSIGN_OR_RETURN(
              auto kernel_arguments,
              emitters::KernelArguments::Create(
                  *buffer_assignment, GetDefaultBufferAlignment(), instr));
          auto launch_dimensions = LaunchDimensions(
              se::BlockDim(call.grid_x, call.grid_y, call.grid_z),
              se::ThreadDim(call.num_warps *
                            gpu_device_info.threads_per_warp()));

          ABSL_ASSIGN_OR_RETURN(
              llvm::Function * kernel,
              RemoveUnusedTritonAbiArguments(
                  local_module.getModuleUnlocked(), kernel_name,
                  kernel_impl_name, call.global_scratch_memory_size > 0));

          AnnotateAttrsIfUnset(kernel_arguments, *kernel);
          ABSL_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
              gpu_device_info, launch_dimensions, kernel,
              local_module.getModuleUnlocked()));

          return kernel_compiler
              ->CompileToTargetBinary(LlvmKernelSource{std::move(local_module)})
              .Map([use_pdl = result.use_pdl, shmem_bytes = result.shmem_bytes,
                    launch_dimensions = std::move(launch_dimensions),
                    tma_metadata = result.tma_metadata,
                    kernel_name = std::move(kernel_name)](
                       const std::vector<uint8_t>& cubin) mutable {
                return KernelReuseCache::Entry{std::move(kernel_name),
                                               launch_dimensions,
                                               /*cluster_dim=*/std::nullopt,
                                               shmem_bytes,
                                               cubin,
                                               tma_metadata,
                                               use_pdl};
              });
        });
  };

  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  auto [status_or_entry, was_cached] =
      ir_emitter_context_->kernel_cache().GetWithStatus(
          instr->raw_backend_config_string(), generate);

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return status_or_entry.Map(
      [info = std::move(info), kernel_arguments = std::move(kernel_arguments),
       call_zeroed_outputs = std::move(call_zeroed_outputs)](
          const KernelReuseCache::Entry* entry) mutable
          -> absl::StatusOr<ThunkSequence> {
        ABSL_ASSIGN_OR_RETURN(CustomKernel custom_kernel,
                         kernel::CreateOwnedCubinCustomKernel(
                             entry->kernel_name, entry->binary,
                             kernel_arguments.args().size(),
                             entry->launch_dimensions.block_counts(),
                             entry->launch_dimensions.thread_counts_per_block(),
                             entry->shmem_bytes));
        return ThunkSequence::Of<CustomKernelThunk>(
            std::move(info), std::move(custom_kernel),
            std::move(kernel_arguments), entry->use_pdl, call_zeroed_outputs,
            entry->tma_metadata);
      });
}

Future<ThunkSequence> ThunkEmitter::EmitDynamicSliceCopyFusion(
    const HloFusionInstruction* instr, DynamicSliceCopyFusion copy) {
  std::vector<BufferAllocation> embedded_allocations;
  embedded_allocations.reserve(copy.parameters.size() + copy.results.size());

  for (const auto& param : copy.parameters) {
    embedded_allocations.emplace_back(embedded_allocations.size(),
                                      ShapeUtil::ByteSizeOf(param.slice_shape),
                                      0);
  }

  for (const auto& res : copy.results) {
    embedded_allocations.emplace_back(embedded_allocations.size(),
                                      ShapeUtil::ByteSizeOf(res.update_shape),
                                      0);
  }

  TF_RET_CHECK(copy.parameters.size() == 1);
  TF_RET_CHECK(copy.results.size() == 1);

  const Shape& copy_shape = copy.copy_operand->shape();
  int64_t byte_size = ShapeUtil::ByteSizeOf(copy_shape);
  BufferAllocation::Slice src_slice(&embedded_allocations[0], 0, byte_size);
  BufferAllocation::Slice dst_slice(
      &embedded_allocations[copy.parameters.size()], 0, byte_size);

  ThunkSequence embedded_thunks = ThunkSequence::Of<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      ShapedSlice{src_slice, copy_shape},
      ShapedSlice{dst_slice, copy.results[0].update_shape}, byte_size);

  std::vector<BufferAllocation::Slice> parameter_buffers;
  parameter_buffers.reserve(instr->operand_count());
  for (const auto* operand : instr->operands()) {
    ABSL_ASSIGN_OR_RETURN(parameter_buffers.emplace_back(),
                     GetAllocationSlice(operand));
  }

  std::vector<BufferAllocation::Slice> result_buffers;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      instr->shape(),
      [&](const Shape&, const ShapeIndex& index) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(result_buffers.emplace_back(),
                         GetAllocationSlice(instr, index));
        return absl::OkStatus();
      }));

  ABSL_RETURN_IF_ERROR(DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
      copy.results, parameter_buffers, result_buffers));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  bool verify_offsets =
      ir_emitter_context_->debug_options()
          .xla_gpu_experimental_dynamic_slice_fusion_verify_offsets();

  return ThunkSequence::Of<DynamicSliceFusionV2Thunk>(
      std::move(info), std::move(copy.parameters), std::move(copy.results),
      std::move(parameter_buffers), std::move(result_buffers),
      std::move(embedded_allocations), std::move(embedded_thunks),
      verify_offsets);
}

Future<ThunkSequence> ThunkEmitter::EmitStaticSliceCopyFusion(
    const HloFusionInstruction* instr, const StaticSliceCopyFusion& copy) {
  if (copy.parameter_number < 0 ||
      copy.parameter_number >= instr->operand_count()) {
    return Internal("Static slice copy parameter %d is out of range for %s",
                    copy.parameter_number, instr->ToString());
  }

  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                   GetAllocationSlice(instr->operand(copy.parameter_number)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_slice,
                   GetAllocationSlice(instr));

  int64_t byte_size = ShapeUtil::ByteSizeOf(copy.slice_shape);
  BufferAllocation::Slice src_slice(
      arg_slice.allocation(), arg_slice.offset() + copy.source_byte_offset,
      byte_size, arg_slice.element_type());

  return ThunkSequence::Of<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      ShapedSlice{src_slice, copy.slice_shape},
      ShapedSlice{dst_slice, instr->shape()}, byte_size);
}

Future<ThunkSequence> ThunkEmitter::EmitFusion(
    const HloFusionInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(std::optional<StaticSliceCopyFusion> static_copy,
                   AnalyzeStaticSliceCopyFusion(instr));
  if (static_copy.has_value()) {
    return EmitStaticSliceCopyFusion(instr, *static_copy);
  }

  ABSL_ASSIGN_OR_RETURN(std::optional<DynamicSliceCopyFusion> dynamic_copy,
                   AnalyzeDynamicSliceCopyFusion(instr));
  if (dynamic_copy.has_value()) {
    return EmitDynamicSliceCopyFusion(instr, std::move(*dynamic_copy));
  }

  analysis_garbage_collector_.push_back(
      std::make_unique<HloFusionAnalysis>(HloFusionAnalysis::Create(
          *instr, ir_emitter_context_->gpu_device_info())));
  const HloFusionAnalysis& fusion_analysis =
      *analysis_garbage_collector_.back();

  // Intercept DynamicSliceFusionV2 custom fusions.
  if (fusion_analysis.emitter_fusion_kind() ==
      HloFusionAnalysis::EmitterFusionKind::kCustomFusion) {
    auto custom_name = GetCustomFusionConfigName(instr);
    if (custom_name.has_value() &&
        *custom_name == kDynamicSliceFusionConfigName) {
      return EmitDynamicSliceFusionV2(instr);
    }
  }

  std::unique_ptr<FusionInterface> emitter = GetFusionEmitter(
      HloFusionInfo(fusion_analysis, instr,
                    &ir_emitter_context_->buffer_assignment(), *call_graph_));
  return emitter->Emit(*ir_emitter_context_, *instr);
}

Future<ThunkSequence> ThunkEmitter::EmitDynamicSliceFusionV2(
    const HloFusionInstruction* instr) {
  const HloComputation* body = instr->fused_instructions_computation();

  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  if (hero == nullptr) {
    return Internal("DynamicSliceFusionV2: no hero operation found");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<DynamicSliceFusion::Parameter> parameters,
                   DynamicSliceFusion::ResolveParameters(hero));
  ABSL_ASSIGN_OR_RETURN(std::vector<DynamicSliceFusion::Result> results,
                   DynamicSliceFusion::ResolveResults(hero));

  // parameter_buffers: one slice per fusion operand, indexed by parameter
  // number.
  std::vector<BufferAllocation::Slice> parameter_buffers;
  parameter_buffers.reserve(instr->operand_count());
  for (const auto* operand : instr->operands()) {
    ABSL_ASSIGN_OR_RETURN(parameter_buffers.emplace_back(),
                     GetAllocationSlice(operand));
  }

  // result_buffers: one entry per fusion output leaf in DFS order.
  std::vector<BufferAllocation::Slice> result_buffers;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      instr->shape(),
      [&](const Shape&, const ShapeIndex& index) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(result_buffers.emplace_back(),
                         GetAllocationSlice(instr, index));
        return absl::OkStatus();
      }));

  ABSL_RETURN_IF_ERROR(DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
      results, parameter_buffers, result_buffers));

  // embedded_allocations: synthetic allocations for the embedded thunk
  // executor. First N entries are for hero operands (one per Parameter),
  // then M entries for hero results (one per Result).
  std::vector<BufferAllocation> embedded_allocations;
  embedded_allocations.reserve(parameters.size() + results.size());

  for (const auto& param : parameters) {
    embedded_allocations.emplace_back(embedded_allocations.size(),
                                      ShapeUtil::ByteSizeOf(param.slice_shape),
                                      0);
  }

  for (const auto& res : results) {
    embedded_allocations.emplace_back(embedded_allocations.size(),
                                      ShapeUtil::ByteSizeOf(res.update_shape),
                                      0);
  }

  // Map hero operands and results to embedded allocations so the embedded
  // thunk emitter resolves the right buffers.
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<BufferAllocation::Slice>>
      overrides;

  for (int64_t i = 0; i < parameters.size(); ++i) {
    int64_t byte_size = ShapeUtil::ByteSizeOf(parameters[i].slice_shape);
    overrides[hero->operand(i)] = {
        BufferAllocation::Slice(&embedded_allocations[i], 0, byte_size)};
  }

  // One override slice per hero output leaf in DFS order.
  ShapeUtil::ForEachLeafShape(
      hero->shape(), [&](const Shape& subshape, const ShapeIndex&) {
        int64_t leaf_idx = overrides[hero].size();
        int64_t byte_size = ShapeUtil::ByteSizeOf(subshape);
        overrides[hero].push_back(BufferAllocation::Slice(
            &embedded_allocations[parameters.size() + leaf_idx], 0, byte_size));
      });

  auto overrides_cleanup = InstallAllocationOverrides(std::move(overrides));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  bool verify_offsets =
      ir_emitter_context_->debug_options()
          .xla_gpu_experimental_dynamic_slice_fusion_verify_offsets();

  return EmitHloInstruction(hero).Map(
      [info = std::move(info), results = std::move(results),
       result_buffers = std::move(result_buffers),
       parameters = std::move(parameters),
       parameter_buffers = std::move(parameter_buffers),
       embedded_allocations = std::move(embedded_allocations),
       verify_offsets](ThunkSequence embedded_thunks) mutable {
        return ThunkSequence::Of<DynamicSliceFusionV2Thunk>(
            std::move(info), std::move(parameters), std::move(results),
            std::move(parameter_buffers), std::move(result_buffers),
            std::move(embedded_allocations), std::move(embedded_thunks),
            verify_offsets);
      });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopy(
    const HloInstruction* instr) {
  TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
      instr->operand(0)->shape(), instr->shape(),
      Layout::Equal().MinorToMajorOnly()));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                   GetAllocationSlice(instr->operand(0)));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                   GetAllocationSlice(instr));
  return ThunkSequence::Of<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      /*source_buffer=*/ShapedSlice{src_buffer, instr->operand(0)->shape()},
      /*destination_buffer=*/ShapedSlice{dst_buffer, instr->shape()},
      /*mem_size=*/src_buffer.size());
}

absl::Status ThunkEmitter::AssertNonDeterminismIsOkay(
    const std::string& op_name) {
  if (RequireDeterminism(ir_emitter_context_->hlo_module().config())) {
    return Unimplemented(
        "HLO instruction %s does not have a deterministic "
        "implementation, "
        "but run-to-run determinism is required.",
        op_name);
  }
  return absl::OkStatus();
}

Future<ThunkSequence> ThunkEmitter::EmitWhile(const HloInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(auto config,
                   instr->backend_config<xla::WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count = std::nullopt;
  if (config.has_known_trip_count()) {
    trip_count = config.known_trip_count().n();
  }

  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Buffer slice holding while loop predicate.
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice pred,
                   GetAllocationSlice(condition->root_instruction(), {}));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  return std::move(tsl::JoinFutures(EmitHloComputation(condition),
                                    EmitHloComputation(body)))
      .Map([info = std::move(info), pred = pred, trip_count = trip_count](
               std::tuple<ThunkSequence, ThunkSequence> tuple) mutable {
        auto [cond_thunks, body_thunks] = std::move(tuple);
        return ThunkSequence::Of<WhileThunk>(
            std::move(info), std::move(pred), std::move(cond_thunks),
            std::move(body_thunks), trip_count);
      });
}

Future<ThunkSequence> ThunkEmitter::EmitCall(const HloInstruction* instr) {
  DCHECK_EQ(instr->opcode(), HloOpcode::kCall);
  DCHECK_EQ(instr->called_computations().size(), 1);
  const HloComputation* computation = instr->called_computations().front();
  return EmitHloComputation(computation);
}

Future<ThunkSequence> ThunkEmitter::EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  ABSL_ASSIGN_OR_RETURN(KernelDefinition<LlvmKernelSource> kernel_def,
                   EmitRngGetAndUpdateStateLLVMIR(instr, ir_emitter_context_,
                                                  kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ABSL_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](auto thunk) { return ThunkSequence::Of(std::move(thunk)); });
}

Future<ThunkSequence> ThunkEmitter::EmitSort(const HloSortInstruction* sort) {
  if (sort->is_stable()) {
    return Internal("Stable sort not supported. Did stable_sort_expander run?");
  }
  std::string op_name(sort->name());
  const Shape& keys_shape = sort->operand(0)->shape();
  ThunkSequence thunks;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    ShapeIndex shape_index =
        sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
    // We assume that the layout of all involved operands and
    // outputs is the same.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, sort->operand(i)->shape(),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));

    BufferAllocation::Slice destination_buffer;
    BufferAllocation::Slice source_address;

    // If possible, we share buffers. If that is not possible, we
    // need to copy the values, because the emitter does the sorting
    // in-place.
    ABSL_ASSIGN_OR_RETURN(destination_buffer, GetAllocationSlice(sort, shape_index));
    ABSL_ASSIGN_OR_RETURN(source_address, GetAllocationSlice(sort->operand(i), {}));

    if (destination_buffer != source_address) {
      // TODO(b/26783907): Figure out why we never seem to share
      // buffers for key/value sort.
      VLOG(2) << op_name << " requires initial D2D copy for operand " << i;
      thunks.Emplace<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              sort, ir_emitter_context_->GetNextThunkId()),
          /*source_buffer=*/
          ShapedSlice{source_address, sort->operand(i)->shape()},
          /*destination_buffer=*/
          ShapedSlice{destination_buffer, sort->operand(i)->shape()},
          ShapeUtil::ByteSizeOf(sort->operand(i)->shape()));
    }
  }

  return EmitBitonicSortLLVMIR(sort, ir_emitter_context_)
      .Map([thunks = std::move(thunks)](ThunkSequence sort_thunks) mutable {
        AppendThunkSequence(thunks, sort_thunks);
        return std::move(thunks);
      });
}

template <typename ThunkType>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReplicaOrPartitionId(
    const HloInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                   GetAllocationSlice(instr, {}));
  return ThunkSequence::Of<ThunkType>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      result_slice);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRngSeed(
    const HloInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                   GetAllocationSlice(instr, {}));
  return ThunkSequence::Of<RngSeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      result_slice);
}

Future<ThunkSequence> ThunkEmitter::EmitCollective(
    const HloInstruction* collective) {
  switch (collective->opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart: {
      auto* all_reduce = Cast<HloAllReduceInstruction>(collective);
      return EmitCollective<AllReduceThunk, HloAllReduceInstruction>(
          Thunk::kAllReduce, all_reduce, all_reduce->use_global_device_ids());
    }

    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart: {
      auto* all_gather = Cast<HloAllGatherInstruction>(collective);
      return EmitCollective<AllGatherThunk, HloAllGatherInstruction>(
          Thunk::kAllGather, all_gather, all_gather->use_global_device_ids());
    }

    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
      return EmitCollective<CollectivePermuteThunk,
                            HloCollectivePermuteInstruction>(
          Thunk::kCollectivePermute,
          Cast<HloCollectivePermuteInstruction>(collective), std::nullopt);

    case HloOpcode::kReduceScatter: {
      auto* reduce_scatter = Cast<HloReduceScatterInstruction>(collective);
      return EmitCollective<ReduceScatterThunk, HloReduceScatterInstruction>(
          Thunk::kReduceScatter, reduce_scatter,
          reduce_scatter->use_global_device_ids());
    }

    case HloOpcode::kAllToAll:
      return EmitCollective<AllToAllThunk, HloAllToAllInstruction>(
          Thunk::kAllToAll, Cast<HloAllToAllInstruction>(collective),
          std::nullopt);

    case HloOpcode::kRaggedAllToAll:
      return EmitCollective<RaggedAllToAllThunk, HloRaggedAllToAllInstruction>(
          Thunk::kRaggedAllToAll,
          Cast<HloRaggedAllToAllInstruction>(collective), std::nullopt);

    case HloOpcode::kCollectiveBroadcast:
      return EmitCollective<CollectiveBroadcastThunk,
                            HloCollectiveBroadcastInstruction>(
          Thunk::kCollectiveBroadcast,
          Cast<HloCollectiveBroadcastInstruction>(collective), std::nullopt);

    default:
      return Internal("Unsupported collective instruction: %s",
                      collective->ToString());
  }
}

Future<ThunkSequence> ThunkEmitter::EmitCollectiveGroup(
    const HloInstruction* instr) {
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return EmitHloComputation(instr->async_wrapped_computation())
      .Map([info = std::move(info)](ThunkSequence thunks) mutable {
        return ThunkSequence::Of<CollectiveGroupThunk>(
            std::move(info), Thunk::Kind::kGroup, std::move(thunks));
      });
}

template <typename CollectiveThunkType, typename HloInstType>
Future<ThunkSequence> ThunkEmitter::EmitCollective(
    Thunk::Kind kind, const HloInstType* inst,
    std::optional<bool> use_global_device_ids) {
  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  int64_t replica_count = hlo_config.replica_count();
  int64_t partition_count = hlo_config.num_partitions();
  int64_t operand_count = inst->operand_count();
  VLOG(2) << CollectiveThunkType::GetHloOpName()
          << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << operand_count;

  // A collective-broadcast may select its root rank at runtime, in which case
  // the last operand is a root-rank vector rather than data to broadcast.
  const bool has_dynamic_root = [](const HloInstType* inst) {
    if constexpr (std::is_same_v<HloInstType,
                                 HloCollectiveBroadcastInstruction>) {
      return inst->has_dynamic_root();
    }
    return false;
  }(inst);

  // CollectivePermuteThunk has its own degeneracy predicate and a different
  // constructor that requires replica/partition counts and permute options.
  constexpr bool is_collective_permute =
      std::is_same_v<CollectiveThunkType, CollectivePermuteThunk>;

  // Stash relevant information in CollectiveThunk::Buffer even if
  // we may not generate a CollectiveThunk.
  ABSL_ASSIGN_OR_RETURN(
      std::vector<CollectiveThunk::Buffer> buffers,
      GetCollectiveBuffers(ir_emitter_context_->buffer_assignment(), inst, kind,
                           has_dynamic_root));

  // A given collective op can be degenerate if across all groups
  // formed by it are singleton. In such a case, we don't need to do
  // any communication and we can just copy the input to the output.
  //
  // The only exception is RaggedAllToAll, which is not degenerate
  // even if all groups are singleton. In a singleton group case,
  // RaggedAllToAll becomes a generic equivalent of
  // DynamicUpdateSlice, except update size is not statically known.
  // This operation can not be expressed in term of standard HLO
  // instructions, so the best solution we have is to use NCCL thunk
  // even for degenerate cases.
  bool is_degenerate = false;
  if (kind != Thunk::Kind::kRaggedAllToAll) {
    if constexpr (is_collective_permute) {
      is_degenerate = CollectivePermuteThunk::IsDegenerate(inst, replica_count,
                                                           partition_count);
    } else {
      is_degenerate = GetCollectiveConfig(inst, use_global_device_ids)
                          .IsDegenerate(replica_count, partition_count);
    }
  }

  if (is_degenerate) {
    return EmitDegeneratedCollective(buffers, inst);
  }

  if constexpr (!is_collective_permute) {
    ABSL_RETURN_IF_ERROR(CollectiveThunkType::CheckImplementable(inst, replica_count,
                                                            partition_count));
  }

  auto info = Thunk::ThunkInfo::WithProfileAnnotation(
      inst, ir_emitter_context_->GetNextThunkId());
  Future<ThunkSequence> thunks;
  // For AllGather the strategy is now determined by the annotation written
  // by CollectiveKernelStrategyAnnotator.
  // `use_triton` was already set above by reading the backend_config
  // annotation.
  if constexpr (is_collective_permute) {
    thunks = ThunkSequence::Of<CollectivePermuteThunk>(
        info, inst, replica_count, partition_count, std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_collective_permute_mode(),
        ir_emitter_context_->debug_options()
            .xla_gpu_collective_permute_connected_components());
  } else if constexpr (std::is_same_v<CollectiveThunkType,
                                      CollectiveBroadcastThunk>) {
    // CollectiveBroadcastThunk needs the dynamic-root flag so it can treat
    // the trailing root-rank buffer specially at run time.
    thunks = ThunkSequence::Of<CollectiveThunkType>(
        info, inst, /*buffers=*/std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p(),
        has_dynamic_root);
  } else if constexpr (std::is_constructible_v<
                           CollectiveThunkType, Thunk::ThunkInfo,
                           decltype(inst),
                           std::vector<CollectiveThunk::Buffer>>) {
    thunks = ThunkSequence::Of<CollectiveThunkType>(
        info, inst, /*buffers=*/std::move(buffers));
  } else {
    thunks = ThunkSequence::Of<CollectiveThunkType>(
        info, inst, /*buffers=*/std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p());
  }
  return thunks;
}

template <typename HloInstType>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitDegeneratedCollective(
    std::vector<CollectiveThunk::Buffer>& buffers, const HloInstType* inst) {
  // Degenerate collectives are simply identity function. Buffer
  // assignment expects a copy, so that's what we do.
  ThunkSequence thunks;
  for (int64_t i = 0; i < buffers.size(); i++) {
    const Shape shape = inst->operand(i)->shape();
    thunks.Emplace<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            inst, ir_emitter_context_->GetNextThunkId()),
        ShapedSlice{buffers[i].source_buffer.slice, shape},
        ShapedSlice{buffers[i].destination_buffer.slice, shape},
        ShapeUtil::ByteSizeOf(shape));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitInfeed(
    const HloInfeedInstruction* instr) {
  // Infeed instruction returns a tuple containing the result data
  // and a token. We only need the result data to construct the
  // infeed thunk.
  std::vector<ShapedSlice> shaped_slices;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple() || subshape.IsToken()) return absl::OkStatus();
        if (subshape.IsArray()) {
          ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                           GetAllocationSlice(instr, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        instr->ToString(), index.ToString());
      }));

  return ThunkSequence::Of<InfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(shaped_slices));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitOutfeed(
    const HloOutfeedInstruction* instr) {
  // HLO outfeed instruction has 2 operands, the source and a token,
  // and a single token output.
  const HloInstruction* source = instr->operand(0);
  std::vector<ShapedSlice> shaped_slices;
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      source->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple()) return absl::OkStatus();
        if (subshape.IsArray()) {
          ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                           GetAllocationSlice(source, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        source->ToString(), index.ToString());
      }));

  return ThunkSequence::Of<OutfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(shaped_slices));
}

static absl::flat_hash_map<std::string, std::string> ConvertFrontendAttributes(
    const FrontendAttributes& attrs) {
  absl::flat_hash_map<std::string, std::string> result;
  // NOLINTNEXTLINE
  for (auto& [k, v] : attrs.map()) {
    result[k] = v;
  }
  return result;
}

static std::optional<GlobalDeviceId> DeviceConstraint(
    const HloInstruction* hlo) {
  if (hlo->has_sharding() && hlo->sharding().IsSingleDevice()) {
    return GlobalDeviceId(hlo->sharding().GetUniqueDevice());
  }
  return std::nullopt;
}

absl::StatusOr<bool> ShapeHasHostMemorySpace(Shape shape, int index,
                                             int host_memory_space) {
  return shape.tuple_shapes(index).has_layout() &&
         shape.tuple_shapes(index).layout().memory_space() == host_memory_space;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyStart(
    const HloCopyStartInstruction* copy_start_instr) {
  // copy-start has a tuple shape: {host, device, context},
  // or {device, host, context}.
  // Only the destination shape is needed to get the output buffer.
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                   GetAllocationSlice(copy_start_instr,
                                      /*index=*/{0}));

  const HloInstruction* src = copy_start_instr->operand(0);
  const Shape& input_shape = src->shape();
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                   GetAllocationSlice(src, {}));
  const Shape& shape = copy_start_instr->shape();
  CHECK(shape.IsTuple());
  auto host_memory_space =
      static_cast<int>(stream_executor::MemorySpace::kHost);
  ABSL_ASSIGN_OR_RETURN(bool is_dst_host_memory,
                   ShapeHasHostMemorySpace(shape, 0, host_memory_space));
  ABSL_ASSIGN_OR_RETURN(bool is_src_host_memory,
                   ShapeHasHostMemorySpace(shape, 1, host_memory_space));
  // H2H is not a supported copy-start variant.
  if (is_dst_host_memory && is_src_host_memory) {
    return absl::InternalError(
        absl::StrFormat("Copy-start %s has host memory space S(%d) on both "
                        "source and destination, which is unsupported",
                        copy_start_instr->ToString(),
                        static_cast<int>(stream_executor::MemorySpace::kHost)));
  }

  // Create the copy thunk with ThunkInfo derived from copy-start.
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      copy_start_instr, ir_emitter_context_->GetNextThunkId());

  std::unique_ptr<Thunk> copy_thunk;
  if (!is_dst_host_memory && !is_src_host_memory) {
    // D2D async copy: both source and destination reside in device memory.
    // The thunk is a raw memcpy, so source and destination layouts must
    // match; layout-changing copies must not reach this path.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        shape.tuple_shapes(0), input_shape, Layout::Equal().MinorToMajorOnly()))
        << "Copy-start " << copy_start_instr->ToString()
        << " has mismatched source/destination layouts";
    copy_thunk = std::make_unique<DeviceToDeviceCopyThunk>(
        info,
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape));
  } else if (is_dst_host_memory) {
    copy_thunk = std::make_unique<DeviceToHostCopyThunk>(
        info,
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape));
  } else {
    copy_thunk = std::make_unique<HostToDeviceCopyThunk>(
        info,
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape));
  }

  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  auto execution_stream_id =
      stream_assignment.GetExecutionStreamId(copy_start_instr);

  // If copy-start is not a scope-start operation, the copy is synchronous.
  if (!execution_stream_id.ok()) {
    return ThunkSequence::Of(std::move(copy_thunk));
  }

  // Wrap the copy thunk in an AsyncStartThunk for asynchronous execution.
  ThunkSequence nested_thunks;
  nested_thunks.push_back(std::move(copy_thunk));

  auto start_thunk = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          copy_start_instr, ir_emitter_context_->GetNextThunkId()),
      *execution_stream_id, std::move(nested_thunks));

  auto [it, inserted] = hlo_async_executions_.emplace(
      copy_start_instr, start_thunk->async_execution());
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    copy_start_instr->ToString());
  }

  return ThunkSequence::Of(std::move(start_thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyDone(
    const HloInstruction* instr) {
  const HloInstruction* copy_start_instr = instr->operand(0);
  CHECK(copy_start_instr->opcode() == HloOpcode::kCopyStart);

  // If the copy-start was asynchronous, emit an AsyncDoneThunk.
  auto it = hlo_async_executions_.find(copy_start_instr);
  if (it != hlo_async_executions_.end()) {
    return ThunkSequence::Of<AsyncDoneThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        it->second);
  }

  // Synchronous copy: copy-done is a no-op.
  return ThunkSequence::Empty();
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSend(
    const HloSendInstruction* instr) {
  TF_RET_CHECK(!instr->is_host_transfer());

  const HloInstruction* src = instr->operand(0);
  ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(src, {}));

  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  const int64_t replica_count = hlo_config.replica_count();
  const int64_t partition_count = hlo_config.num_partitions();
  const int64_t memory_space =
      instr->shape().IsTuple()
          ? instr->shape().tuple_shapes(0).layout().memory_space()
          : instr->shape().layout().memory_space();

  const CollectiveThunk::Buffer buffer = {
      /*element_count=*/ShapeUtil::ElementsIn(src->shape()),
      /*source_buffer=*/slice,
      /*destination_buffer=*/slice,
      /*source_memory_space=*/memory_space,
      /*destination_memory_space=*/memory_space};
  return ThunkSequence::Of<SendThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr, replica_count, partition_count, buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRecv(
    const HloRecvInstruction* instr) {
  TF_RET_CHECK(!instr->is_host_transfer());
  TF_RET_CHECK(instr->shape().IsTuple());

  ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(instr, {0}));

  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  const int64_t replica_count = hlo_config.replica_count();
  const int64_t partition_count = hlo_config.num_partitions();
  const int64_t memory_space =
      instr->shape().tuple_shapes(0).layout().memory_space();

  const CollectiveThunk::Buffer buffer = {
      /*element_count=*/ShapeUtil::ElementsIn(instr->shape().tuple_shapes(0)),
      /*source_buffer=*/slice,
      /*destination_buffer=*/slice,
      /*source_memory_space=*/memory_space,
      /*destination_memory_space=*/memory_space};
  return ThunkSequence::Of<RecvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr, replica_count, partition_count, buffer);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSendDone(
    const HloSendDoneInstruction* instr) {
  TF_RET_CHECK(!instr->is_host_transfer());
  return EmitAsyncDone(instr, FindCanonicalSendRecvStartOp(instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRecvDone(
    const HloRecvDoneInstruction* instr) {
  TF_RET_CHECK(!instr->is_host_transfer());
  return EmitAsyncDone(instr, FindCanonicalSendRecvStartOp(instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostSend(
    const HloSendInstruction* instr) {
  TF_RET_CHECK(instr->is_host_transfer());

  const HloInstruction* src = instr->operand(0);
  ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(src, {}));

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send instruction");
  }

  return ThunkSequence::Of<HostSendThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      src->shape(), slice.slice, *instr->channel_id(), send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostRecv(
    const HloRecvInstruction* instr) {
  TF_RET_CHECK(instr->is_host_transfer());
  TF_RET_CHECK(instr->shape().IsTuple());

  ABSL_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(instr, {0}));

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv instruction");
  }

  return ThunkSequence::Of<HostRecvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr->shape().tuple_shapes()[0], slice.slice, *instr->channel_id(),
      send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostSendDone(
    const HloInstruction* done, const HloSendRecvInstruction* host_transfer) {
  TF_RET_CHECK(host_transfer->is_host_transfer());
  if (!host_transfer->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send done instruction");
  }

  return ThunkSequence::Of<HostSendDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          done, ir_emitter_context_->GetNextThunkId()),
      *host_transfer->channel_id(), send_recv_events_,
      DeviceConstraint(host_transfer));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostRecvDone(
    const HloInstruction* done, const HloSendRecvInstruction* host_transfer) {
  TF_RET_CHECK(host_transfer->is_host_transfer());
  if (!host_transfer->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv done instruction");
  }

  return ThunkSequence::Of<HostRecvDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          done, ir_emitter_context_->GetNextThunkId()),
      *host_transfer->channel_id(), send_recv_events_,
      DeviceConstraint(host_transfer));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostExecuteStart(
    const HloInstruction* async_start,
    const HloCustomCallInstruction* host_execute) {
  TF_RET_CHECK(IsHostExecuteCustomCall(*host_execute));

  std::unique_ptr<HloModule> hlo_module =
      ExtractComputationIntoNewModule(*host_execute->called_computation());

  // All offloaded computations are marked as host computations from the
  // perspective of the GPU backend. Since these will execute on the main
  // thread from the CPU backend perspective, mark them as such.
  for (auto* computation : hlo_module->computations()) {
    computation->SetExecutionThread(HloInstruction::kMainExecutionThread);
  }

  absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4> operand_slices;
  for (HloInstruction* operand : host_execute->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      ABSL_ASSIGN_OR_RETURN(auto slice,
                       ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                           operand, indexed.index));
      operand_slices.push_back({slice, indexed.shape});
    }
  }

  absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4> result_slices;
  for (auto& indexed : ShapeUtil::GetLeafShapes(host_execute->shape())) {
    ABSL_ASSIGN_OR_RETURN(auto slice,
                     ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                         host_execute, indexed.index));
    result_slices.push_back({slice, indexed.shape});
  }

  HostOffloadingExecutableProto host_offloading_executable_proto;
  *host_offloading_executable_proto.mutable_hlo_module() =
      hlo_module->ToProto();
  host_offloading_executable_proto.set_executable_type(
      HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT);

  ABSL_ASSIGN_OR_RETURN(auto thunk,
                   HostExecuteStartThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           async_start, ir_emitter_context_->GetNextThunkId()),
                       std::move(host_offloading_executable_proto),
                       std::move(operand_slices), std::move(result_slices)));

  auto [it, inserted] = GetInstructionToHostExecuteAsyncEvents().emplace(
      host_execute, thunk->async_events());
  if (!inserted) {
    return Internal(
        "Async events already exist for host offloading custom call %s.",
        host_execute->ToString());
  }
  return ThunkSequence::Of(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHostExecuteDone(
    const HloInstruction* async_done,
    const HloCustomCallInstruction* host_execute) {
  TF_RET_CHECK(IsHostExecuteCustomCall(*host_execute));

  auto it = GetInstructionToHostExecuteAsyncEvents().find(host_execute);
  TF_RET_CHECK(it != GetInstructionToHostExecuteAsyncEvents().end())
      << "could not find async events for host execute operation";
  return ThunkSequence::Of<HostExecuteDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          async_done, ir_emitter_context_->GetNextThunkId()),
      it->second);
}

Future<ThunkSequence> ThunkEmitter::EmitAsyncStart(
    const HloInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<AsyncExecution> execution,
                   RegisterAsyncExecution(instr));

  Future<ThunkSequence> nested =
      HasCollectivesGroupAttribute(instr)
          ? EmitCollectiveGroup(instr)
          : EmitHloComputation(instr->async_wrapped_computation());

  return std::move(nested).Map([this, instr, execution = std::move(execution)](
                                   ThunkSequence thunks) mutable {
    return EmitAsyncStart(std::move(execution), instr, std::move(thunks));
  });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncStart(
    std::shared_ptr<AsyncExecution> execution,
    const HloInstruction* async_start, ThunkSequence thunks) {
  const ExecutionStreamAssignment& streams =
      ir_emitter_context_->execution_stream_assignment();
  ABSL_ASSIGN_OR_RETURN(ExecutionStreamId stream_id,
                   streams.GetExecutionStreamId(async_start));

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      async_start, execution->start_thunk_id());
  return ThunkSequence::Of<AsyncStartThunk>(
      std::move(info), stream_id, std::move(thunks), std::move(execution));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncSendRecvStart(
    const HloSendRecvInstruction* async_start, ThunkSequence thunks) {
  // Device send/recv outside an async computation can pipeline multiple start
  // thunks through one AsyncExecution. They use a canonical owner so every
  // pipelined start shares the same execution.
  const HloInstruction* owner = FindCanonicalSendRecvStartOp(async_start);

  const ExecutionStreamAssignment& streams =
      ir_emitter_context_->execution_stream_assignment();
  ABSL_ASSIGN_OR_RETURN(ExecutionStreamId stream_id,
                   streams.GetExecutionStreamId(owner));

  if (auto it = hlo_async_executions_.find(owner);
      it != hlo_async_executions_.end()) {
    Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
        async_start, ir_emitter_context_->GetNextThunkId());
    return ThunkSequence::Of<AsyncStartThunk>(std::move(info), stream_id,
                                              std::move(thunks), it->second);
  }

  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      async_start, ir_emitter_context_->GetNextThunkId());
  auto [it, inserted] = hlo_async_executions_.emplace(
      owner, std::make_shared<AsyncExecution>(info));
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    owner->ToString());
  }

  return ThunkSequence::Of<AsyncStartThunk>(std::move(info), stream_id,
                                            std::move(thunks), it->second);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncDone(
    const HloInstruction* done, const HloInstruction* start) {
  auto it = hlo_async_executions_.find(start);
  TF_RET_CHECK(it != hlo_async_executions_.end())
      << "could not find async execution for start operation";
  return ThunkSequence::Of<AsyncDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          done, ir_emitter_context_->GetNextThunkId()),
      it->second);
}

Future<ThunkSequence> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* hlo) {
  switch (hlo->opcode()) {
    // Legacy non-async-wrapped collective-start operations.
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCollectivePermuteStart:
      return DispatchLegacyCollectiveStart(hlo);

    // Legacy non-async-wrapped collective-done operations.
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectivePermuteDone:
      return DispatchAsyncDone(hlo);

    // Synchronous collective operations (async execution if needed added by
    // wrapping into generic async-start/async-done).
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kReduceScatter:
      return EmitCollective(hlo);

    // Generic async start/done wrapping asynchronous computation (operation).
    case HloOpcode::kAsyncStart:
      return DispatchAsyncStart(hlo);
    case HloOpcode::kAsyncDone:
      return DispatchAsyncDone(hlo);

    // Send/recv and their done operations dispatch first by
    // `is_host_transfer()`. Device transfer start emission then depends on
    // whether it is inside an async computation.
    case HloOpcode::kSend:
      return DispatchSend(Cast<HloSendInstruction>(hlo));
    case HloOpcode::kSendDone:
      return DispatchSendDone(Cast<HloSendDoneInstruction>(hlo));
    case HloOpcode::kRecv:
      return DispatchRecv(Cast<HloRecvInstruction>(hlo));
    case HloOpcode::kRecvDone:
      return DispatchRecvDone(Cast<HloRecvDoneInstruction>(hlo));

    case HloOpcode::kCall:
      return EmitCall(hlo);
    case HloOpcode::kConditional:
      return EmitConditional(hlo);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(hlo));
    case HloOpcode::kCustomCall:
      return DispatchCustomCall(hlo);
    case HloOpcode::kFusion:
      return EmitFusion(Cast<HloFusionInstruction>(hlo));
    case HloOpcode::kCopy:
      return EmitCopy(hlo);
    case HloOpcode::kInfeed:
      return EmitInfeed(Cast<HloInfeedInstruction>(hlo));
    case HloOpcode::kOutfeed:
      return EmitOutfeed(Cast<HloOutfeedInstruction>(hlo));
    case HloOpcode::kPartitionId:
      return EmitReplicaOrPartitionId<PartitionIdThunk>(hlo);
    case HloOpcode::kFft:
      return EmitFft(Cast<HloFftInstruction>(hlo));

    case HloOpcode::kReplicaId:
      return EmitReplicaOrPartitionId<ReplicaIdThunk>(hlo);
    case HloOpcode::kRngGetAndUpdateState:
      return EmitRngGetAndUpdateState(
          Cast<HloRngGetAndUpdateStateInstruction>(hlo));

    case HloOpcode::kSort:
      return EmitSort(Cast<HloSortInstruction>(hlo));
    case HloOpcode::kWhile:
      return EmitWhile(hlo);
    case HloOpcode::kCopyStart:
      return EmitCopyStart(Cast<HloCopyStartInstruction>(hlo));
    case HloOpcode::kCopyDone:
      return EmitCopyDone(hlo);

    // HLO module is already scheduled, so instructions for ordering
    // are noops.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    // We don't need to emit thunks for these operations because
    // their semantics are encoded by buffers.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return ThunkSequence::Empty();
    default:
      return Internal("Unsupported instruction opcode: %s",
                      HloOpcodeString(hlo->opcode()));
  }
  return Internal("Unhandled HLO instruction");
}

Future<ThunkSequence> ThunkEmitter::EmitHloEntryComputation(
    const HloModule* module) {
  return EmitHloComputation(module->entry_computation());
}

Future<ThunkSequence> ThunkEmitter::EmitHloComputation(
    const HloComputation* computation) {
  const HloSchedule& schedule = computation->parent()->schedule();
  const HloModule* hlo_module = schedule.module();
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_command_buffer_scheduling_mode() ==
      DebugOptions::CONCURRENT_REGIONS) {
    if (concurrent_regions_ordering_.count(hlo_module) == 0) {
      concurrent_regions_ordering_[hlo_module] =
          std::make_unique<ConcurrentRegionsHloOrdering>(schedule);
    }
  }
  if (!schedule.is_computation_scheduled(computation)) {
    return Internal("Sequence not found for computation: %s",
                    computation->name());
  }
  const std::vector<HloInstruction*>& instructions =
      schedule.sequence(computation).instructions();
  std::vector<Future<ThunkSequence>> futures(instructions.size());
  for (int i = 0; i < instructions.size(); i++) {
    futures[i] = EmitHloInstruction(instructions[i]);
  }

  return tsl::JoinFutures(absl::MakeSpan(futures))
      .Map([&instructions,
            &concurrent_regions_ordering = concurrent_regions_ordering_,
            hlo_module](std::vector<ThunkSequence> sequences) {
        absl::flat_hash_map<const HloInstruction*, Thunk*> instr_to_thunk;
        for (int i = 0; i < instructions.size(); i++) {
          const HloInstruction* instr = instructions[i];
          ThunkSequence& thunks = sequences[i];
          if (!thunks.empty()) {
            instr_to_thunk[instr] = thunks.back().get();
          }
          // Set the concurrent region id for the thunks, if it exists.
          if (concurrent_regions_ordering.count(hlo_module)) {
            auto concurrent_region_id =
                concurrent_regions_ordering.at(hlo_module)
                    ->GetConcurrentRegionId(instr);
            for (auto& thunk : thunks) {
              if (concurrent_region_id.has_value()) {
                thunk->set_concurrent_region_id(concurrent_region_id.value());
              }
            }
          }
        }

        return FlattenThunkSequence(std::move(sequences));
      });
}

}  // namespace xla::gpu

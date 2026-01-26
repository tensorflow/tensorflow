/*Copyright 2022 The OpenXLA Authors.

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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
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
#include "xla/backends/gpu/codegen/llvm/llvm_emitter.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/collective_metadata_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/convolution_filter_thunk.pb.h"
#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"
#include "xla/backends/gpu/runtime/convolution_thunk.h"
#include "xla/backends/gpu/runtime/copy_done_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/cub_sort_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/device_to_host_copy_thunk.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/host_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/infeed_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/norm_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_recv_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_send_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/select_k_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/topk.h"
#include "xla/backends/gpu/runtime/triangular_solve_thunk.h"
#include "xla/backends/gpu/runtime/wait_for_streams_thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/ffi/attribute_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_traversal.h"
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
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/custom_kernel_emitter.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_norm_runner.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/gpu/triton_call.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/human_readable_json.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

struct EmitCollectiveResult {
  std::unique_ptr<CollectiveKernelThunk> thunk;
  std::unique_ptr<llvm::Module> llvm_module;
};

// TODO: move into a host_execute specific file.
bool IsHostExecuteCustomCall(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() ==
             "HostExecute";  // TODO: this constant string should be shared with
                             // the TPU one
}

template <typename ThunkType>
static constexpr bool kRequiresCollectiveKernelThunk =
    std::is_constructible_v<ThunkType, Thunk::ThunkInfo,
                            const HloAllReduceInstruction*,
                            std::vector<CollectiveThunk::Buffer>,
                            std::unique_ptr<CollectiveKernelThunk>,
                            /*p2p_memcpy_enabled=*/bool>;

// The signature of this function would change to absl::Status once we lift the
// CollectiveKernelThunk out as a top level thunk. It would then become a member
// function of ThunkEmitter.
// As it stands now the collective kernel thunk is wrapped inside other
// collective thunks such as AllReduceStart. So this function is only
// responsible for emitting the collective kernel thunk and its dependencies.
absl::StatusOr<EmitCollectiveResult> EmitCollectiveKernelThunk(
    IrEmitterContext* ir_emitter_context, const CallGraph* call_graph,
    Thunk::ThunkInfo thunk_info, std::vector<CollectiveThunk::Buffer> buffers,
    const HloAllReduceInstruction* instr, const AllReduceConfig& config) {
  std::unique_ptr<HloModule> fused_module =
      NewModuleWithFusion(instr, HloInstruction::FusionKind::kLoop);
  HloFusionInstruction* fusion_instr = Cast<HloFusionInstruction>(
      fused_module->entry_computation()->root_instruction());
  const se::DeviceDescription& device_info =
      ir_emitter_context->gpu_device_info();
  const auto make_thunk = [&](absl::string_view kernel_name,
                              int32_t shmem_bytes,
                              std::unique_ptr<llvm::Module> local_module) {
    return EmitCollectiveResult{
        std::make_unique<CollectiveKernelThunk>(
            thunk_info, config.config, config.reduction_kind,
            /*is_async=*/!IsGPUSyncCollective(*instr), std::move(buffers),
            /*is_collective_kernel_enabled=*/
            instr->GetModule()
                ->config()
                .debug_options()
                .xla_gpu_unsupported_use_all_reduce_one_shot_kernel(),
            /*kernel_name=*/kernel_name,
            /*shmem_bytes=*/shmem_bytes,
            /*is_multimem_enabled=*/false),
        std::move(local_module)};
  };
  TF_ASSIGN_OR_RETURN(bool did_set_config, TrySetGpuBackendConfigForCollective(
                                               device_info, fusion_instr));
  if (!did_set_config) {
    return make_thunk(/*kernel_name=*/"", 0, nullptr);
  }
  const HloFusionAnalysis fusion_analysis =
      HloFusionAnalysis::Create(*fusion_instr, device_info);
  auto emitter = std::make_unique<TritonFusion>(fusion_analysis);
  TritonFusion::EmitResult result;
  {
    XLA_SCOPED_LOGGING_TIMER("Emit collective kernel thunk");
    TF_ASSIGN_OR_RETURN(std::vector<Shape> unmanaged_arguments,
                        GetCollectiveUnmanagedKernelArguments(fusion_instr));
    TF_ASSIGN_OR_RETURN(
        result, emitter->Emit(*ir_emitter_context, *fusion_instr,
                              /*instr_override=*/instr, unmanaged_arguments));
  }
  return make_thunk(result.kernel_thunk->kernel_name(),
                    result.kernel_thunk->shmem_bytes(),
                    std::move(result.llvm_module));
}

// If the fusion instruction is a dynamic-slice-fusion instruction,
// with a collective hero operation, then this function returns the
// collective operation. Returns std::nullopt otherwise.
std::optional<const HloInstruction*> GetCollectiveHeroForDynamicSliceFusion(
    const HloFusionInstruction* instruction) {
  if (!IsDynamicSliceFusion(instruction)) {
    return std::nullopt;
  }
  return HloBfsFindIf(
      {instruction->fused_instructions_computation()->root_instruction()},
      [](const HloInstruction* instr) { return IsCollective(instr); });
}

// Find the canonical send/recv start op for one of send, recv,
// send-done, or recv-done. For trivial cases send/recv and
// send-done/recv-done come in pairs and the canonical start op is
// the send/recv op of the pair. If send/recv is partially
// pipelined, we will use the send/recv leading into the while loop
// as the canonical start op, which will serve as a key for the
// async events.
//
// Example:
// ```
// send_ctx = send(src, ...)  <-- canonical start op
// send_ctx_final = while(send_ctx) {
//   send_ctx_in = parameter(0)
//   send-done(send_ctx_in)
//   ...
//   ROOT send_ctx_out = send(next_src, ...)
// }
// send-done(send_ctx_final)
// ```
static const HloInstruction* FindCanonicalSendRecvStartOp(
    const HloInstruction* inst) {
  CHECK(inst->opcode() == HloOpcode::kSend ||
        inst->opcode() == HloOpcode::kRecv ||
        inst->opcode() == HloOpcode::kSendDone ||
        inst->opcode() == HloOpcode::kRecvDone);
  // If the instruction is wrapped in an async computation, return
  // the instruction itself.
  if (inst->parent()->IsAsyncComputation()) {
    return inst;
  }

  // Find container while loop and index for the send/recv case or
  // return canonical start op directly.
  const HloInstruction* while_op = nullptr;
  int64_t i = -1;
  if (inst->opcode() == HloOpcode::kSend ||
      inst->opcode() == HloOpcode::kRecv) {
    CHECK_EQ(inst->users().size(), 1);
    const HloInstruction* unique_user = inst->users().front();

    // Return send/recv inst directly if this is a simple send/recv
    // pair.
    if (unique_user->opcode() == HloOpcode::kSendDone ||
        unique_user->opcode() == HloOpcode::kRecvDone) {
      return inst;
    }

    // Find while loop and index, otherwise.
    CHECK(unique_user->opcode() == HloOpcode::kTuple ||
          unique_user->opcode() == HloOpcode::kWhile);
    if (unique_user->IsRoot()) {
      // send/recv op in the loop body.
      auto maybe_while_op =
          unique_user->parent()->GetUniqueCaller(HloOpcode::kWhile);
      CHECK(maybe_while_op);
      while_op = *maybe_while_op;
      i = unique_user->operand_index(inst);
    } else {
      // send/recv leading into the loop.
      CHECK_EQ(unique_user->users().size(), 1);
      CHECK(unique_user->users().front()->opcode() == HloOpcode::kWhile);
      while_op = unique_user->users().front();
      i = unique_user->operand_index(inst);
    }
  }

  // Find container while loop and index for the send-done/recv-done
  // case or return canonical start op directly.
  if (inst->opcode() == HloOpcode::kSendDone ||
      inst->opcode() == HloOpcode::kRecvDone) {
    const HloInstruction* operand = inst->operand(0);

    // Return send/recv inst directly if this is a simple send/recv
    // pair.
    if (operand->opcode() == HloOpcode::kSend ||
        operand->opcode() == HloOpcode::kRecv) {
      return operand;
    }

    // Find while loop and index, otherwise.
    CHECK(operand->opcode() == HloOpcode::kGetTupleElement);
    const auto* gte = Cast<HloGetTupleElementInstruction>(operand);
    const HloInstruction* iter_tuple = operand->operand(0);
    if (iter_tuple->opcode() == HloOpcode::kParameter) {
      // send-done/recv-done in the loop body.
      CHECK(Cast<HloParameterInstruction>(iter_tuple)->parameter_number() == 0);
      auto maybe_while =
          iter_tuple->parent()->GetUniqueCaller(HloOpcode::kWhile);
      CHECK(maybe_while);
      while_op = *maybe_while;
      i = gte->tuple_index();
    } else {
      // send-done/recv-done proceeding the loop.
      CHECK(iter_tuple->opcode() == HloOpcode::kWhile);
      while_op = iter_tuple;
      i = gte->tuple_index();
    }
  }

  // Extract canonical start op from while loop's init.
  CHECK(while_op != nullptr);
  CHECK(0 <= i && i < while_op->shape().tuple_shapes().size());
  const HloInstruction* init = while_op->operand(0);
  const HloInstruction* canonical_start_op = init->operand(i);
  CHECK(canonical_start_op->opcode() == HloOpcode::kSend ||
        canonical_start_op->opcode() == HloOpcode::kRecv);
  return canonical_start_op;
}

}  // namespace

ThunkEmitter::ThunkEmitter(
    IrEmitterContext* absl_nonnull ir_emitter_context,
    llvm_ir::LLVMCommandLineOptionsReleasableLock* absl_nonnull
        llvm_options_lock)
    : ir_emitter_context_(ir_emitter_context),
      send_recv_events_(std::make_shared<HostSendRecvAsyncEvents>()),
      copy_events_(std::make_shared<CopyThunk::AsyncEvents>()),
      nvshmem_buffer_addresses_(std::make_shared<NvshmemBufferAddresses>()),
      call_graph_(CallGraph::Build(&ir_emitter_context->hlo_module())),
      constants_module_(ir_emitter_context_->CreateLLVMModule(
          absl::StrCat(ir_emitter_context_->hlo_module().name(), "_consts"))),
      llvm_options_lock_(llvm_options_lock) {}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConstant(
    const HloConstantInstruction* instr) {
  TF_ASSIGN_OR_RETURN(DenseDataIntermediate content,
                      LiteralToXlaFormat(instr->literal()));

  int element_bytes =
      primitive_util::ByteWidth(instr->literal().shape().element_type());
  TF_RET_CHECK(content.span().size() % element_bytes == 0);
  // Treat packed constants as a byte constant.
  int num_elements = content.span().size() / element_bytes;

  std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSliceForHlo(instr, {}));

  GpuExecutable::ConstantInfo info =
      AppendGlobalConstant(constants_module_.get(), num_elements, element_bytes,
                           global_name, slice.index(), std::move(content));
  ir_emitter_context_->constants().push_back(std::move(info));
  return ThunkSequence{};
}

ThunkSequence GetThunkSequence(std::unique_ptr<Thunk> ir_emitter) {
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(ir_emitter));
  return thunk_sequence;
}

void AppendThunkSequence(ThunkSequence& thunks,
                         ThunkSequence& additional_thunks) {
  thunks.insert(thunks.end(),
                std::make_move_iterator(additional_thunks.begin()),
                std::make_move_iterator(additional_thunks.end()));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConditional(
    const HloInstruction* instr) {
  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  branch_thunks.reserve(instr->branch_count());
  for (auto comp : instr->branch_computations()) {
    TF_ASSIGN_OR_RETURN(auto thunk_sequence, EmitHloComputation(comp));
    Thunk::ThunkInfo branch_thunk_info =
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId());
    branch_thunk_info.profile_annotation +=
        absl::StrCat("_branch_", comp->name());
    branch_thunks.push_back(std::make_unique<SequentialThunk>(
        branch_thunk_info, std::move(thunk_sequence)));
  }
  TF_ASSIGN_OR_RETURN(auto slice,
                      GetAllocationSliceForHlo(instr->operand(0), {}));

  auto placeholder = GetThunkSequence(std::make_unique<ConditionalThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      ShapedSlice{slice, instr->operand(0)->shape()},
      std::move(branch_thunks)));
  return placeholder;
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitPadToStatic(
    const HloCustomCallInstruction* instr) {
  std::string ir_name = std::string(instr->name());
  auto local_llvm_module = ir_emitter_context_->CreateLLVMModule(ir_name);

  TF_ASSIGN_OR_RETURN(auto thunk_sequence,
                      EmitPadToStaticLLVMIR(instr, local_llvm_module.get(),
                                            ir_emitter_context_));
  kernel_modules_.push_back(std::move(local_llvm_module));
  return thunk_sequence;
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSliceToDynamic(
    const HloCustomCallInstruction* instr) {
  std::string ir_name = std::string(instr->name());
  auto local_llvm_module = ir_emitter_context_->CreateLLVMModule(ir_name);

  TF_ASSIGN_OR_RETURN(auto thunk_sequence,
                      EmitSliceToDynamicLLVMIR(instr, local_llvm_module.get(),
                                               ir_emitter_context_));
  kernel_modules_.push_back(std::move(local_llvm_module));
  return thunk_sequence;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCommandBufferThunk(
    const HloInstruction* instr) {
  // Spawn a new ThunkEmitter to emit thunks for the command buffer computation.
  // Then convert emitted thunks to a sequence of CommandBufferCmd. The
  // resulting thunk added to the thunk sequence is a CommandBufferThunk. Thunks
  // emitted from the command buffer computation are discarded.
  DCHECK_EQ(instr->called_computations().size(), 1);
  const HloComputation* command_buffer = instr->called_computations().front();
  TF_ASSIGN_OR_RETURN(auto thunk_sequence, EmitHloComputation(command_buffer));

  // Maybe serialize all commands in a sequence by forcing barriers
  // between all recorded commands. This guarantees that we execute
  // all device operations in the exact same order as a thunk
  // sequence.
  CommandBufferCmdExecutor::SynchronizationMode synchronization_mode;
  auto mode = ir_emitter_context_->debug_options()
                  .xla_gpu_command_buffer_scheduling_mode();
  switch (mode) {
    case DebugOptions::SERIALIZE:
      synchronization_mode =
          CommandBufferCmdExecutor::SynchronizationMode::kSerialize;
      break;
    case DebugOptions::CONCURRENT:
      synchronization_mode =
          CommandBufferCmdExecutor::SynchronizationMode::kConcurrent;
      break;
    case DebugOptions::LHS:
      synchronization_mode =
          CommandBufferCmdExecutor::SynchronizationMode::kLHS;
      break;
    default:
      return Internal("Unsupported command buffer scheduling mode: %d", mode);
  }

  bool enable_loop_unroll = ir_emitter_context_->debug_options()
                                .xla_gpu_command_buffer_unroll_loops();
  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdExecutor cmd_executor,
      ConvertToCommands(
          thunk_sequence,
          ConvertToCommandsOptions{synchronization_mode, enable_loop_unroll}));

  return GetThunkSequence(std::make_unique<CommandBufferThunk>(
      std::move(cmd_executor),
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo{},
                                        std::move(thunk_sequence)),
      ir_emitter_context_->debug_options()
          .xla_enable_command_buffers_during_profiling()));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolutionThunk(
    const HloCustomCallInstruction* instr) {
  std::vector<ShapedSlice> operand_slices;
  operand_slices.reserve(instr->operand_count());
  for (const HloInstruction* operand : instr->operands()) {
    ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(operand, {}));
    operand_slices.push_back(slice);
  }

  // The first and the last element in the result tuple for a convolution are
  // always the result and the scratch buffer. It may have auxiliary results in
  // addition to the main result.
  std::vector<ShapedSlice> result_slices;
  for (int i = 0; i < instr->shape().tuple_shapes().size() - 1; i++) {
    ASSIGN_OR_RETURN(ShapedSlice result_slice,
                     GetShapedSliceForHlo(instr, {i}));
    result_slices.push_back(result_slice);
  }

  TF_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      instr->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& backend_config =
      gpu_config.cudnn_conv_backend_config();
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scratch_slice,
                      GetAllocationSliceForHlo(
                          instr, {instr->shape().tuple_shapes_size() - 1}));
  GpuConvDescriptor descriptor = {kind,
                                  backend_config,
                                  instr->operand(0)->shape(),
                                  instr->operand(1)->shape(),
                                  instr->shape().tuple_shapes(0),
                                  static_cast<size_t>(scratch_slice.size()),
                                  instr->window(),
                                  instr->convolution_dimension_numbers(),
                                  instr->feature_group_count()};
  TF_ASSIGN_OR_RETURN(auto thunk,
                      ConvolutionThunk::Create(
                          Thunk::ThunkInfo::WithProfileAnnotation(
                              instr, ir_emitter_context_->GetNextThunkId()),
                          std::move(descriptor), std::move(operand_slices),
                          std::move(result_slices), scratch_slice));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitGemmThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a,
                      GetAllocationSliceForHlo(instr->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSliceForHlo(instr->operand(1), {}));

  // Result of a legacy cuBLAS custom call can be a tuple if we
  // explicitly allocate workspace buffer in HLO. If result is an
  // array, it means that workspace is not available, and cuBLAS
  // will allocate its own workspace.
  BufferAllocation::Slice c;
  std::optional<BufferAllocation::Slice> workspace;

  if (instr->shape().IsArray()) {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr, {}));
  } else {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr, {0}));
    TF_ASSIGN_OR_RETURN(workspace, GetAllocationSliceForHlo(instr, {1}));
  }

  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(static_cast<const HloInstruction*>(instr),
                      ir_emitter_context_->gpu_compute_capability()));
  auto thunk = std::make_unique<GemmThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(config), a, b, c, workspace,
      RequireDeterminism(ir_emitter_context_->hlo_module().config()));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  TF_ASSIGN_OR_RETURN(bool has_vector_bias,
                      xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));
  bool has_matrix_bias = config.beta() != 0;

  TF_RET_CHECK(instr->operand_count() ==
               2 + int{has_matrix_bias} + int{has_vector_bias});

  TF_ASSIGN_OR_RETURN(
      bool has_aux_output,
      xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));
  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSliceForHlo(instr->operand(1)));
  BufferAllocation::Slice c;
  if (has_matrix_bias) {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr->operand(2)));
  } else {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr, output_index));
  }
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d,
                      GetAllocationSliceForHlo(instr, output_index));

  BufferAllocation::Slice bias;
  if (has_vector_bias) {
    TF_ASSIGN_OR_RETURN(bias, GetAllocationSliceForHlo(
                                  instr->operand(has_matrix_bias ? 3 : 2)));
  }

  BufferAllocation::Slice aux;
  if (has_aux_output) {
    TF_ASSIGN_OR_RETURN(aux, GetAllocationSliceForHlo(instr, {1}));
  }

  std::optional<BufferAllocation::Slice> workspace_buffer;
  if (instr->shape().IsTuple() &&
      (instr->shape().tuple_shapes().size() - has_aux_output - 1)) {
    TF_RET_CHECK(
        (has_aux_output && instr->shape().tuple_shapes().size() == 3) ||
        (!has_aux_output && instr->shape().tuple_shapes().size() == 2));
    TF_ASSIGN_OR_RETURN(workspace_buffer,
                        GetAllocationSliceForHlo(
                            instr, {instr->shape().tuple_shapes_size() - 1}));
  }

  TF_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(static_cast<const HloInstruction*>(instr),
                      ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  BufferAllocation::Slice a_scale, b_scale, c_scale, d_scale, d_amax;
  TF_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                      gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  std::string canonical_hlo = instr->ToString(
      HloPrintOptions::Fingerprint().set_print_backend_config(true));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax, workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulThunkF8(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() > 3 && instr->operand_count() < 8);
  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  TF_ASSIGN_OR_RETURN(bool has_vector_bias,
                      xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  TF_ASSIGN_OR_RETURN(
      bool has_aux_output,
      xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSliceForHlo(instr->operand(1)));
  BufferAllocation::Slice c;
  bool has_matrix_bias = config.beta() != 0;
  if (has_matrix_bias) {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr->operand(2)));
  } else {
    TF_ASSIGN_OR_RETURN(c, GetAllocationSliceForHlo(instr, output_index));
  }
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d,
                      GetAllocationSliceForHlo(instr, output_index));

  int a_scale_index = has_matrix_bias ? 3 : 2;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a_scale,
                      GetAllocationSliceForHlo(instr->operand(a_scale_index)));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice b_scale,
      GetAllocationSliceForHlo(instr->operand(a_scale_index + 1)));

  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();
  bool is_fp8 = instr->shape().tuple_shapes(0).element_type() == F8E4M3FN ||
                instr->shape().tuple_shapes(0).element_type() == F8E5M2;
  // cublasLT requires c_scale/d_scale to be null when C/D is not
  // FP8. Currently, C cannot be FP8.
  BufferAllocation::Slice c_scale, d_scale;
  if (is_cuda && is_fp8) {
    TF_ASSIGN_OR_RETURN(d_scale,
                        GetAllocationSliceForHlo(instr->operands().back()));
  }

  BufferAllocation::Slice bias;
  if (has_vector_bias) {
    TF_ASSIGN_OR_RETURN(
        bias, GetAllocationSliceForHlo(instr->operand(a_scale_index + 2)));
  }

  BufferAllocation::Slice d_amax;
  if (config.damax_output()) {
    TF_ASSIGN_OR_RETURN(d_amax, GetAllocationSliceForHlo(instr, {1}));
  }

  TF_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(static_cast<const HloInstruction*>(instr),
                      ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  BufferAllocation::Slice aux;  // Not used.
  TF_RET_CHECK(!has_aux_output);
  std::optional<BufferAllocation::Slice> workspace_buffer;
  if (instr->shape().tuple_shapes().size() - config.damax_output() == 2) {
    TF_ASSIGN_OR_RETURN(workspace_buffer,
                        GetAllocationSliceForHlo(
                            instr, {instr->shape().tuple_shapes_size() - 1}));
  }

  TF_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                      gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  std::string canonical_hlo = instr->ToString(
      HloPrintOptions::Fingerprint().set_print_backend_config(true));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax, workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConvolutionReorderThunk(
    const HloCustomCallInstruction* instr) {
  bool has_bias = instr->operand_count() > 1;

  TF_ASSIGN_OR_RETURN(ShapedSlice filter_input,
                      GetShapedSliceForHlo(instr->operand(0)));

  ShapedSlice filter_output;
  std::optional<ConvolutionReorderThunk::BiasBuffers> biases;
  if (has_bias) {
    TF_ASSIGN_OR_RETURN(filter_output, GetShapedSliceForHlo(instr, {0}));

    TF_ASSIGN_OR_RETURN(ShapedSlice bias_input,
                        GetShapedSliceForHlo(instr->operand(1)));
    TF_ASSIGN_OR_RETURN(ShapedSlice bias_output,
                        GetShapedSliceForHlo(instr, {1}));
    biases = {{bias_input, bias_output}};
  } else {
    TF_ASSIGN_OR_RETURN(filter_output, GetShapedSliceForHlo(instr));
  }

  ASSIGN_OR_RETURN(auto thunk,
                   ConvolutionReorderThunk::Create(
                       Thunk::ThunkInfo::WithProfileAnnotation(
                           instr, ir_emitter_context_->GetNextThunkId()),
                       filter_input, filter_output, biases));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNormThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto const gpu_backend_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::CudnnNormBackendConfig& backend_config =
      gpu_backend_config.cudnn_norm_backend_config();

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice x_slice,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scale_slice,
                      GetAllocationSliceForHlo(instr->operand(1)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice y_or_dx_slice,
                      GetAllocationSliceForHlo(instr, {0}));

  std::optional<BufferAllocation::Slice> bias_slice, expectation_slice,
      norm_factor_slice, dy_slice, dscale_slice, dbias_slice;

  if (backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER ||
      backend_config.kind() ==
          xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    TF_ASSIGN_OR_RETURN(bias_slice,
                        GetAllocationSliceForHlo(instr->operand(2)));
  }
  if (backend_config.kind() ==
      xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN) {
    TF_ASSIGN_OR_RETURN(expectation_slice,
                        GetAllocationSliceForHlo(instr, {1}));
    TF_ASSIGN_OR_RETURN(norm_factor_slice,
                        GetAllocationSliceForHlo(instr, {2}));
  }
  if (backend_config.kind() == xla::gpu::CudnnNormBackendConfig::LAYER_BWD) {
    TF_ASSIGN_OR_RETURN(dy_slice, GetAllocationSliceForHlo(instr->operand(2)));
    TF_ASSIGN_OR_RETURN(expectation_slice,
                        GetAllocationSliceForHlo(instr->operand(3)));
    TF_ASSIGN_OR_RETURN(norm_factor_slice,
                        GetAllocationSliceForHlo(instr->operand(4)));
    TF_ASSIGN_OR_RETURN(dscale_slice, GetAllocationSliceForHlo(instr, {1}));
    TF_ASSIGN_OR_RETURN(dbias_slice, GetAllocationSliceForHlo(instr, {2}));
  }
  TF_ASSIGN_OR_RETURN(
      ShapedSlice scratch_slice,
      GetShapedSliceForHlo(instr, {instr->shape().tuple_shapes_size() - 1}));

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

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<NormThunk> thunk,
      NormThunk::Create(Thunk::ThunkInfo::WithProfileAnnotation(
                            instr, ir_emitter_context_->GetNextThunkId()),
                        std::move(descriptor), x_slice, scale_slice,
                        y_or_dx_slice, bias_slice, expectation_slice,
                        norm_factor_slice, dy_slice, dscale_slice, dbias_slice,
                        scratch_slice.slice));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCuDnnThunk(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      emitters::KernelArguments::Create(
                          ir_emitter_context_->buffer_assignment(),
                          GetDefaultBufferAlignment(), instr));
  TF_ASSIGN_OR_RETURN(const std::string fingerprint,
                      FingerprintWithBackendConfig<GpuBackendConfig>(*instr));
  // check if sdpa dropout is enabled
  std::optional<int64_t> dropout_seed = std::nullopt;
  if (MHACallHasDropout(instr->custom_call_target())) {
    TF_ASSIGN_OR_RETURN(const auto gpu_config,
                        instr->backend_config<xla::gpu::GpuBackendConfig>());
    dropout_seed = gpu_config.cudnn_fmha_backend_config().seed();
  }
  return GetThunkSequence(std::make_unique<CuDnnThunk>(
      fingerprint,
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      kernel_arguments.GetArgumentBufferSlices(),
      kernel_arguments.GetArgumentOutputFlags(), dropout_seed));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitPtxCustomCall(
    const HloCustomCallInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto thunk,
                      EmitPtxCustomKernelThunk(instr, ir_emitter_context_));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSliceForHlo(
    const HloInstruction* instr, const ShapeIndex& index) const {
  return ir_emitter_context_->buffer_assignment().GetUniqueSlice(instr, index);
}

absl::StatusOr<ShapedSlice> ThunkEmitter::GetShapedSliceForHlo(
    const HloInstruction* instr, const ShapeIndex& index) const {
  ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSliceForHlo(instr, index));
  ASSIGN_OR_RETURN(
      Shape shape,
      ir_emitter_context_->buffer_assignment().GetShapeForUniqueSlice(instr,
                                                                      index));
  return ShapedSlice{slice, shape};
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCubDeviceRadixSort(
    const HloCustomCallInstruction* instr) {
  if (instr->operand_count() != 1 && instr->operand_count() != 2) {
    return Internal("Invalid number of operands for radix sort");
  }

  absl::InlinedVector<BufferAllocation::Slice, 2> operands;
  for (int i = 0; i < instr->operand_count(); ++i) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice operand,
                        GetAllocationSliceForHlo(instr->operand(i), {}));
    operands.push_back(operand);
  }

  absl::InlinedVector<BufferAllocation::Slice, 2> results;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result,
                      GetAllocationSliceForHlo(instr, {0}));
  results.push_back(result);

  BufferAllocation::Slice scratch;
  if (instr->operand_count() == 1) {
    TF_ASSIGN_OR_RETURN(scratch, GetAllocationSliceForHlo(instr, {1}));
  } else {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result,
                        GetAllocationSliceForHlo(instr, {1}));
    results.push_back(result);
    TF_ASSIGN_OR_RETURN(scratch, GetAllocationSliceForHlo(instr, {2}));
  }

  TF_ASSIGN_OR_RETURN(xla::SortOptions options,
                      instr->backend_config<xla::SortOptions>());
  const Shape& operand_shape = instr->operand(0)->shape();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<CubSortThunk> thunk,
      CubSortThunk::Create(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          operand_shape.element_type(),
          instr->operand_count() == 2
              ? std::optional(instr->operand(1)->shape().element_type())
              : std::nullopt,
          operands, results, scratch, options.descending(),
          Product(operand_shape.dimensions()) /
              operand_shape.dimensions(operand_shape.dimensions().size() - 1),
          ir_emitter_context_->platform_name()));
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCustomCallThunk(
    const HloCustomCallInstruction* instr) {
  const std::string& call_target_name = instr->custom_call_target();

  // Typed FFI custom calls is a replacement for legacy custom calls
  // with a rich type safe API.
  bool is_ffi_custom_call =
      instr->api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  using Slices = std::vector<NullableShapedSlice>;

  Slices operands;
  for (auto* operand : instr->operands()) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            operands.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          TF_ASSIGN_OR_RETURN(auto slice,
                              GetAllocationSliceForHlo(operand, index));
          operands.push_back(ShapedSlice{slice, subshape});
          return absl::OkStatus();
        }));
  }

  Slices results;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          results.push_back(std::nullopt);
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSliceForHlo(instr, index));
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

  auto ffi_thunk = [&]() -> absl::StatusOr<std::unique_ptr<CustomCallThunk>> {
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
      TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    }
    auto released_lock_keeper = llvm_options_lock_->TemporarilyReleaseLock();
    return CustomCallThunk::Create(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        call_target_name, std::move(operands), std::move(results),
        std::move(attributes),
        called_computations.empty() ? nullptr : called_computations[0],
        ir_emitter_context_->platform_name());
  };

  auto legacy_thunk =
      [&]() -> absl::StatusOr<std::unique_ptr<CustomCallThunk>> {
    std::string opaque =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().opaque()
            : instr->raw_backend_config_string();
    return CustomCallThunk::Create(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        call_target_name, std::move(operands), std::move(results),
        std::move(opaque), instr->api_version(),
        ir_emitter_context_->platform_name());
  };

  absl::StatusOr<std::unique_ptr<CustomCallThunk>> custom_call_thunk =
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

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFftThunk(
    const HloFftInstruction* instr) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                      GetAllocationSliceForHlo(instr));
  return GetThunkSequence(std::make_unique<FftThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr->fft_type(), instr->fft_length(),
      /*input_buffer=*/arg_slice,
      /*output_buffer=*/dest_slice,
      /*input_shape=*/instr->operand(0)->shape(),
      /*output_shape=*/instr->shape()));
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

  ASSIGN_OR_RETURN(ShapedSlice a_slice, GetShapedSliceForHlo(operands[0]));
  ASSIGN_OR_RETURN(ShapedSlice b_slice, GetShapedSliceForHlo(operands[1]));
  ASSIGN_OR_RETURN(ShapedSlice result_slice, GetShapedSliceForHlo(instr, {0}));
  ASSIGN_OR_RETURN(ShapedSlice temp_slice, GetShapedSliceForHlo(instr, {1}));

  TriangularSolveOptions backend_config;
  auto& backend_config_str = instr->raw_backend_config_string();
  if (!backend_config_str.empty()) {
    TF_RETURN_IF_ERROR(
        tsl::HumanReadableJsonToProto(backend_config_str, &backend_config));
  }

  ThunkSequence thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output
  // if they aren't the same buffer.
  if (b_slice.slice != result_slice.slice) {
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        /*source_buffer=*/b_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(b_slice.shape)));
  }

  thunks.push_back(std::make_unique<TriangularSolveThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      backend_config, a_slice, result_slice, temp_slice));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    return thunks;
  }
  auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  // Don't repeat the annotation from inside thunks
  thunk_info.profile_annotation = {};
  return GetThunkSequence(
      std::make_unique<SequentialThunk>(thunk_info, std::move(thunks)));
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
  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      emitters::KernelArguments::Create(
                          ir_emitter_context_->buffer_assignment(),
                          GetDefaultBufferAlignment(), instr));

  auto dtype = data_shape.element_type();
  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();
  if (is_cuda && instr->GetModule()
                     ->config()
                     .debug_options()
                     .xla_gpu_experimental_use_raft_select_k()) {
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

    Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
        instr, ir_emitter_context_->GetNextThunkId());
    if (use_raft_select_k) {
      return GetThunkSequence(std::make_unique<SelectKThunk>(
          std::move(thunk_info), batch_size, n, k, dtype, kernel_arguments));
    }
  }

  auto wavefront_size =
      ir_emitter_context_->gpu_device_info().threads_per_warp();

  TF_RET_CHECK(k <= 16) << "CustomCall TopK requires k <= 16";
  // Load TopK custom kernel.
  TF_ASSIGN_OR_RETURN(
      CustomKernel kernel,
      kernel::topk::GetTopKKernel("topk", dtype, n, k, batch_size,
                                  platform_name(), wavefront_size));

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  return GetThunkSequence(std::make_unique<CustomKernelThunk>(
      std::move(thunk_info), std::move(kernel), kernel_arguments));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitTritonCustomCall(
    const HloCustomCallInstruction* instr) {
  auto generate = [this, &instr]() -> absl::StatusOr<KernelReuseCache::Entry> {
    mlir::MLIRContext& mlir_context = *ir_emitter_context_->mlir_context();
    LoadMlirDialectsForTriton(mlir_context);
    auto call =
        TritonCall::Parse(instr->raw_backend_config_string(), &mlir_context);
    auto kernel_name = ir_emitter_context_->GetSanitizedUniqueName(call.name);
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

    auto triton_fn =
        triton_module->lookupSymbol<mlir::triton::FuncOp>(call.name);
    TF_RET_CHECK(triton_fn)
        << "Call name not found in the Triton module: " << call.name;
    triton_fn.setName(kernel_name);

    HloModule* hlo_module = instr->GetModule();
    // If emit_kernels if false (i.e., when deserializing an already
    // compiled executable), we do not emit code, but we still need
    // to run part of the compiler to figure out the size of the
    // shared memory and the cluster dimensions for the thunk. We
    // also must call the name uniqifier as if emitting code so that
    // the future generated names remain in sync.
    bool emit_kernels = ir_emitter_context_->emit_kernels();

    BlockLevelParameters block_level_parameters;
    block_level_parameters.num_stages = call.num_stages;
    block_level_parameters.num_warps = call.num_warps;
    block_level_parameters.num_ctas = 1;

    TF_ASSIGN_OR_RETURN(
        auto result,
        CompileTritonToLLVM(kernel_name, *hlo_module,
                            ir_emitter_context_->gpu_device_info(),
                            block_level_parameters, triton_module.get(),
                            ir_emitter_context_->target_triple(),
                            ir_emitter_context_->data_layout(),
                            *ir_emitter_context_->llvm_context(), mlir_context,
                            /*is_xla_fusion=*/false, emit_kernels));

    TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                        emitters::KernelArguments::Create(
                            ir_emitter_context_->buffer_assignment(),
                            GetDefaultBufferAlignment(), instr));
    auto launch_dimensions = LaunchDimensions(
        se::BlockDim(call.grid_x, call.grid_y, call.grid_z),
        se::ThreadDim(
            call.num_warps *
            ir_emitter_context_->gpu_device_info().threads_per_warp()));

    if (emit_kernels) {
      TF_ASSIGN_OR_RETURN(
          llvm::Function * kernel,
          RemoveUnusedTritonAbiArguments(result.llvm_module.get(),
                                         *ir_emitter_context_, kernel_name));

      AnnotateAttrsIfUnset(kernel_arguments, *kernel);
      TF_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
          ir_emitter_context_->gpu_device_info(), launch_dimensions, kernel,
          result.llvm_module.get()));
    }

    kernel_modules_.push_back(std::move(result.llvm_module));
    return {{kernel_name, launch_dimensions, /*cluster_dim=*/std::nullopt,
             result.shmem_bytes}};
  };

  auto [status_or_entry, was_cached] =
      ir_emitter_context_->kernel_cache().GetWithStatus(
          instr->raw_backend_config_string(), generate);
  TF_ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry, status_or_entry);

  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      emitters::KernelArguments::Create(
                          ir_emitter_context_->buffer_assignment(),
                          GetDefaultBufferAlignment(), instr));

  return GetThunkSequence(std::make_unique<KernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      entry->kernel_name, kernel_arguments, entry->launch_dimensions,
      /*cluster_dim=*/std::nullopt, entry->shmem_bytes, entry->tma_metadata));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncComputation(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(auto stream,
                      stream_assignment.GetSyncExecutionStreamId(wrapped));
  TF_RET_CHECK(wrapped->called_computations().size() == 1);
  auto computation = wrapped->called_computations().front();
  TF_ASSIGN_OR_RETURN(auto comp_thunks, EmitHloComputation(computation));
  auto sequential_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo{}, std::move(comp_thunks));
  for (auto& thunk : sequential_thunk->thunks()) {
    thunk->set_execution_stream_id(stream);
  }
  auto* async_start = Cast<HloAsyncInstruction>(instr);
  TF_ASSIGN_OR_RETURN(
      ExecutionStreamAssignment::AsyncExecutionStreamIds async_streams,
      stream_assignment.GetAsyncExecutionStreamIds(async_start));
  // We launch the thunk sequence computation on a concurrent
  // stream. The concurrent stream needs to first wait until the
  // main stream has finished calculating any values that may be
  // used as input. We enforce this by inlining a `WaitForStreams`
  // thunk on the main stream.
  ThunkSequence thunks;
  thunks.push_back(std::make_unique<WaitForStreamsThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      async_streams.destination_stream_id, async_streams.source_stream_id));
  thunks.push_back(std::move(sequential_thunk));
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitFusion(
    const HloFusionInstruction* instr) {
  const se::DeviceDescription& device_info =
      ir_emitter_context_->gpu_device_info();
  const HloFusionAnalysis fusion_analysis =
      HloFusionAnalysis::Create(*instr, device_info);
  VLOG(3) << "ThunkEmitter::EmitFusion:start";
  std::unique_ptr<FusionInterface> emitter = GetFusionEmitter(
      /*fusion_info=*/HloFusionInfo(
          /*analysis=*/fusion_analysis, instr,
          /*buffer_assignment=*/
          &ir_emitter_context_->buffer_assignment(),
          /*call_graph=*/*call_graph_),
      ir_emitter_context_->mlir_context());
  TF_ASSIGN_OR_RETURN(auto result, emitter->Emit(*ir_emitter_context_, *instr));

  // Use override flag because libdevice functions can be present in both.
  if (result.module) {
    kernel_modules_.push_back(std::move(result.module));
  }

  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  for (std::unique_ptr<Thunk>& thunk : result.thunks) {
    TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                        stream_assignment.GetSyncExecutionStreamId(instr));
    thunk->set_execution_stream_id(execution_stream_id);
  }
  VLOG(3) << "ThunkEmitter::EmitFusion:complete";
  return std::move(result.thunks);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopy(
    const HloInstruction* instr) {
  TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
      instr->operand(0)->shape(), instr->shape(),
      Layout::Equal().MinorToMajorOnly()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(instr));
  return GetThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      /*source_buffer=*/ShapedSlice{src_buffer, instr->operand(0)->shape()},
      /*destination_buffer=*/ShapedSlice{dst_buffer, instr->shape()},
      /*mem_size=*/src_buffer.size()));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncCustomCallStart(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  auto* async_start = Cast<HloAsyncInstruction>(instr);
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(
      ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
      stream_assignment.GetAsyncExecutionStreamIds(async_start));
  ThunkSequence thunks = GetThunkSequence(std::make_unique<WaitForStreamsThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      streams.destination_stream_id, streams.source_stream_id));
  TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                      stream_assignment.GetSyncExecutionStreamId(wrapped));

  auto* custom_call = Cast<HloCustomCallInstruction>(wrapped);
  if (IsLegacyCublasMatmul(*wrapped)) {
    TF_ASSIGN_OR_RETURN(auto gemm_thunks, EmitGemmThunk(custom_call));
    CHECK_EQ(gemm_thunks.size(), 1);
    gemm_thunks.back()->set_execution_stream_id(execution_stream_id);
    AppendThunkSequence(thunks, gemm_thunks);
    return thunks;
  }
  if (IsCublasLtMatmul(*wrapped)) {
    TF_ASSIGN_OR_RETURN(auto cublas_lt_matmul_thunks,
                        EmitCublasLtMatmulThunk(custom_call));
    CHECK_EQ(cublas_lt_matmul_thunks.size(), 1);
    cublas_lt_matmul_thunks.back()->set_execution_stream_id(
        execution_stream_id);
    AppendThunkSequence(thunks, cublas_lt_matmul_thunks);
    return thunks;
  }
  if (IsCublasLtMatmulF8(*wrapped)) {
    TF_ASSIGN_OR_RETURN(auto cublas_lt_matmul_thunks,
                        EmitCublasLtMatmulThunkF8(custom_call));
    CHECK_EQ(cublas_lt_matmul_thunks.size(), 1);
    cublas_lt_matmul_thunks.back()->set_execution_stream_id(
        execution_stream_id);
    AppendThunkSequence(thunks, cublas_lt_matmul_thunks);
    return thunks;
  }
  return Internal("Unsupported async custom call instruction: %s",
                  HloOpcodeString(wrapped->opcode()));
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

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitWhile(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto config,
                      instr->backend_config<xla::WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count = std::nullopt;
  if (config.has_known_trip_count()) {
    trip_count = config.known_trip_count().n();
  }

  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Generate thunk sequence for while 'condition'.
  TF_ASSIGN_OR_RETURN(auto cond_thunks, EmitHloComputation(condition));

  // Generate thunk sequence for while 'body'.
  TF_ASSIGN_OR_RETURN(auto body_thunks, EmitHloComputation(body));

  // Buffer slice holding while loop predicate.
  TF_ASSIGN_OR_RETURN(
      auto pred, GetAllocationSliceForHlo(condition->root_instruction(), {}));

  Thunk::ThunkInfo while_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  Thunk::ThunkInfo cond_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  cond_thunk_info.profile_annotation += "_condition";
  Thunk::ThunkInfo body_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  body_thunk_info.profile_annotation += "_body";

  return GetThunkSequence(
      std::make_unique<WhileThunk>(while_thunk_info, instr, pred,
                                   std::make_unique<SequentialThunk>(
                                       cond_thunk_info, std::move(cond_thunks)),
                                   std::make_unique<SequentialThunk>(
                                       body_thunk_info, std::move(body_thunks)),
                                   trip_count));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr) {
  std::string ir_name = std::string(instr->name());
  auto local_llvm_module = ir_emitter_context_->CreateLLVMModule(ir_name);

  TF_ASSIGN_OR_RETURN(auto thunk_sequence,
                      EmitRngGetAndUpdateStateLLVMIR(
                          instr, local_llvm_module.get(), ir_emitter_context_));
  kernel_modules_.push_back(std::move(local_llvm_module));
  return thunk_sequence;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSort(
    const HloSortInstruction* sort) {
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
    TF_ASSIGN_OR_RETURN(destination_buffer,
                        GetAllocationSliceForHlo(sort, shape_index));
    TF_ASSIGN_OR_RETURN(source_address,
                        GetAllocationSliceForHlo(sort->operand(i), {}));

    if (destination_buffer != source_address) {
      // TODO(b/26783907): Figure out why we never seem to share
      // buffers for key/value sort.
      VLOG(2) << op_name << " requires initial D2D copy for operand " << i;
      thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              sort, ir_emitter_context_->GetNextThunkId()),
          /*source_buffer=*/
          ShapedSlice{source_address, sort->operand(i)->shape()},
          /*destination_buffer=*/
          ShapedSlice{destination_buffer, sort->operand(i)->shape()},
          ShapeUtil::ByteSizeOf(sort->operand(i)->shape())));
    }
  }

  auto local_llvm_module = ir_emitter_context_->CreateLLVMModule(op_name);

  TF_ASSIGN_OR_RETURN(ThunkSequence sort_thunks,
                      EmitBitonicSortLLVMIR(sort, local_llvm_module.get(),
                                            ir_emitter_context_));
  AppendThunkSequence(thunks, sort_thunks);
  kernel_modules_.push_back(std::move(local_llvm_module));
  return thunks;
}

template <typename ThunkType>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReplicaOrPartitionId(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSliceForHlo(instr, {}));
  return GetThunkSequence(std::make_unique<ThunkType>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      result_slice));
}

bool IsNvshmemCollective(const HloInstruction* instr) {
  if (instr->has_backend_config()) {
    auto gpu_config = instr->backend_config<GpuBackendConfig>();
    const CollectiveBackendConfig& backend_config =
        gpu_config.value().collective_backend_config();
    return backend_config.backend() == CollectiveBackendConfig::NVSHMEM;
  }
  return false;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectiveMetadata(
    const HloInstruction* instr) {
  std::vector<CollectiveMetadataThunk::Buffer> buffers;
  buffers.reserve(instr->operands().size());
  for (const HloInstruction* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        GetAllocationSliceForHlo(operand, {}));
    buffers.push_back({slice, operand->shape().layout().memory_space()});
  }

  // Operation result should be a tuple where the last element is the buffer for
  // the metadata.
  ShapeIndex result_shape_index = {static_cast<int64_t>(buffers.size())};
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result,
                      GetAllocationSliceForHlo(instr, result_shape_index));

  return GetThunkSequence(std::make_unique<CollectiveMetadataThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      CollectiveMetadataThunk::GetCollectiveConfig(*instr), std::move(buffers),
      result));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectivePermute(
    const HloCollectivePermuteInstruction* instr) {
  // First output is aliased.
  TF_RET_CHECK(
      instr->shape().IsTuple() && instr->shape().tuple_shapes().size() == 2 &&
      Shape::Equal().IgnoreMemorySpaceInLayout()(
          instr->shape().tuple_shapes(0), instr->shape().tuple_shapes(1)));

  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  const int64_t replica_count = hlo_config.replica_count();
  const int64_t partition_count = hlo_config.num_partitions();

  auto operands = instr->operands();
  std::vector<CollectiveThunk::Buffer> buffers;
  ThunkSequence thunks;
  for (int oprd_idx = 0; oprd_idx < operands.size(); ++oprd_idx) {
    const auto operand = operands.at(oprd_idx);
    const ShapeIndex nested_shape_idx = {1, oprd_idx}, normal_shape_idx = {1};
    const Shape operand_shape = operand->shape();
    const Shape result_shape = instr->shape().tuple_shapes(1);
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                        GetAllocationSliceForHlo(
                            instr, result_shape.IsTuple() ? nested_shape_idx
                                                          : normal_shape_idx));

    const int64_t src_memory_space = operand_shape.layout().memory_space();
    Shape result_buffer_shape = (result_shape.IsTuple())
                                    ? result_shape.tuple_shapes(oprd_idx)
                                    : result_shape;

    const int64_t dst_memory_space =
        result_buffer_shape.layout().memory_space();

    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice source_slice,
                        GetAllocationSliceForHlo(operand));
    if (CollectivePermuteStartThunk::IsDegenerate(instr, replica_count,
                                                  partition_count)) {
      // For a degenerate collective permute, just generate a copy
      // thunk.
      thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          /*source_buffer=*/ShapedSlice{source_slice, operand_shape},
          /*destination_buffer=*/
          ShapedSlice{result_slice, result_buffer_shape},
          /*mem_size=*/ShapeUtil::ByteSizeOf(operand_shape)));
      // Signal that start thunk not created with nullptr.
      GetCollectivesAsyncEvents().try_emplace(instr, nullptr);
    } else {
      const CollectiveThunk::Buffer buffer = {
          /*element_count=*/ShapeUtil::ElementsIn(operand_shape),
          /*source_buffer=*/ShapedSlice{source_slice, operand_shape},
          /*destination_buffer=*/ShapedSlice{result_slice, result_buffer_shape},
          /*source_memory_space=*/src_memory_space,
          /*destination_memory_space=*/dst_memory_space};
      buffers.push_back(buffer);
    }
  }
  if (!CollectivePermuteStartThunk::IsDegenerate(instr, replica_count,
                                                 partition_count)) {
    if (IsNvshmemCollective(instr)) {
      // Note: xla_gpu_use_memcpy_local_p2p flag won't be used for now since the
      // NVSHMEM collective permute thunk doesn't perform any memcpy operations
      // at the moment.
      auto thunk = std::make_unique<NvshmemCollectivePermuteStartThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          instr, replica_count, partition_count, buffers,
          ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p(),
          GetStreamKindForP2P(instr));
      GetCollectivesAsyncEvents().try_emplace(instr, thunk->async_events());
      thunks.push_back(std::move(thunk));
    } else {
      auto thunk = std::make_unique<CollectivePermuteStartThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          instr, replica_count, partition_count, buffers,
          ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p());
      GetCollectivesAsyncEvents().try_emplace(instr, thunk->async_events());
      thunks.push_back(std::move(thunk));
    }
  }
  return thunks;
}

template <typename CollectiveThunkType, typename HloInstType>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectiveThunk(
    Thunk::Kind kind, const HloInstruction* async_start,
    const HloInstType* inst, std::optional<bool> use_global_device_ids) {
  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  int64_t replica_count = hlo_config.replica_count();
  int64_t partition_count = hlo_config.num_partitions();
  int64_t operand_count = inst->operand_count();
  VLOG(2) << CollectiveThunkType::GetHloOpName()
          << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << operand_count;

  // Stash relevant information in CollectiveThunk::Buffer even if
  // we may not generate an CollectiveThunk.
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(operand_count);

  // Adds a source and destination buffers pair to `buffers`.
  auto add_buffer = [&](const HloInstruction* src, const HloInstruction* dst,
                        const ShapeIndex& dst_shape_index) -> absl::Status {
    const Shape& src_shape = src->shape();
    const Shape& dst_shape =
        ShapeUtil::GetSubshape(dst->shape(), dst_shape_index);
    TF_ASSIGN_OR_RETURN(auto src_slice, GetAllocationSliceForHlo(src));
    TF_ASSIGN_OR_RETURN(auto dst_slice,
                        GetAllocationSliceForHlo(dst, dst_shape_index));

    buffers.push_back(CollectiveThunk::Buffer{
        /*element_count=*/ShapeUtil::ElementsIn(src_shape),
        /*source_buffer=*/{src_slice, src_shape},
        /*destination_buffer=*/{dst_slice, dst_shape},
        /*source_memory_space=*/src_shape.layout().memory_space(),
        /*destination_memory_space=*/dst_shape.layout().memory_space()});
    return absl::OkStatus();
  };

  if (kind == Thunk::Kind::kAllGatherStart) {
    // Start operations return a tuple of (<<inputs>>, <<outputs>>)
    // where outputs can be a tuple itself (if operation has
    // multiple operands).
    for (int64_t i = 0; i < operand_count; i++) {
      ShapeIndex idx = operand_count > 1 ? ShapeIndex({1, i}) : ShapeIndex({1});
      TF_RETURN_IF_ERROR(add_buffer(inst->operand(i), inst, idx));
    }
  } else if (kind == Thunk::Kind::kRaggedAllToAll) {
    // RaggedAllToAll operation has 6 operands: input, output,
    // input_offset, send_size, output_offset, recv_size. `output`
    // operand is aliased with the instruction result. All other
    // operands are not aliased.
    TF_RETURN_IF_ERROR(
        add_buffer(inst->operand(0), inst->operand(0), ShapeIndex({})));
    TF_RETURN_IF_ERROR(add_buffer(inst->operand(1), inst, ShapeIndex({})));

    for (int64_t i = 2; i < operand_count; i++) {
      TF_RETURN_IF_ERROR(
          add_buffer(inst->operand(i), inst->operand(i), ShapeIndex({})));
    }
  } else {
    // For other operations simply zip operands with results.
    for (int64_t i = 0; i < operand_count; i++) {
      ShapeIndex idx =
          inst->shape().IsTuple() ? ShapeIndex({i}) : ShapeIndex({});

      TF_RETURN_IF_ERROR(add_buffer(inst->operand(i), inst, idx));
    }
  }

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
  bool is_degenerate = kind != Thunk::Kind::kRaggedAllToAll &&
                       GetCollectiveConfig(inst, use_global_device_ids)
                           .IsDegenerate(replica_count, partition_count);

  if (is_degenerate) {
    return EmitDegeneratedCollectiveThunk(buffers, async_start, inst);
  }

  TF_RETURN_IF_ERROR(CollectiveThunkType::CheckImplementable(
      inst, replica_count, partition_count));

  auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      inst, ir_emitter_context_->GetNextThunkId());
  // The wrapper name is used when syntactic sugar is turned on.
  if (ir_emitter_context_->debug_options().xla_syntax_sugar_async_ops()) {
    thunk_info.profile_annotation = async_start->name();
  }
  std::unique_ptr<CollectiveThunkType> thunk;
  // TODO(b/828435206) Remove this constexpr once collective kernel thunk is
  // lifted out of the all reduce thunk.
  if constexpr (kRequiresCollectiveKernelThunk<CollectiveThunkType>) {
    TF_ASSIGN_OR_RETURN(
        auto emit_result,
        EmitCollectiveKernelThunk(
            ir_emitter_context_, call_graph_.get(), thunk_info, buffers,
            Cast<HloAllReduceInstruction>(inst), GetAllReduceConfigInst(inst)));
    if (emit_result.llvm_module != nullptr) {
      kernel_modules_.push_back(std::move(emit_result.llvm_module));
    }
    thunk = std::make_unique<CollectiveThunkType>(
        thunk_info, inst, /*buffers=*/std::move(buffers),
        std::move(emit_result.thunk),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p());
  } else {
    thunk = std::make_unique<CollectiveThunkType>(
        thunk_info, inst, /*buffers=*/std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p());
  }
  GetCollectivesAsyncEvents().insert({async_start, thunk->async_events()});
  return GetThunkSequence(std::move(thunk));
}

std::vector<const HloInstruction*> GetRealDependencyInstructions(
    const HloInstruction* instr) {
  std::vector<const HloInstruction*> real_deps;
  switch (instr->opcode()) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
      return {FindCanonicalSendRecvStartOp(instr)};
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopyDone:
      return {instr->operand(0)};
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kCustomCall:
    case HloOpcode::kFusion:
    case HloOpcode::kCopy:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPartitionId:
    case HloOpcode::kFft:
    case HloOpcode::kReplicaId:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kSort:
    case HloOpcode::kWhile:
    case HloOpcode::kCopyStart:
      return {instr};
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kTuple:
      for (const HloInstruction* operand : instr->operands()) {
        auto deps = GetRealDependencyInstructions(operand);
        real_deps.insert(real_deps.end(), deps.begin(), deps.end());
      }
      return real_deps;
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement: {
      auto deps = GetRealDependencyInstructions(instr->operand(0));
      real_deps.insert(real_deps.end(), deps.begin(), deps.end());
    }
      return real_deps;
    default:
      return {};
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectiveGroupStartThunk(
    const HloInstruction* instr) {
  ThunkSequence thunks;
  for (const HloInstruction* nested_instruction :
       instr->async_wrapped_computation()->instructions()) {
    ASSIGN_OR_RETURN(
        auto comp_thunks,
        EmitHloInstruction(nested_instruction, /*emit_group_thunks=*/true));
    AppendThunkSequence(thunks, comp_thunks);
  }
  auto thunk = std::make_unique<CollectiveGroupThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      Thunk::Kind::kGroupStart, std::move(thunks));

  GetCollectivesAsyncEvents().insert({instr, thunk->async_events()});
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectiveAsyncDone(
    Thunk::Kind kind, const HloInstruction* inst) {
  // Partial pipelining is only implemented for send/recv.
  bool is_send_recv =
      kind == Thunk::Kind::kRecvDone || kind == Thunk::Kind::kSendDone;
  const HloInstruction* start =
      is_send_recv ? FindCanonicalSendRecvStartOp(inst) : inst->operand(0);

  // Find canonical async event.
  CollectivesAsyncEvents& collectives_async_events =
      GetCollectivesAsyncEvents();
  auto async_events_it = collectives_async_events.find(start);
  TF_RET_CHECK(async_events_it != collectives_async_events.end())
      << "couldn't find async events for start operation";

  // Can be null if no start thunk was created (e.g. if the start op
  // is degenerate), in which case there's nothing to do here.
  if (!async_events_it->second) {
    return ThunkSequence{};
  }

  return GetThunkSequence(std::make_unique<CollectiveDoneThunk>(
      kind,
      Thunk::ThunkInfo::WithProfileAnnotation(
          inst, ir_emitter_context_->GetNextThunkId()),
      async_events_it->second));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNvshmemAsyncDone(
    Thunk::Kind kind, const HloInstruction* inst) {
  bool is_send_recv = kind == Thunk::Kind::kNvshmemRecvDone ||
                      kind == Thunk::Kind::kNvshmemSendDone;
  const HloInstruction* start =
      is_send_recv ? FindCanonicalSendRecvStartOp(inst) : inst->operand(0);

  // Find canonical async event.
  CollectivesAsyncEvents& collectives_async_events =
      GetCollectivesAsyncEvents();
  auto async_events_it = collectives_async_events.find(start);
  TF_RET_CHECK(async_events_it != collectives_async_events.end())
      << "couldn't find async events for start operation";

  // Can be null if no start thunk was created (e.g. if the start op is
  // degenerate), in which case there's nothing to do here.
  if (!async_events_it->second) {
    return ThunkSequence{};
  }

  AsyncStreamKind stream_kind = AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE;
  if (is_send_recv) {
    stream_kind = GetStreamKindForP2P(start);
  }

  if (kind == Thunk::Kind::kNvshmemCollectivePermuteDone) {
    return GetThunkSequence(std::make_unique<NvshmemCollectivePermuteDoneThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            inst, ir_emitter_context_->GetNextThunkId()),
        async_events_it->second, stream_kind));
  }
  return GetThunkSequence(std::make_unique<NvshmemCollectiveDoneThunk>(
      kind,
      Thunk::ThunkInfo::WithProfileAnnotation(
          inst, ir_emitter_context_->GetNextThunkId()),
      async_events_it->second, stream_kind));
}

template <typename NvshmemAllReduceThunkType, typename HloAllReduceInstruction>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNvshmemThunk(
    Thunk::Kind kind, const HloInstruction* async_start,
    const HloAllReduceInstruction* inst,
    std::optional<bool> use_global_device_ids) {
  CHECK(kind == Thunk::Kind::kNvshmemAllReduceStart);
  const auto& hlo_config = ir_emitter_context_->hlo_module().config();
  int64_t replica_count = hlo_config.replica_count();
  int64_t partition_count = hlo_config.num_partitions();
  VLOG(2) << NvshmemAllReduceThunkType::GetHloOpName()
          << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << inst->operand_count();

  // A given collective op can be degenerate if across all groups formed
  // by it are singleton. In such a case, we don't need to do any communication
  // and we can just copy the input to the output.
  bool is_degenerate = GetCollectiveConfig(inst, use_global_device_ids)
                           .IsDegenerate(replica_count, partition_count);
  absl::Status implementable_status =
      NvshmemAllReduceThunkType::CheckImplementable(inst, replica_count,
                                                    partition_count);
  bool should_use_nvshmem_thunk = !is_degenerate && implementable_status.ok();

  // Stash relevant information in CollectiveThunk::Buffer even if we may
  // not generate an NvshmemCollectiveThunk.
  std::vector<CollectiveThunk::Buffer> buffers;

  int64_t operand_count = inst->operand_count();
  buffers.reserve(operand_count);

  // Adds a source and destination buffers pair to `buffers`.
  auto add_buffer = [&](int64_t element_count, const ShapedSlice& src,
                        int64_t src_memory_space, const ShapedSlice& dst,
                        int64_t dst_memory_space) {
    buffers.push_back(
        CollectiveThunk::Buffer{/*element_count=*/element_count,
                                /*source_buffer=*/src,
                                /*destination_buffer=*/dst,
                                /*source_memory_space=*/src_memory_space,
                                /*destination_memory_space=*/dst_memory_space});
  };

  // For other operations simply zip operands with results.
  for (int64_t i = 0; i < operand_count; i++) {
    ShapeIndex idx = operand_count > 1 ? ShapeIndex({i}) : ShapeIndex({});
    const Shape& src_shape = inst->operand(i)->shape();
    const Shape& dst_shape = ShapeUtil::GetSubshape(inst->shape(), idx);
    TF_ASSIGN_OR_RETURN(auto src, GetAllocationSliceForHlo(inst->operand(i)));
    TF_ASSIGN_OR_RETURN(auto dst, GetAllocationSliceForHlo(inst, idx));
    add_buffer(ShapeUtil::ElementsIn(src_shape), {src, src_shape},
               src_shape.layout().memory_space(), {dst, dst_shape},
               dst_shape.layout().memory_space());
  }

  if (should_use_nvshmem_thunk) {
    auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
        inst, ir_emitter_context_->GetNextThunkId());
    // The wrapper name is used when syntactic sugar is turned on.
    if (ir_emitter_context_->debug_options().xla_syntax_sugar_async_ops()) {
      thunk_info.profile_annotation = async_start->name();
    }
    auto thunk = std::make_unique<NvshmemAllReduceThunkType>(
        thunk_info, inst, /*buffers=*/std::move(buffers),
        ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p());
    GetCollectivesAsyncEvents().insert({async_start, thunk->async_events()});
    return GetThunkSequence(std::move(thunk));
  }

  if (!is_degenerate) {
    return implementable_status;
  }
  return EmitDegeneratedCollectiveThunk(buffers, async_start, inst);
}

template <typename HloInstType>
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitDegeneratedCollectiveThunk(
    std::vector<CollectiveThunk::Buffer>& buffers,
    const HloInstruction* async_start, const HloInstType* inst) {
  // Signal that start thunk not created with nullptr.
  GetCollectivesAsyncEvents().insert({async_start, nullptr});

  // Degenerate collectives are simply identity function. Buffer
  // assignment expects a copy, so that's what we do.
  ThunkSequence thunks;
  for (int64_t i = 0; i < buffers.size(); i++) {
    const Shape shape = inst->operand(i)->shape();
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            inst, ir_emitter_context_->GetNextThunkId()),
        ShapedSlice{buffers[i].source_buffer.slice, shape},
        ShapedSlice{buffers[i].destination_buffer.slice, shape},
        ShapeUtil::ByteSizeOf(shape)));
  }
  if (thunks.size() == 1) {
    return thunks;
  }
  return GetThunkSequence(std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          inst, ir_emitter_context_->GetNextThunkId()),
      std::move(thunks)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitInfeed(
    const HloInfeedInstruction* instr) {
  // Infeed instruction returns a tuple containing the result data
  // and a token. We only need the result data to construct the
  // infeed thunk.
  std::vector<ShapedSlice> shaped_slices;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple() || subshape.IsToken()) return absl::OkStatus();
        if (subshape.IsArray()) {
          TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                              GetAllocationSliceForHlo(instr, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        instr->ToString(), index.ToString());
      }));

  return GetThunkSequence(std::make_unique<InfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(shaped_slices)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitOutfeed(
    const HloOutfeedInstruction* instr) {
  // HLO outfeed instruction has 2 operands, the source and a token,
  // and a single token output.
  const HloInstruction* source = instr->operand(0);
  std::vector<ShapedSlice> shaped_slices;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      source->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple()) return absl::OkStatus();
        if (subshape.IsArray()) {
          TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data,
                              GetAllocationSliceForHlo(source, index));
          ShapedSlice shaped_slice = {data, subshape};
          shaped_slices.push_back(shaped_slice);
          return absl::OkStatus();
        }
        return Internal("Unexpected shape kind for %s and shape index %s",
                        source->ToString(), index.ToString());
      }));

  return GetThunkSequence(std::make_unique<OutfeedThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      std::move(shaped_slices)));
}

static absl::flat_hash_map<std::string, std::string> ConvertFrontendAttributes(
    const FrontendAttributes& attrs) {
  absl::flat_hash_map<std::string, std::string> result;
  for (auto& [k, v] : attrs.map()) {
    result[k] = v;
  }
  return result;
}

static std::optional<GlobalDeviceId> DeviceConstraint(
    const HloInstruction* hlo) {
  if (hlo->has_sharding() && hlo->sharding().HasUniqueDevice()) {
    return GlobalDeviceId(hlo->sharding().GetUniqueDevice());
  }
  return std::nullopt;
}

absl::StatusOr<bool> ShapeHasHostMemorySpace(Shape shape, int index,
                                             int host_memory_space) {
  return shape.tuple_shapes(index).has_layout() &&
         shape.tuple_shapes(index).layout().memory_space() == host_memory_space;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyStartThunk(
    const HloCopyStartInstruction* copy_start_instr) {
  // copy-start has a tuple shape: {host, device, context},
  // or {device, host, context}.
  // Only the destination shape is needed to get the output buffer.
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(copy_start_instr,
                                               /*index=*/{0}));

  const HloInstruction* src = copy_start_instr->operand(0);
  const Shape& input_shape = src->shape();
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(src, {}));
  const Shape& shape = copy_start_instr->shape();
  CHECK(shape.IsTuple());
  int host_memory_space = static_cast<int>(stream_executor::MemorySpace::kHost);
  TF_ASSIGN_OR_RETURN(bool is_dst_host_memory,
                      ShapeHasHostMemorySpace(shape, 0, host_memory_space));
  TF_ASSIGN_OR_RETURN(bool is_src_host_memory,
                      ShapeHasHostMemorySpace(shape, 1, host_memory_space));
  if (is_dst_host_memory == is_src_host_memory) {
    return absl::InternalError(
        absl::StrFormat("Copy-start %s doesn't have correct host memory space "
                        "color S(%d)",
                        copy_start_instr->ToString(),
                        static_cast<int>(stream_executor::MemorySpace::kHost)));
  }
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(
      ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
      stream_assignment.GetAsyncExecutionStreamIds(copy_start_instr));
  // Insert a waitFor() thunk for asynchronous memcpy only when the
  // source and destination stream IDs differ. If the IDs are the
  // same, the memcpy operation is synchronous within that stream.
  ThunkSequence thunks;
  if (streams.destination_stream_id != streams.source_stream_id) {
    thunks.push_back(std::make_unique<WaitForStreamsThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            copy_start_instr, ir_emitter_context_->GetNextThunkId()),
        streams.destination_stream_id, streams.source_stream_id));
  }
  if (is_dst_host_memory) {
    auto thunk = std::make_unique<DeviceToHostCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            copy_start_instr, ir_emitter_context_->GetNextThunkId()),
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape),
        /*copy_events=*/copy_events_,
        /*copy_start_instr=*/copy_start_instr);
    thunk->set_execution_stream_id(streams.destination_stream_id);
    thunks.push_back(std::move(thunk));
  } else {
    auto thunk = std::make_unique<HostToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            copy_start_instr, ir_emitter_context_->GetNextThunkId()),
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape),
        /*copy_events=*/copy_events_,
        /*copy_start_instr=*/copy_start_instr);
    thunk->set_execution_stream_id(streams.destination_stream_id);
    thunks.push_back(std::move(thunk));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyDoneThunk(
    const HloInstruction* instr) {
  const HloInstruction* copy_start_instr = instr->operand(0);
  CHECK(copy_start_instr->opcode() == HloOpcode::kCopyStart);

  return GetThunkSequence(std::make_unique<CopyDoneThunk>(
      Thunk::kCopyDone,
      Thunk::ThunkInfo::WithProfileAnnotation(
          copy_start_instr, ir_emitter_context_->GetNextThunkId()),
      /*copy_events=*/copy_events_,
      /*copy_start_instr=*/copy_start_instr));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSendThunk(
    const HloSendInstruction* instr, bool emit_group_thunks) {
  const HloInstruction* src = instr->operand(0);
  TF_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(src, {}));
  if (!instr->is_host_transfer()) {
    const auto& hlo_config = ir_emitter_context_->hlo_module().config();
    const int64_t replica_count = hlo_config.replica_count();
    const int64_t partition_count = hlo_config.num_partitions();
    const int64_t memory_space =
        instr->shape().IsTuple()
            ? instr->shape().tuple_shapes(0).layout().memory_space()
            : instr->shape().layout().memory_space();

    std::unique_ptr<Thunk> thunk;
    const CollectiveThunk::Buffer buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(src->shape()),
        /*source_buffer=*/slice,
        /*destination_buffer=*/slice,
        /*source_memory_space=*/memory_space,
        /*destination_memory_space=*/memory_space};
    if (IsNvshmemCollective(instr)) {
      thunk = std::make_unique<NvshmemSendThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          instr, replica_count, partition_count, buffer,
          nvshmem_buffer_addresses_);
    } else {
      thunk = std::make_unique<SendThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          instr, replica_count, partition_count, buffer);
    }
    CollectivesAsyncEvents& collectives_async_events =
        GetCollectivesAsyncEvents();

    // Wire up async events if the send thunk isn't emitted as a
    // part of a group thunk.
    if (!emit_group_thunks) {
      const HloInstruction* canonical_send_instr =
          FindCanonicalSendRecvStartOp(instr);
      if (collectives_async_events.contains(canonical_send_instr)) {
        if (IsNvshmemCollective(instr)) {
          tsl::down_cast<NvshmemSendThunk*>(thunk.get())
              ->set_async_events(
                  collectives_async_events[canonical_send_instr]);
        } else {
          tsl::down_cast<SendThunk*>(thunk.get())
              ->set_async_events(
                  collectives_async_events[canonical_send_instr]);
        }
      } else {
        if (IsNvshmemCollective(instr)) {
          collectives_async_events.try_emplace(
              instr,
              tsl::down_cast<NvshmemSendThunk*>(thunk.get())->async_events());
        } else {
          collectives_async_events.try_emplace(
              instr, tsl::down_cast<SendThunk*>(thunk.get())->async_events());
        }
      }
    }
    return GetThunkSequence(std::move(thunk));
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send instruction");
  }

  return GetThunkSequence(std::make_unique<HostSendThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      src->shape(), slice.slice, *instr->channel_id(), send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSendDoneThunk(
    const HloSendDoneInstruction* instr) {
  if (!instr->is_host_transfer()) {
    if (IsNvshmemCollective(instr)) {
      return EmitNvshmemAsyncDone(Thunk::kNvshmemSendDone, instr);
    }
    return EmitCollectiveAsyncDone(Thunk::kSendDone, instr);
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer send done "
        "instruction");
  }

  return GetThunkSequence(std::make_unique<HostSendDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      *instr->channel_id(), send_recv_events_, DeviceConstraint(instr)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRecvThunk(
    const HloRecvInstruction* instr, bool emit_group_thunks) {
  TF_RET_CHECK(instr->shape().IsTuple());
  TF_ASSIGN_OR_RETURN(ShapedSlice slice, GetShapedSliceForHlo(instr, {0}));

  if (!instr->is_host_transfer()) {
    const auto& hlo_config = ir_emitter_context_->hlo_module().config();
    const int64_t replica_count = hlo_config.replica_count();
    const int64_t partition_count = hlo_config.num_partitions();

    const int64_t memory_space =
        instr->shape().IsTuple()
            ? instr->shape().tuple_shapes(0).layout().memory_space()
            : instr->shape().layout().memory_space();

    std::unique_ptr<Thunk> thunk;
    const CollectiveThunk::Buffer buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(instr->shape().tuple_shapes(0)),
        /*source_buffer=*/slice,
        /*destination_buffer=*/slice,
        /*source_memory_space=*/memory_space,
        /*destination_memory_space=*/memory_space};
    if (IsNvshmemCollective(instr)) {
      thunk = std::make_unique<NvshmemRecvThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          instr, replica_count, partition_count, buffer,
          nvshmem_buffer_addresses_);
    } else {
      thunk = std::make_unique<RecvThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          instr, replica_count, partition_count, buffer);
    }
    CollectivesAsyncEvents& collectives_async_events =
        GetCollectivesAsyncEvents();

    // Wire up async events.
    if (!emit_group_thunks) {
      const HloInstruction* canonical_recv_instr =
          FindCanonicalSendRecvStartOp(instr);
      if (collectives_async_events.contains(canonical_recv_instr)) {
        if (IsNvshmemCollective(instr)) {
          tsl::down_cast<NvshmemRecvThunk*>(thunk.get())
              ->set_async_events(
                  collectives_async_events[canonical_recv_instr]);
        } else {
          tsl::down_cast<RecvThunk*>(thunk.get())
              ->set_async_events(
                  collectives_async_events[canonical_recv_instr]);
        }
      } else {
        if (IsNvshmemCollective(instr)) {
          collectives_async_events.try_emplace(
              instr,
              tsl::down_cast<NvshmemRecvThunk*>(thunk.get())->async_events());
        } else {
          collectives_async_events.try_emplace(
              instr, tsl::down_cast<RecvThunk*>(thunk.get())->async_events());
        }
      }
    }
    return GetThunkSequence(std::move(thunk));
  }

  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv instruction");
  }

  return GetThunkSequence(std::make_unique<HostRecvThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      instr->shape().tuple_shapes()[0], slice.slice, *instr->channel_id(),
      send_recv_events_,
      ConvertFrontendAttributes(instr->frontend_attributes()),
      DeviceConstraint(instr)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitRecvDoneThunk(
    const HloRecvDoneInstruction* instr) {
  if (!instr->is_host_transfer()) {
    if (IsNvshmemCollective(instr)) {
      return EmitNvshmemAsyncDone(Thunk::kNvshmemRecvDone, instr);
    }
    return EmitCollectiveAsyncDone(Thunk::kRecvDone, instr);
  }
  if (!instr->channel_id().has_value()) {
    return absl::InternalError(
        "Unknown channel id in host transfer recv done "
        "instruction");
  }
  return GetThunkSequence(std::make_unique<HostRecvDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      *instr->channel_id(), send_recv_events_, DeviceConstraint(instr)));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncDone(
    const HloInstruction* instr) {
  if (!instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    return EmitCollectiveAsyncDone(Thunk::kGroupDone, instr);
  }
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  ThunkSequence thunks;
  switch (wrapped->opcode()) {
    case HloOpcode::kReduceScatter:
      return EmitCollectiveAsyncDone(Thunk::kReduceScatterDone, instr);
    case HloOpcode::kAllToAll:
      return EmitCollectiveAsyncDone(Thunk::kAllToAllDone, instr);
    case HloOpcode::kRaggedAllToAll:
      return EmitCollectiveAsyncDone(Thunk::kRaggedAllToAllDone, instr);
    case HloOpcode::kCollectiveBroadcast:
      return EmitCollectiveAsyncDone(Thunk::kCollectiveBroadcastDone, instr);
    case HloOpcode::kCollectivePermute:
      return EmitCollectiveAsyncDone(Thunk::kCollectivePermuteDone, instr);
    case HloOpcode::kRecv:
      return EmitCollectiveAsyncDone(Thunk::kRecvDone, instr);
    case HloOpcode::kSend:
      return EmitCollectiveAsyncDone(Thunk::kSendDone, instr);
    case HloOpcode::kFusion: {
      auto collective_hero = GetCollectiveHeroForDynamicSliceFusion(
          Cast<HloFusionInstruction>(wrapped));
      if (collective_hero.has_value()) {
        switch ((*collective_hero)->opcode()) {
          case HloOpcode::kReduceScatter: {
            TF_ASSIGN_OR_RETURN(
                auto async_done_thunks,
                EmitCollectiveAsyncDone(Thunk::kReduceScatterDone, instr));
            AppendThunkSequence(thunks, async_done_thunks);
            break;
          }
          default:
            return absl::InternalError(
                absl::StrFormat("Unhandled collective in dynamic slice fusion "
                                "instruction: %s",
                                (*collective_hero)
                                    ->fused_instructions_computation()
                                    ->ToString()));
        }
      }
      // We still want to emit the stream done thunk.
      [[clang::fallthrough]];
    }
    case HloOpcode::kCall:
    case HloOpcode::kCustomCall: {
      if (IsHostExecuteCustomCall(*wrapped)) {
        auto custom_call = Cast<HloCustomCallInstruction>(wrapped);

        auto async_events =
            GetInstructionToHostExecuteAsyncEvents().at(custom_call);

        thunks.push_back(std::make_unique<HostExecuteDoneThunk>(
            Thunk::ThunkInfo::WithProfileAnnotation(
                instr, ir_emitter_context_->GetNextThunkId()),
            async_events));
        return thunks;
      }
      // Wait until the concurrent stream has finished.
      auto* async_done = Cast<HloAsyncInstruction>(instr);
      const ExecutionStreamAssignment& stream_assignment =
          ir_emitter_context_->execution_stream_assignment();
      TF_ASSIGN_OR_RETURN(
          ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
          stream_assignment.GetAsyncExecutionStreamIds(async_done));
      thunks.push_back(std::make_unique<WaitForStreamsThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          streams.source_stream_id, streams.destination_stream_id));
      return thunks;
    }
    default:
      return Internal("Unsupported async done wrapped instruction: %s",
                      HloOpcodeString(wrapped->opcode()));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAsyncStart(
    const HloInstruction* instr) {
  // Multi-op async start will emit a NCCL group thunk.
  if (!instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    return EmitCollectiveGroupStartThunk(instr);
  }
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  switch (wrapped->opcode()) {
    case HloOpcode::kReduceScatter: {
      auto* reduce_scatter = Cast<HloReduceScatterInstruction>(wrapped);
      return EmitCollectiveThunk<ReduceScatterStartThunk,
                                 HloReduceScatterInstruction>(
          Thunk::kReduceScatter, instr, reduce_scatter,
          reduce_scatter->use_global_device_ids());
    }
    case HloOpcode::kAllToAll: {
      auto* all_to_all = Cast<HloAllToAllInstruction>(wrapped);
      return EmitCollectiveThunk<AllToAllStartThunk, HloAllToAllInstruction>(
          Thunk::kAllToAll, instr, all_to_all, std::nullopt);
    }
    case HloOpcode::kRaggedAllToAll: {
      auto* ragged_all_to_all = Cast<HloRaggedAllToAllInstruction>(wrapped);
      return EmitCollectiveThunk<RaggedAllToAllStartThunk,
                                 HloRaggedAllToAllInstruction>(
          Thunk::kRaggedAllToAll, instr, ragged_all_to_all, std::nullopt);
    }
    case HloOpcode::kCollectiveBroadcast: {
      auto* collective_broadcast =
          Cast<HloCollectiveBroadcastInstruction>(wrapped);
      return EmitCollectiveThunk<CollectiveBroadcastStartThunk,
                                 HloCollectiveBroadcastInstruction>(
          Thunk::kCollectiveBroadcast, instr, collective_broadcast,
          std::nullopt);
    }
    case HloOpcode::kFusion: {
      // We'll launch the fusion computation on a concurrent
      // stream. The concurrent stream needs to first wait until
      // the main stream has finished calculating any values
      // that may be used as inputs to the fusion computation.
      // We enforce this by inlining a `WaitForStreams` thunk.
      auto* async_start = Cast<HloAsyncInstruction>(instr);
      const ExecutionStreamAssignment& stream_assignment =
          ir_emitter_context_->execution_stream_assignment();
      TF_ASSIGN_OR_RETURN(
          ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
          stream_assignment.GetAsyncExecutionStreamIds(async_start));
      ThunkSequence thunks =
          GetThunkSequence(std::make_unique<WaitForStreamsThunk>(
              Thunk::ThunkInfo::WithProfileAnnotation(
                  instr, ir_emitter_context_->GetNextThunkId()),
              streams.destination_stream_id, streams.source_stream_id));

      TF_ASSIGN_OR_RETURN(ThunkSequence fusion_thunks,
                          EmitFusion(Cast<HloFusionInstruction>(wrapped)));
      AppendThunkSequence(thunks, fusion_thunks);
      return thunks;
    }
    case HloOpcode::kCall: {
      return EmitAsyncComputation(instr);
    }
    case HloOpcode::kCustomCall: {
      if (IsHostExecuteCustomCall(*wrapped)) {
        auto custom_call = Cast<HloCustomCallInstruction>(wrapped);

        std::unique_ptr<HloModule> hlo_module =
            ExtractComputationIntoNewModule(*custom_call->called_computation());

        // All offloaded computations are marked as host computations from
        // the perspective of the GPU backend. Since these will execute on
        // the main thread from the CPU backend perspective, we need to mark
        // them as such.
        for (auto* computation : hlo_module->computations()) {
          computation->SetExecutionThread(HloInstruction::kMainExecutionThread);
        }

        absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4>
            operand_slices;
        for (HloInstruction* operand : wrapped->operands()) {
          for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
            TF_ASSIGN_OR_RETURN(
                auto slice,
                ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                    operand, indexed.index));
            operand_slices.push_back({slice, indexed.shape});
          }
        }

        // Collect buffer slices for all results.
        absl::InlinedVector<HostExecuteStartThunk::SliceAndShape, 4>
            result_slices;
        for (auto& indexed : ShapeUtil::GetLeafShapes(wrapped->shape())) {
          TF_ASSIGN_OR_RETURN(
              auto slice,
              ir_emitter_context_->buffer_assignment().GetUniqueSlice(
                  wrapped, indexed.index));
          result_slices.push_back({slice, indexed.shape});
        }

        HostOffloadingExecutableProto host_offloading_executable_proto;
        *host_offloading_executable_proto.mutable_hlo_module() =
            hlo_module->ToProto();
        host_offloading_executable_proto.set_executable_type(
            HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT);

        TF_ASSIGN_OR_RETURN(
            auto thunk,
            HostExecuteStartThunk::Create(
                Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(host_offloading_executable_proto),
                std::move(operand_slices), std::move(result_slices)));

        auto async_events = thunk->async_events();

        auto [it, inserted] = GetInstructionToHostExecuteAsyncEvents().emplace(
            custom_call, async_events);
        if (!inserted) {
          return Internal(
              "Async events already exist for host offloading custom call "
              "%s.",
              custom_call->ToString());
        }
        return GetThunkSequence(std::move(thunk));
      }
      return EmitAsyncCustomCallStart(instr);
    }
    default:
      return Internal("Unsupported async start wrapped instruction: %s",
                      HloOpcodeString(wrapped->opcode()));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* hlo, bool emit_group_thunks) {
  switch (hlo->opcode()) {
    case HloOpcode::kAllGatherDone:
      return EmitCollectiveAsyncDone(Thunk::kAllGatherDone, hlo);
    case HloOpcode::kAllGatherStart: {
      auto* all_gather = Cast<HloAllGatherInstruction>(hlo);
      return EmitCollectiveThunk<AllGatherStartThunk, HloAllGatherInstruction>(
          Thunk::kAllGatherStart, all_gather, all_gather,
          all_gather->use_global_device_ids());
    }
    case HloOpcode::kAllReduceDone:
      return IsNvshmemCollective(hlo)
                 ? EmitNvshmemAsyncDone(Thunk::kNvshmemAllReduceDone, hlo)
                 : EmitCollectiveAsyncDone(Thunk::kAllReduceDone, hlo);
    case HloOpcode::kAllReduceStart: {
      auto* all_reduce = Cast<HloAllReduceInstruction>(hlo);
      if (IsNvshmemCollective(hlo)) {
        return EmitNvshmemThunk<NvshmemAllReduceStartThunk,
                                HloAllReduceInstruction>(
            Thunk::kNvshmemAllReduceStart, all_reduce, all_reduce,
            all_reduce->use_global_device_ids());
      }
      return EmitCollectiveThunk<AllReduceStartThunk, HloAllReduceInstruction>(
          Thunk::kAllReduceStart, all_reduce, all_reduce,
          all_reduce->use_global_device_ids());
    }
    case HloOpcode::kAsyncDone:
      return EmitAsyncDone(hlo);
    case HloOpcode::kAsyncStart:
      return EmitAsyncStart(hlo);
    case HloOpcode::kCall:
      return EmitCommandBufferThunk(hlo);
    case HloOpcode::kCollectivePermuteDone:
      return IsNvshmemCollective(hlo)
                 ? EmitNvshmemAsyncDone(Thunk::kNvshmemCollectivePermuteDone,
                                        hlo)
                 : EmitCollectiveAsyncDone(Thunk::kCollectivePermuteDone, hlo);
    case HloOpcode::kCollectivePermuteStart:
      return EmitCollectivePermute(Cast<HloCollectivePermuteInstruction>(hlo));
    case HloOpcode::kConditional:
      return EmitConditional(hlo);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(hlo));
    case HloOpcode::kCustomCall: {
      auto* custom_call = Cast<HloCustomCallInstruction>(hlo);
      if (IsLegacyCublasMatmul(*hlo)) {
        return EmitGemmThunk(custom_call);
      }
      if (IsCublasLtMatmul(*hlo)) {
        return EmitCublasLtMatmulThunk(custom_call);
      }
      if (IsCublasLtMatmulF8(*hlo)) {
        return EmitCublasLtMatmulThunkF8(custom_call);
      }
      if (IsCudnnConvolutionReorder(*hlo)) {
        return EmitConvolutionReorderThunk(custom_call);
      }
      if (IsCustomCallToDnnNorm(*hlo)) {
        return EmitNormThunk(custom_call);
      }
      if (IsCustomCallTofMHA(*hlo) || IsCustomCallTofMHAF8(*hlo) ||
          IsCustomCallToBlockScaledDot(*hlo)) {
        return EmitCuDnnThunk(custom_call);
      }
      if (IsCustomCallToPtxKernel(*hlo)) {
        return EmitPtxCustomCall(custom_call);
      }
      if (IsCustomCallToTopK(*hlo)) {
        return EmitTopKCustomCall(custom_call);
      }
      if (IsCustomCallToDnnConvolution(*hlo)) {
        return EmitConvolutionThunk(custom_call);
      }
      if (IsTriangularSolve(*hlo)) {
        return EmitTriangularSolveCustomCall(hlo);
      }
      if (IsCubDeviceRadixSort(*hlo)) {
        return EmitCubDeviceRadixSort(custom_call);
      }
      if (custom_call->custom_call_target() == "PadToStatic") {
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
        return ThunkSequence{};
      }
      if (hlo->custom_call_target() == kPinCustomCallTarget ||
          hlo->custom_call_target() == kUnpinCustomCallTarget ||
          hlo->custom_call_target() == kCreateBufferCustomCallTarget) {
        return ThunkSequence{};
      }
      if (hlo->custom_call_target() == kCollectiveMetadataCustomCallTarget) {
        return EmitCollectiveMetadata(hlo);
      }
      return EmitCustomCallThunk(custom_call);
    }
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
      return EmitFftThunk(Cast<HloFftInstruction>(hlo));

    case HloOpcode::kRecv:
      return EmitRecvThunk(Cast<HloRecvInstruction>(hlo), emit_group_thunks);
    case HloOpcode::kRecvDone:
      return EmitRecvDoneThunk(Cast<HloRecvDoneInstruction>(hlo));

    case HloOpcode::kReplicaId:
      return EmitReplicaOrPartitionId<ReplicaIdThunk>(hlo);
    case HloOpcode::kRngGetAndUpdateState:
      return EmitRngGetAndUpdateState(
          Cast<HloRngGetAndUpdateStateInstruction>(hlo));

    case HloOpcode::kSend:
      return EmitSendThunk(Cast<HloSendInstruction>(hlo), emit_group_thunks);
    case HloOpcode::kSendDone:
      return EmitSendDoneThunk(Cast<HloSendDoneInstruction>(hlo));

    case HloOpcode::kSort:
      return EmitSort(Cast<HloSortInstruction>(hlo));
    case HloOpcode::kWhile:
      return EmitWhile(hlo);
    case HloOpcode::kCopyStart:
      return EmitCopyStartThunk(Cast<HloCopyStartInstruction>(hlo));
    case HloOpcode::kCopyDone:
      return EmitCopyDoneThunk(hlo);

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
      return ThunkSequence{};
    default:
      return Internal("Unsupported instruction opcode: %s",
                      HloOpcodeString(hlo->opcode()));
  }
  return Internal("Unhandled HLO instruction");
}

absl::StatusOr<std::unique_ptr<SequentialThunk>>
ThunkEmitter::EmitHloEntryComputation(const HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto thunks,
                      EmitHloComputation(module->entry_computation()));
  return std::make_unique<SequentialThunk>(Thunk::ThunkInfo{},
                                           std::move(thunks));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloComputation(
    const HloComputation* computation) {
  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation)) {
    return Internal("Sequence not found for computation: %s",
                    computation->name());
  }

  ThunkSequence thunk_sequence;
  const HloInstructionSequence& sequence = schedule.sequence(computation);
  absl::flat_hash_map<const HloInstruction*, Thunk*> instr_to_thunk;
  for (const HloInstruction* instr : sequence.instructions()) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitHloInstruction(instr));
    if (!thunks.empty()) {
      instr_to_thunk[instr] = thunks.back().get();
    }
    AppendThunkSequence(thunk_sequence, thunks);
    for (const HloInstruction* control_predecessor :
         instr->control_predecessors()) {
      std::vector<const HloInstruction*> real_successors =
          GetRealDependencyInstructions(instr);
      std::vector<const HloInstruction*> real_predecessors =
          GetRealDependencyInstructions(control_predecessor);
      for (const HloInstruction* real_predecessor : real_predecessors) {
        for (const HloInstruction* real_successor : real_successors) {
          // if the instruction does not have a thunk, it is a degenerated
          // instruction, and we skip it.
          if (instr_to_thunk.contains(real_successor) &&
              instr_to_thunk.contains(real_predecessor)) {
            instr_to_thunk[real_successor]->add_control_predecessor(
                instr_to_thunk[real_predecessor]);
            VLOG(3) << "Add thunk control dependency for predecessor:  "
                    << instr_to_thunk[real_predecessor]->ToString(0)
                    << " successor: "
                    << instr_to_thunk[real_successor]->ToString(0);
          }
        }
      }
    }
  }
  return thunk_sequence;
}

}  // namespace xla::gpu

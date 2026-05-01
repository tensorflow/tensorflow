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
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
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
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/llvm/llvm_emitter.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/convolution_filter_thunk.pb.h"
#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"
#include "xla/backends/gpu/runtime/convolution_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/device_to_host_copy_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/fft_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/host_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/infeed_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/legacy_custom_call_thunk.h"
#include "xla/backends/gpu/runtime/norm_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_recv_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_send_thunk.h"
#include "xla/backends/gpu/runtime/outfeed_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/select_k_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/topk.h"
#include "xla/backends/gpu/runtime/triangular_solve_thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/ffi/attribute_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
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
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
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
                              std::optional<LaunchDimensions> launch_dimensions,
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
            /*launch_dimensions=*/launch_dimensions,
            /*shmem_bytes=*/shmem_bytes,
            /*is_multimem_enabled=*/false),
        std::move(local_module)};
  };
  TF_ASSIGN_OR_RETURN(bool did_set_config, TrySetGpuBackendConfigForCollective(
                                               device_info, fusion_instr));
  if (!did_set_config) {
    return make_thunk(/*kernel_name=*/"",
                      /*shmem_bytes=*/0,
                      /*launch_dimensions=*/std::nullopt,
                      /*local_module=*/nullptr);
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
  return make_thunk(
      result.kernel_thunk->kernel_name(), result.kernel_thunk->shmem_bytes(),
      result.kernel_thunk->launch_dimensions(), std::move(result.llvm_module));
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

}  // namespace

ThunkEmitter::ThunkEmitter(
    IrEmitterContext* absl_nonnull ir_emitter_context,
    llvm_ir::LLVMCommandLineOptionsReleasableLock* absl_nonnull
        llvm_options_lock)
    : ir_emitter_context_(ir_emitter_context),
      send_recv_events_(std::make_shared<HostSendRecvAsyncEvents>()),
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
  return ThunkSequence{};
}

ThunkSequence GetThunkSequence(std::unique_ptr<Thunk> ir_emitter) {
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(ir_emitter));
  return thunk_sequence;
}

AsyncThunkSequence ThunkEmitter::EmitConditional(const HloInstruction* instr) {
  std::vector<AsyncThunkSequence> branch_thunks;
  branch_thunks.reserve(instr->branch_count());
  for (HloComputation* comp : instr->branch_computations()) {
    branch_thunks.emplace_back(EmitHloComputation(comp));
  }
  ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                   GetAllocationSliceForHlo(instr->operand(0), {}));

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  ShapedSlice shaped_slice{slice, instr->operand(0)->shape()};
  return tsl::JoinFutures(absl::MakeSpan(branch_thunks))
      .Map([thunk_info = std::move(thunk_info),
            shaped_slice = std::move(shaped_slice)](
               std::vector<ThunkSequence> branch_thunks) {
        return GetThunkSequence(std::make_unique<ConditionalThunk>(
            std::move(thunk_info), std::move(shaped_slice),
            std::move(branch_thunks)));
      });
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
AsyncThunkSequence ThunkEmitter::EmitPadToStatic(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  ASSIGN_OR_RETURN(
      KernelDefinition<LlvmKernelSource> kernel_def,
      EmitPadToStaticLLVMIR(instr, ir_emitter_context_, kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](std::unique_ptr<Thunk> thunk) {
        return ThunkSequence::Of(std::move(thunk));
      });
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
AsyncThunkSequence ThunkEmitter::EmitSliceToDynamic(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));
  ASSIGN_OR_RETURN(
      KernelDefinition<LlvmKernelSource> kernel_def,
      EmitSliceToDynamicLLVMIR(instr, ir_emitter_context_, kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](std::unique_ptr<Thunk> thunk) {
        return ThunkSequence::Of(std::move(thunk));
      });
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
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));
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

  TF_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ShapedSlice c;
  if (has_matrix_bias) {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(2)));
  } else {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    TF_ASSIGN_OR_RETURN(
        bias, GetShapedSliceForHlo(instr->operand(has_matrix_bias ? 3 : 2)));
  }

  std::optional<ShapedSlice> aux;
  if (has_aux_output) {
    TF_ASSIGN_OR_RETURN(aux, GetShapedSliceForHlo(instr, {1}));
  }

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().IsTuple() &&
      (instr->shape().tuple_shapes().size() - has_aux_output - 1)) {
    TF_RET_CHECK(
        (has_aux_output && instr->shape().tuple_shapes().size() == 3) ||
        (!has_aux_output && instr->shape().tuple_shapes().size() == 2));
    TF_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(instr, {instr->shape().tuple_shapes_size() - 1}));
  }

  TF_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  TF_ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                      gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  std::string canonical_hlo = instr->ToString(
      HloPrintOptions::Fingerprint().set_print_backend_config(true));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      bias, aux, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulThunkF8(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() > 3 && instr->operand_count() < 8);
  ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  ASSIGN_OR_RETURN(bool has_vector_bias,
                   xla::gpu::gpublas_lt::EpilogueAddsVectorBias(epilogue));

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  ASSIGN_OR_RETURN(bool has_aux_output,
                   xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));

  ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ShapedSlice c;
  bool has_matrix_bias = config.beta() != 0;
  if (has_matrix_bias) {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr->operand(2)));
  } else {
    ASSIGN_OR_RETURN(c, GetShapedSliceForHlo(instr, output_index));
  }
  ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  int a_scale_index = has_matrix_bias ? 3 : 2;
  ASSIGN_OR_RETURN(ShapedSlice a_scale,
                   GetShapedSliceForHlo(instr->operand(a_scale_index)));
  ASSIGN_OR_RETURN(ShapedSlice b_scale,
                   GetShapedSliceForHlo(instr->operand(a_scale_index + 1)));

  bool is_cuda = ir_emitter_context_->gpu_compute_capability().IsCuda();
  bool is_fp8 = instr->shape().tuple_shapes(0).element_type() == F8E4M3FN ||
                instr->shape().tuple_shapes(0).element_type() == F8E5M2;
  // cublasLT requires c_scale/d_scale to be null when C/D is not
  // FP8. Currently, C cannot be FP8.
  std::optional<ShapedSlice> d_scale;
  if (is_cuda && is_fp8) {
    ASSIGN_OR_RETURN(d_scale, GetShapedSliceForHlo(instr->operands().back()));
  }

  std::optional<ShapedSlice> bias;
  if (has_vector_bias) {
    ASSIGN_OR_RETURN(bias,
                     GetShapedSliceForHlo(instr->operand(a_scale_index + 2)));
  }

  std::optional<ShapedSlice> d_amax;
  if (config.damax_output()) {
    ASSIGN_OR_RETURN(d_amax, GetShapedSliceForHlo(instr, {1}));
  }

  ASSIGN_OR_RETURN(
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
    ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(instr, {instr->shape().tuple_shapes_size() - 1}));
  }

  ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  std::string canonical_hlo = instr->ToString(
      HloPrintOptions::Fingerprint().set_print_backend_config(true));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      bias, std::nullopt, a_scale, b_scale, std::nullopt, d_scale, d_amax,
      workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtGroupedMatmulThunk(
    const HloCustomCallInstruction* instr) {
  ASSIGN_OR_RETURN(const auto gpu_config,
                   instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config =
      gpu_config.grouped_gemm_backend_config().gemm_backend_config();

  TF_RET_CHECK(instr->operand_count() == 3);

  xla::ShapeIndex output_index =
      instr->shape().IsTuple() ? xla::ShapeIndex{0} : xla::ShapeIndex{};

  TF_ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  TF_ASSIGN_OR_RETURN(ShapedSlice group_sizes,
                      GetShapedSliceForHlo(instr->operand(2)));
  // No bias
  TF_ASSIGN_OR_RETURN(ShapedSlice c, GetShapedSliceForHlo(instr, output_index));
  TF_ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().IsTuple() && (instr->shape().tuple_shapes().size() - 1)) {
    TF_ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(instr, {instr->shape().tuple_shapes_size() - 1}));
  }
  TF_ASSIGN_OR_RETURN(
      auto gemm_config,
      GroupedGemmConfig::For(static_cast<const HloInstruction*>(instr),
                             ir_emitter_context_->gpu_compute_capability()));

  // Use the first algorithm by default (i.e. fastest according to
  // heuristics).
  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  std::string canonical_hlo = instr->ToString(
      HloPrintOptions::Fingerprint().set_print_backend_config(true));

  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      se::gpu::BlasLt::Epilogue::kDefault, algorithm,
      config.autotune_workspace_size(), a, b, c, d, group_sizes, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, workspace_buffer);
  return GetThunkSequence(std::move(thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCublasLtMatmulThunkMx(
    const HloCustomCallInstruction* instr) {
  TF_RET_CHECK(instr->operand_count() == 4);
  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const xla::gpu::GemmBackendConfig& config = gpu_config.gemm_backend_config();
  xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();

  TF_RET_CHECK(instr->shape().IsTuple());
  xla::ShapeIndex output_index = xla::ShapeIndex{0};

  ASSIGN_OR_RETURN(ShapedSlice a, GetShapedSliceForHlo(instr->operand(0)));
  ASSIGN_OR_RETURN(ShapedSlice b, GetShapedSliceForHlo(instr->operand(1)));
  ASSIGN_OR_RETURN(ShapedSlice a_scale,
                   GetShapedSliceForHlo(instr->operand(2)));
  ASSIGN_OR_RETURN(ShapedSlice b_scale,
                   GetShapedSliceForHlo(instr->operand(3)));

  ASSIGN_OR_RETURN(ShapedSlice c, GetShapedSliceForHlo(instr, output_index));
  ASSIGN_OR_RETURN(ShapedSlice d, GetShapedSliceForHlo(instr, output_index));

  ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(instr, ir_emitter_context_->gpu_compute_capability()));

  int64_t algorithm =
      config.algorithm_case() == GemmBackendConfig::kSelectedAlgorithm
          ? config.selected_algorithm()
          : 0;

  std::optional<ShapedSlice> workspace_buffer;
  if (instr->shape().tuple_shapes().size() == 2) {
    ASSIGN_OR_RETURN(
        workspace_buffer,
        GetShapedSliceForHlo(instr, {instr->shape().tuple_shapes_size() - 1}));
  }

  ASSIGN_OR_RETURN(se::gpu::BlasLt::Epilogue blas_lt_epilogue,
                   gpublas_lt::AsBlasLtEpilogue(epilogue));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  std::string canonical_hlo = instr->ToString(
      HloPrintOptions::Fingerprint().set_print_backend_config(true));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
      blas_lt_epilogue, algorithm, config.autotune_workspace_size(), a, b, c, d,
      /*bias=*/std::nullopt, /*aux=*/std::nullopt, a_scale, b_scale,
      /*c_scale=*/std::nullopt, /*d_scale=*/std::nullopt,
      /*d_amax=*/std::nullopt, workspace_buffer);
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
      kernel_arguments.GetArgumentShapedSlices(),
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
      TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
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
        ir_emitter_context_->cpu_target_machine_options());
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
  mlir::MLIRContext& mlir_context = *ir_emitter_context_->mlir_context();

  auto generate = [this, &instr,
                   &mlir_context]() -> absl::StatusOr<KernelReuseCache::Entry> {
    LoadMlirDialectsForTriton(mlir_context);
    TritonCall call =
        TritonCall::Parse(instr->raw_backend_config_string(), &mlir_context);
    std::string kernel_name =
        ir_emitter_context_->GetSanitizedUniqueName(call.name);

    ASSIGN_OR_RETURN(TritonKernelSource triton_source,
                     EmitTritonFrom(call, kernel_name, mlir_context));

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
    block_level_parameters.global_scratch_memory_size =
        call.global_scratch_memory_size;
    block_level_parameters.is_tma_allowed = call.is_tma_allowed;

    ASSIGN_OR_RETURN(
        TritonWrapperResult result,
        CompileTritonToLLVM(
            kernel_name, *hlo_module, ir_emitter_context_->gpu_device_info(),
            block_level_parameters, ir_emitter_context_->target_triple(),
            ir_emitter_context_->data_layout(), std::move(triton_source),
            *ir_emitter_context_->llvm_context(), mlir_context,
            /*is_xla_fusion=*/false, emit_kernels));

    ASSIGN_OR_RETURN(auto kernel_arguments,
                     emitters::KernelArguments::Create(
                         ir_emitter_context_->buffer_assignment(),
                         GetDefaultBufferAlignment(), instr));
    auto launch_dimensions = LaunchDimensions(
        se::BlockDim(call.grid_x, call.grid_y, call.grid_z),
        se::ThreadDim(
            call.num_warps *
            ir_emitter_context_->gpu_device_info().threads_per_warp()));

    if (emit_kernels) {
      ASSIGN_OR_RETURN(llvm::Function * kernel,
                       RemoveUnusedTritonAbiArguments(
                           result.llvm_module.get(), *ir_emitter_context_,
                           kernel_name, call.global_scratch_memory_size > 0));

      AnnotateAttrsIfUnset(kernel_arguments, *kernel);
      TF_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
          ir_emitter_context_->gpu_device_info(), launch_dimensions, kernel,
          result.llvm_module.get()));
    }

    kernel_modules_.push_back(std::move(result.llvm_module));
    return {{kernel_name, launch_dimensions, /*cluster_dim=*/std::nullopt,
             result.shmem_bytes, /*binary=*/"", /*tma_metadata=*/{},
             result.use_pdl}};
  };

  auto [status_or_entry, was_cached] =
      ir_emitter_context_->kernel_cache().GetWithStatus(
          instr->raw_backend_config_string(), generate);
  ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry,
                   status_or_entry.Await());

  ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  LoadMlirDialectsForTriton(mlir_context);
  auto call =
      TritonCall::Parse(instr->raw_backend_config_string(), &mlir_context);

  return ThunkSequence::Of(std::make_unique<KernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      entry->kernel_name, kernel_arguments, entry->launch_dimensions,
      /*cluster_dim=*/std::nullopt, entry->shmem_bytes, entry->tma_metadata,
      /*zeroed_output_buffer_indices=*/call.zeroed_outputs, entry->use_pdl));
}

AsyncThunkSequence ThunkEmitter::EmitAsyncComputation(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  TF_RET_CHECK(wrapped->called_computations().size() == 1);
  HloComputation* computation = wrapped->called_computations().front();

  auto* async_start = Cast<HloAsyncInstruction>(instr);
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                   stream_assignment.GetExecutionStreamId(async_start));

  AsyncThunkSequence nested_thunks = EmitHloComputation(computation);

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  std::shared_ptr<AsyncExecution> async_execution =
      std::make_shared<AsyncExecution>(thunk_info);
  auto [it, inserted] = hlo_async_executions_.emplace(wrapped, async_execution);
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    wrapped->ToString());
  }

  return std::move(nested_thunks)
      .Map([thunk_info = std::move(thunk_info),
            async_execution = std::move(async_execution),
            execution_stream_id](ThunkSequence nested_thunks) {
        return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
            std::move(thunk_info), execution_stream_id,
            std::move(nested_thunks), async_execution));
      });
}

AsyncThunkSequence ThunkEmitter::EmitFusion(const HloFusionInstruction* instr) {
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
  ASSIGN_OR_RETURN(auto result, emitter->Emit(*ir_emitter_context_, *instr));

  // Use override flag because libdevice functions can be present in both.
  if (result.module) {
    kernel_modules_.push_back(std::move(result.module));
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

AsyncThunkSequence ThunkEmitter::EmitAsyncCustomCallStart(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  AsyncThunkSequence custom_call_thunks = EmitCustomCallSwitch(wrapped);

  auto* async_start = Cast<HloAsyncInstruction>(instr);
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                   stream_assignment.GetExecutionStreamId(async_start));

  Thunk::ThunkInfo start_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  std::shared_ptr<AsyncExecution> async_execution =
      std::make_shared<AsyncExecution>(start_thunk_info);
  auto [it, inserted] = hlo_async_executions_.emplace(wrapped, async_execution);
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    wrapped->ToString());
  }

  return std::move(custom_call_thunks)
      .Map([start_thunk_info = std::move(start_thunk_info),
            async_execution = std::move(async_execution),
            execution_stream_id](ThunkSequence custom_call_thunks) {
        return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
            std::move(start_thunk_info), execution_stream_id,
            std::move(custom_call_thunks), std::move(async_execution)));
      });
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

AsyncThunkSequence ThunkEmitter::EmitWhile(const HloInstruction* instr) {
  ASSIGN_OR_RETURN(auto config,
                   instr->backend_config<xla::WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count = std::nullopt;
  if (config.has_known_trip_count()) {
    trip_count = config.known_trip_count().n();
  }

  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Buffer slice holding while loop predicate.
  ASSIGN_OR_RETURN(BufferAllocation::Slice pred,
                   GetAllocationSliceForHlo(condition->root_instruction(), {}));
  Thunk::ThunkInfo info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  return std::move(tsl::JoinFutures(EmitHloComputation(condition),
                                    EmitHloComputation(body)))
      .Map([info = std::move(info), pred = pred, trip_count = trip_count](
               std::tuple<ThunkSequence, ThunkSequence> tuple) {
        auto [cond_thunks, body_thunks] = std::move(tuple);
        return GetThunkSequence(std::make_unique<WhileThunk>(
            std::move(info), std::move(pred), std::move(cond_thunks),
            std::move(body_thunks), trip_count));
      });
}

AsyncThunkSequence ThunkEmitter::EmitCallComputation(
    const HloInstruction* instr) {
  DCHECK_EQ(instr->called_computations().size(), 1);
  const HloComputation* computation = instr->called_computations().front();
  return EmitHloComputation(computation);
}

AsyncThunkSequence ThunkEmitter::EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr) {
  ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context_->buffer_assignment(),
                       GetDefaultBufferAlignment(), instr));

  ASSIGN_OR_RETURN(KernelDefinition<LlvmKernelSource> kernel_def,
                   EmitRngGetAndUpdateStateLLVMIR(instr, ir_emitter_context_,
                                                  kernel_arguments));

  KernelSpec spec = kernel_def.spec();
  ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      LaunchDimensions::FromWorkDimensions(spec.work_dimensions()));

  return ir_emitter_context_->kernel_compiler()
      ->Compile(Thunk::ThunkInfo::WithProfileAnnotation(
                    instr, ir_emitter_context_->GetNextThunkId()),
                std::move(kernel_def).TakeSource(), std::string(spec.name()),
                kernel_arguments, launch_dimensions)
      .Map([](std::unique_ptr<Thunk> thunk) {
        return ThunkSequence::Of(std::move(thunk));
      });
}

AsyncThunkSequence ThunkEmitter::EmitSort(const HloSortInstruction* sort) {
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
    ASSIGN_OR_RETURN(destination_buffer,
                     GetAllocationSliceForHlo(sort, shape_index));
    ASSIGN_OR_RETURN(source_address,
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

  return EmitBitonicSortLLVMIR(sort, ir_emitter_context_)
      .Map([thunks = std::move(thunks)](ThunkSequence sort_thunks) mutable {
        AppendThunkSequence(thunks, sort_thunks);
        return std::move(thunks);
      });
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

[[deprecated("Use NCCL 2.28+ primitives instead.")]]
bool IsNvshmemCollective(const HloInstruction* instr) {
  if (instr->has_backend_config()) {
    auto gpu_config = instr->backend_config<GpuBackendConfig>();
    const CollectiveBackendConfig& backend_config =
        gpu_config.value().collective_backend_config();
    return backend_config.backend() == CollectiveBackendConfig::NVSHMEM;
  }
  return false;
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
    if (CollectivePermuteThunk::IsDegenerate(instr, replica_count,
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
  if (!CollectivePermuteThunk::IsDegenerate(instr, replica_count,
                                            partition_count)) {
    if (IsNvshmemCollective(instr)) {
      // Note: xla_gpu_use_memcpy_local_p2p flag won't be used for now since the
      // NVSHMEM collective permute thunk doesn't perform any memcpy operations
      // at the moment.
      thunks.push_back(std::make_unique<NvshmemCollectivePermuteThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          instr, replica_count, partition_count, buffers,
          ir_emitter_context_->debug_options().xla_gpu_use_memcpy_local_p2p()));
    } else {
      thunks.push_back(std::make_unique<CollectivePermuteThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          instr, replica_count, partition_count, buffers,
          ir_emitter_context_->debug_options()
              .xla_gpu_collective_permute_mode(),
          ir_emitter_context_->debug_options()
              .xla_gpu_collective_permute_connected_components()));
    }
  }

  // For synchronous collectives, emit thunks directly without async wrapping.
  // However, if parallel collective overlap limit is > 1, multiple collectives
  // may be in-flight on different streams. Emitting them synchronously would be
  // unsafe as they could share communicators across streams. Force async
  // emission in that case.
  if (IsGPUSyncCollective(*instr) &&
      ir_emitter_context_->debug_options()
              .xla_gpu_experimental_parallel_collective_overlap_limit() <= 1) {
    hlo_async_executions_.try_emplace(instr, nullptr);
    return thunks;
  }

  // Wrap in AsyncStartThunk for asynchronous execution.
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                      stream_assignment.GetExecutionStreamId(instr));

  auto start_thunk = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          instr, ir_emitter_context_->GetNextThunkId()),
      execution_stream_id, std::move(thunks));

  auto [it, inserted] =
      hlo_async_executions_.emplace(instr, start_thunk->async_execution());
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    instr->ToString());
  }

  return GetThunkSequence(std::move(start_thunk));
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

  if (kind == Thunk::Kind::kAllGather) {
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
  // For synchronous collectives, emit thunk directly without async wrapping.
  // However, if parallel collective overlap limit is > 1, multiple collectives
  // may be in-flight on different streams. Emitting them synchronously would be
  // unsafe as they could share communicators across streams. Force async
  // emission in that case.
  if (IsGPUSyncCollective(*async_start) &&
      ir_emitter_context_->debug_options()
              .xla_gpu_experimental_parallel_collective_overlap_limit() <= 1) {
    hlo_async_executions_.try_emplace(async_start, nullptr);
    return ThunkSequence::Of(std::move(thunk));
  }

  // Wrap collective thunk in AsyncStartThunk for asynchronous execution.
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                      stream_assignment.GetExecutionStreamId(async_start));

  auto start_thunk = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          async_start, ir_emitter_context_->GetNextThunkId()),
      execution_stream_id, ThunkSequence::Of(std::move(thunk)));

  auto [it, inserted] = hlo_async_executions_.emplace(
      async_start, start_thunk->async_execution());
  if (!inserted) {
    return Internal("Async execution already exists for instruction %s",
                    async_start->ToString());
  }

  return GetThunkSequence(std::move(start_thunk));
}

AsyncThunkSequence ThunkEmitter::EmitCollectiveGroupStartThunk(
    const HloInstruction* instr) {
  std::vector<AsyncThunkSequence> futures;
  for (const HloInstruction* nested_instruction :
       instr->async_wrapped_computation()->instructions()) {
    futures.push_back(
        EmitHloInstruction(nested_instruction, /*emit_group_thunks=*/true));
  }

  Thunk::ThunkInfo group_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());
  Thunk::ThunkInfo start_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context_->GetNextThunkId());

  bool is_sync = IsGPUSyncCollective(*instr);

  std::shared_ptr<AsyncExecution> async_execution;
  std::optional<ExecutionStreamId> execution_stream_id;
  if (is_sync) {
    hlo_async_executions_.try_emplace(instr, nullptr);
  } else {
    async_execution = std::make_shared<AsyncExecution>(start_thunk_info);

    auto [it, inserted] = hlo_async_executions_.emplace(instr, async_execution);
    if (!inserted) {
      return Internal("Async execution already exists for instruction %s",
                      instr->ToString());
    }

    const ExecutionStreamAssignment& stream_assignment =
        ir_emitter_context_->execution_stream_assignment();
    ASSIGN_OR_RETURN(execution_stream_id,
                     stream_assignment.GetExecutionStreamId(instr));
  }

  return tsl::JoinFutures(absl::MakeSpan(futures))
      .Map([group_thunk_info = std::move(group_thunk_info),
            start_thunk_info = std::move(start_thunk_info),
            async_execution = std::move(async_execution), execution_stream_id,
            is_sync](std::vector<ThunkSequence> sequences) {
        ThunkSequence thunks = FlattenThunkSequence(std::move(sequences));
        auto group_thunk = std::make_unique<CollectiveGroupThunk>(
            std::move(group_thunk_info), Thunk::Kind::kGroup,
            std::move(thunks));

        // For synchronous collectives, emit group thunk directly without async
        // wrapping.
        if (is_sync) {
          return ThunkSequence::Of(std::move(group_thunk));
        }

        return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
            std::move(start_thunk_info), *execution_stream_id,
            ThunkSequence::Of(std::move(group_thunk)),
            std::move(async_execution)));
      });
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectiveAsyncDone(
    const HloInstruction* inst) {
  // Determine if this is a send/recv done operation.
  bool is_send_recv =
      inst->opcode() == HloOpcode::kSendDone ||
      inst->opcode() == HloOpcode::kRecvDone ||
      (inst->opcode() == HloOpcode::kAsyncDone &&
       (inst->async_wrapped_instruction()->opcode() == HloOpcode::kSend ||
        inst->async_wrapped_instruction()->opcode() == HloOpcode::kRecv));
  const HloInstruction* start =
      is_send_recv ? FindCanonicalSendRecvStartOp(inst) : inst->operand(0);

  // Find the async execution for the start operation.
  auto it = hlo_async_executions_.find(start);
  TF_RET_CHECK(it != hlo_async_executions_.end())
      << "couldn't find async execution for start operation";

  // Can be null if no start thunk was created (e.g. if the start op
  // is degenerate), in which case there's nothing to do here.
  if (!it->second) {
    return ThunkSequence{};
  }

  return GetThunkSequence(std::make_unique<AsyncDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          inst, ir_emitter_context_->GetNextThunkId()),
      it->second));
}

[[deprecated("Use NCCL 2.28+ primitives instead.")]]
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNvshmemAsyncDone(
    const HloInstruction* inst) {
  bool is_send_recv =
      inst->opcode() == HloOpcode::kSendDone ||
      inst->opcode() == HloOpcode::kRecvDone ||
      (inst->opcode() == HloOpcode::kAsyncDone &&
       (inst->async_wrapped_instruction()->opcode() == HloOpcode::kSend ||
        inst->async_wrapped_instruction()->opcode() == HloOpcode::kRecv));
  const HloInstruction* start =
      is_send_recv ? FindCanonicalSendRecvStartOp(inst) : inst->operand(0);

  // Find the async execution for the start operation.
  auto it = hlo_async_executions_.find(start);
  TF_RET_CHECK(it != hlo_async_executions_.end())
      << "couldn't find async execution for start operation";

  // Can be null if no start thunk was created (e.g. if the start op is
  // degenerate), in which case there's nothing to do here.
  if (!it->second) {
    return ThunkSequence{};
  }

  return GetThunkSequence(std::make_unique<AsyncDoneThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          inst, ir_emitter_context_->GetNextThunkId()),
      it->second));
}

template <typename NvshmemAllReduceThunkType, typename HloAllReduceInstruction>
[[deprecated("Use NCCL 2.28+ primitives instead.")]]
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitNvshmemThunk(
    Thunk::Kind kind, const HloInstruction* async_start,
    const HloAllReduceInstruction* inst,
    std::optional<bool> use_global_device_ids) {
  CHECK(kind == Thunk::Kind::kNvshmemAllReduce);
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

    // For synchronous collectives, emit thunk directly without async wrapping.
    if (IsGPUSyncCollective(*async_start)) {
      hlo_async_executions_.try_emplace(async_start, nullptr);
      return ThunkSequence::Of(std::move(thunk));
    }

    // Wrap in AsyncStartThunk for asynchronous execution.
    const ExecutionStreamAssignment& stream_assignment =
        ir_emitter_context_->execution_stream_assignment();
    TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                        stream_assignment.GetExecutionStreamId(async_start));

    auto start_thunk = std::make_unique<AsyncStartThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            async_start, ir_emitter_context_->GetNextThunkId()),
        execution_stream_id, ThunkSequence::Of(std::move(thunk)));

    auto [it, inserted] = hlo_async_executions_.emplace(
        async_start, start_thunk->async_execution());
    if (!inserted) {
      return Internal("Async execution already exists for instruction %s",
                      async_start->ToString());
    }

    return GetThunkSequence(std::move(start_thunk));
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
  // Signal that start thunk not created (degenerate) with nullptr.
  hlo_async_executions_.try_emplace(async_start, nullptr);

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
  auto host_memory_space =
      static_cast<int>(stream_executor::MemorySpace::kHost);
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

  // Create the copy thunk with ThunkInfo derived from copy-start.
  Thunk::ThunkInfo copy_thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      copy_start_instr, ir_emitter_context_->GetNextThunkId());

  std::unique_ptr<CopyThunk> copy_thunk;
  if (is_dst_host_memory) {
    copy_thunk = std::make_unique<DeviceToHostCopyThunk>(
        copy_thunk_info,
        /*source_buffer=*/ShapedSlice{src_buffer, input_shape},
        /*destination_buffer=*/ShapedSlice{dst_buffer, input_shape},
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape));
  } else {
    copy_thunk = std::make_unique<HostToDeviceCopyThunk>(
        copy_thunk_info,
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
    return GetThunkSequence(std::move(copy_thunk));
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

  return GetThunkSequence(std::move(start_thunk));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyDoneThunk(
    const HloInstruction* instr) {
  const HloInstruction* copy_start_instr = instr->operand(0);
  CHECK(copy_start_instr->opcode() == HloOpcode::kCopyStart);

  // If the copy-start was asynchronous, emit an AsyncDoneThunk.
  auto it = hlo_async_executions_.find(copy_start_instr);
  if (it != hlo_async_executions_.end()) {
    return GetThunkSequence(std::make_unique<AsyncDoneThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(
            instr, ir_emitter_context_->GetNextThunkId()),
        it->second));
  }

  // Synchronous copy: copy-done is a no-op.
  return ThunkSequence();
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
    // Wrap in AsyncStartThunk if not emitted as part of a group thunk.
    if (!emit_group_thunks) {
      const HloInstruction* canonical_send_instr =
          FindCanonicalSendRecvStartOp(instr);

      const ExecutionStreamAssignment& stream_assignment =
          ir_emitter_context_->execution_stream_assignment();
      TF_ASSIGN_OR_RETURN(
          ExecutionStreamId execution_stream_id,
          stream_assignment.GetExecutionStreamId(canonical_send_instr));

      // Check if an async execution already exists for this canonical
      // send/recv pair (pipelined send/recv share the same async stream).
      auto existing_it = hlo_async_executions_.find(canonical_send_instr);
      if (existing_it != hlo_async_executions_.end()) {
        auto start_thunk = std::make_unique<AsyncStartThunk>(
            Thunk::ThunkInfo::WithProfileAnnotation(
                instr, ir_emitter_context_->GetNextThunkId()),
            execution_stream_id, ThunkSequence::Of(std::move(thunk)),
            existing_it->second);
        return GetThunkSequence(std::move(start_thunk));
      }

      auto start_thunk = std::make_unique<AsyncStartThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          execution_stream_id, ThunkSequence::Of(std::move(thunk)));

      hlo_async_executions_.try_emplace(canonical_send_instr,
                                        start_thunk->async_execution());
      return GetThunkSequence(std::move(start_thunk));
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
      return EmitNvshmemAsyncDone(instr);
    }
    return EmitCollectiveAsyncDone(instr);
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
    // Wrap in AsyncStartThunk if not emitted as part of a group thunk.
    if (!emit_group_thunks) {
      const HloInstruction* canonical_recv_instr =
          FindCanonicalSendRecvStartOp(instr);

      const ExecutionStreamAssignment& stream_assignment =
          ir_emitter_context_->execution_stream_assignment();
      TF_ASSIGN_OR_RETURN(
          ExecutionStreamId execution_stream_id,
          stream_assignment.GetExecutionStreamId(canonical_recv_instr));

      // Check if an async execution already exists for this canonical
      // send/recv pair (pipelined send/recv share the same async stream).
      auto existing_it = hlo_async_executions_.find(canonical_recv_instr);
      if (existing_it != hlo_async_executions_.end()) {
        auto start_thunk = std::make_unique<AsyncStartThunk>(
            Thunk::ThunkInfo::WithProfileAnnotation(
                instr, ir_emitter_context_->GetNextThunkId()),
            execution_stream_id, ThunkSequence::Of(std::move(thunk)),
            existing_it->second);
        return GetThunkSequence(std::move(start_thunk));
      }

      auto start_thunk = std::make_unique<AsyncStartThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          execution_stream_id, ThunkSequence::Of(std::move(thunk)));

      hlo_async_executions_.try_emplace(canonical_recv_instr,
                                        start_thunk->async_execution());
      return GetThunkSequence(std::move(start_thunk));
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
      return EmitNvshmemAsyncDone(instr);
    }
    return EmitCollectiveAsyncDone(instr);
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
    return EmitCollectiveAsyncDone(instr);
  }
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  ThunkSequence thunks;
  switch (wrapped->opcode()) {
    case HloOpcode::kReduceScatter:
      return EmitCollectiveAsyncDone(instr);
    case HloOpcode::kAllToAll:
      return EmitCollectiveAsyncDone(instr);
    case HloOpcode::kRaggedAllToAll:
      return EmitCollectiveAsyncDone(instr);
    case HloOpcode::kCollectiveBroadcast:
      return EmitCollectiveAsyncDone(instr);
    case HloOpcode::kCollectivePermute:
      return EmitCollectiveAsyncDone(instr);
    case HloOpcode::kRecv:
      return EmitCollectiveAsyncDone(instr);
    case HloOpcode::kSend:
      return EmitCollectiveAsyncDone(instr);
    case HloOpcode::kFusion:
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
      auto it = hlo_async_executions_.find(wrapped);
      if (it == hlo_async_executions_.end()) {
        return Internal(
            "Async execution not found for instruction %s. "
            "EmitAsyncComputation must be called before EmitAsyncDone.",
            wrapped->ToString());
      }
      thunks.push_back(std::make_unique<AsyncDoneThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId()),
          it->second));
      return thunks;
    }
    default:
      return Internal("Unsupported async done wrapped instruction: %s",
                      HloOpcodeString(wrapped->opcode()));
  }
}

AsyncThunkSequence ThunkEmitter::EmitAsyncStart(const HloInstruction* instr) {
  // Multi-op async start will emit a NCCL group thunk.
  if (!instr->async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
    return EmitCollectiveGroupStartThunk(instr);
  }
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  switch (wrapped->opcode()) {
    case HloOpcode::kReduceScatter: {
      auto* reduce_scatter = Cast<HloReduceScatterInstruction>(wrapped);
      return EmitCollectiveThunk<ReduceScatterThunk,
                                 HloReduceScatterInstruction>(
          Thunk::kReduceScatter, instr, reduce_scatter,
          reduce_scatter->use_global_device_ids());
    }
    case HloOpcode::kAllToAll: {
      auto* all_to_all = Cast<HloAllToAllInstruction>(wrapped);
      return EmitCollectiveThunk<AllToAllThunk, HloAllToAllInstruction>(
          Thunk::kAllToAll, instr, all_to_all, std::nullopt);
    }
    case HloOpcode::kRaggedAllToAll: {
      auto* ragged_all_to_all = Cast<HloRaggedAllToAllInstruction>(wrapped);
      return EmitCollectiveThunk<RaggedAllToAllThunk,
                                 HloRaggedAllToAllInstruction>(
          Thunk::kRaggedAllToAll, instr, ragged_all_to_all, std::nullopt);
    }
    case HloOpcode::kCollectiveBroadcast: {
      auto* collective_broadcast =
          Cast<HloCollectiveBroadcastInstruction>(wrapped);
      return EmitCollectiveThunk<CollectiveBroadcastThunk,
                                 HloCollectiveBroadcastInstruction>(
          Thunk::kCollectiveBroadcast, instr, collective_broadcast,
          std::nullopt);
    }
    case HloOpcode::kFusion: {
      AsyncThunkSequence fusion_thunks =
          EmitFusion(Cast<HloFusionInstruction>(wrapped));

      Thunk::ThunkInfo start_thunk_info =
          Thunk::ThunkInfo::WithProfileAnnotation(
              instr, ir_emitter_context_->GetNextThunkId());
      std::shared_ptr<AsyncExecution> async_execution =
          std::make_shared<AsyncExecution>(start_thunk_info);
      auto [it, inserted] =
          hlo_async_executions_.emplace(wrapped, async_execution);
      if (!inserted) {
        return Internal("Async execution already exists for instruction %s",
                        wrapped->ToString());
      }

      auto* async_start = Cast<HloAsyncInstruction>(instr);
      const ExecutionStreamAssignment& stream_assignment =
          ir_emitter_context_->execution_stream_assignment();
      ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                       stream_assignment.GetExecutionStreamId(async_start));

      return std::move(fusion_thunks)
          .Map([start_thunk_info = std::move(start_thunk_info),
                async_execution = std::move(async_execution),
                execution_stream_id](ThunkSequence fusion_thunks) {
            return ThunkSequence::Of(std::make_unique<AsyncStartThunk>(
                std::move(start_thunk_info), execution_stream_id,
                std::move(fusion_thunks), std::move(async_execution)));
          });
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

AsyncThunkSequence ThunkEmitter::EmitCustomCallSwitch(
    const HloInstruction* hlo) {
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
  if (IsCublasLtGroupedMatmul(*hlo)) {
    return EmitCublasLtGroupedMatmulThunk(custom_call);
  }
  if (IsCublasLtMatmulMx(*hlo)) {
    return EmitCublasLtMatmulThunkMx(custom_call);
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
    return ThunkSequence{};
  }
  if (hlo->custom_call_target() == kPinCustomCallTarget ||
      hlo->custom_call_target() == kUnpinCustomCallTarget ||
      hlo->custom_call_target() == kCreateBufferCustomCallTarget) {
    return ThunkSequence{};
  }
  return EmitCustomCallThunk(custom_call);
}

AsyncThunkSequence ThunkEmitter::EmitHloInstruction(const HloInstruction* hlo,
                                                    bool emit_group_thunks) {
  switch (hlo->opcode()) {
    case HloOpcode::kAllGatherDone:
      return EmitCollectiveAsyncDone(hlo);
    case HloOpcode::kAllGatherStart: {
      auto* all_gather = Cast<HloAllGatherInstruction>(hlo);
      return EmitCollectiveThunk<AllGatherThunk, HloAllGatherInstruction>(
          Thunk::kAllGather, all_gather, all_gather,
          all_gather->use_global_device_ids());
    }
    case HloOpcode::kAllReduceDone:
      return IsNvshmemCollective(hlo) ? EmitNvshmemAsyncDone(hlo)
                                      : EmitCollectiveAsyncDone(hlo);
    case HloOpcode::kAllReduceStart: {
      auto* all_reduce = Cast<HloAllReduceInstruction>(hlo);
      if (IsNvshmemCollective(hlo)) {
        return EmitNvshmemThunk<NvshmemAllReduceThunk, HloAllReduceInstruction>(
            Thunk::kNvshmemAllReduce, all_reduce, all_reduce,
            all_reduce->use_global_device_ids());
      }
      return EmitCollectiveThunk<AllReduceThunk, HloAllReduceInstruction>(
          Thunk::kAllReduce, all_reduce, all_reduce,
          all_reduce->use_global_device_ids());
    }
    case HloOpcode::kAsyncDone:
      return EmitAsyncDone(hlo);
    case HloOpcode::kAsyncStart:
      return EmitAsyncStart(hlo);
    case HloOpcode::kCall:
      return EmitCallComputation(hlo);
    case HloOpcode::kCollectivePermuteDone:
      return IsNvshmemCollective(hlo) ? EmitNvshmemAsyncDone(hlo)
                                      : EmitCollectiveAsyncDone(hlo);
    case HloOpcode::kCollectivePermuteStart:
      return EmitCollectivePermute(Cast<HloCollectivePermuteInstruction>(hlo));
    case HloOpcode::kConditional:
      return EmitConditional(hlo);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(hlo));
    case HloOpcode::kCustomCall:
      return EmitCustomCallSwitch(hlo);
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
  ASSIGN_OR_RETURN(
      ThunkSequence thunks,
      std::move(EmitHloComputation(module->entry_computation())).Await());
  return std::make_unique<SequentialThunk>(Thunk::ThunkInfo{},
                                           std::move(thunks));
}

AsyncThunkSequence ThunkEmitter::EmitHloComputation(
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
  std::vector<AsyncThunkSequence> futures(instructions.size());
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

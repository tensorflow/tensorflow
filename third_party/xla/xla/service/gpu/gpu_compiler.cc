/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_compiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/collectives/all_gather_broadcast_reorder.h"
#include "xla/hlo/transforms/collectives/all_reduce_contiguous.h"
#include "xla/hlo/transforms/collectives/collective_quantizer.h"
#include "xla/hlo/transforms/collectives/collectives_schedule_linearizer.h"
#include "xla/hlo/transforms/convert_memory_placement_to_internal_annotations.h"
#include "xla/hlo/transforms/expanders/bitcast_dtypes_expander.h"
#include "xla/hlo/transforms/expanders/comparison_expander.h"
#include "xla/hlo/transforms/expanders/convolution_4d_expander.h"
#include "xla/hlo/transforms/expanders/convolution_pred_expander.h"
#include "xla/hlo/transforms/expanders/dot_decomposer.h"
#include "xla/hlo/transforms/expanders/dynamic_index_splitter.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/logistic_expander.h"
#include "xla/hlo/transforms/expanders/optimization_barrier_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/hlo/transforms/expanders/real_imag_expander.h"
#include "xla/hlo/transforms/expanders/reduce_decomposer.h"
#include "xla/hlo/transforms/expanders/reshape_decomposer.h"
#include "xla/hlo/transforms/expanders/rng_bit_generator_expander.h"
#include "xla/hlo/transforms/expanders/rng_expander.h"
#include "xla/hlo/transforms/expanders/stable_sort_expander.h"
#include "xla/hlo/transforms/expanders/stochastic_convert_decomposer.h"
#include "xla/hlo/transforms/host_offload_legalize.h"
#include "xla/hlo/transforms/host_offloader.h"
#include "xla/hlo/transforms/operand_upcaster.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/all_reduce_folder.h"
#include "xla/hlo/transforms/simplifiers/broadcast_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/conditional_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/dot_merger.h"
#include "xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_computation_deduplicator.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization.h"
#include "xla/hlo/transforms/simplifiers/host_memory_transfer_asyncifier.h"
#include "xla/hlo/transforms/simplifiers/optimize_input_output_buffer_alias.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_rewriter.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/result_caster.h"
#include "xla/hlo/transforms/simplifiers/simplify_fp_conversions.h"
#include "xla/hlo/transforms/simplifiers/slice_sinker.h"
#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"
#include "xla/hlo/transforms/simplifiers/sub_byte_normalization.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/maybe_owning.h"
#include "xla/service/all_reduce_promotion.h"
#include "xla/service/all_reduce_reassociate.h"
#include "xla/service/all_reduce_simplifier.h"
#include "xla/service/batched_gather_scatter_normalizer.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_inliner.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_permute_decomposer.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/collective_utils.h"
#include "xla/service/compiler.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/dump.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/executable.h"
#include "xla/service/export_hlo.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/custom_kernel_fusion_autotuner.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/conv_layout_normalization.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/flag_utils.h"
#include "xla/service/gpu/fusion_dispatch_pipeline.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/gpu_p2p_pipeliner.h"
#include "xla/service/gpu/gpu_spmd_pipeline.h"
#include "xla/service/gpu/hlo_fusion_stats.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_unnested.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/model/gpu_cost_model_stats_collection.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/sol_gpu_cost_model_stats_collection.h"
#include "xla/service/gpu/pre_scheduling_copy_insertion_pipeline.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/gpu/runtime_intrinsics.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/algebraic_simplifier.h"
#include "xla/service/gpu/transforms/algorithm_checker.h"
#include "xla/service/gpu/transforms/all_gather_dynamic_slice_simplifier.h"
#include "xla/service/gpu/transforms/all_gather_optimizer.h"
#include "xla/service/gpu/transforms/all_reduce_blueconnect.h"
#include "xla/service/gpu/transforms/all_reduce_splitter.h"
#include "xla/service/gpu/transforms/async_wrapper.h"
#include "xla/service/gpu/transforms/collective_permute_cycle_decomposer.h"
#include "xla/service/gpu/transforms/collective_permute_valid_iteration_annotator.h"
#include "xla/service/gpu/transforms/collective_select_folder.h"
#include "xla/service/gpu/transforms/collectives/all_gather_combiner.h"
#include "xla/service/gpu/transforms/collectives/all_reduce_combiner.h"
#include "xla/service/gpu/transforms/collectives/convert_async_collectives_to_sync.h"
#include "xla/service/gpu/transforms/collectives/gpu_collective_combiner_utils.h"
#include "xla/service/gpu/transforms/collectives/reduce_scatter_combiner.h"
#include "xla/service/gpu/transforms/command_buffer_scheduling.h"
#include "xla/service/gpu/transforms/conv_rewriter.h"
#include "xla/service/gpu/transforms/cudnn_custom_call_converter.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/dot_dimension_sorter.h"
#include "xla/service/gpu/transforms/dot_normalizer.h"
#include "xla/service/gpu/transforms/dot_operand_converter.h"
#include "xla/service/gpu/transforms/double_buffer_loop_unrolling.h"
#include "xla/service/gpu/transforms/dynamic_slice_fusion_rewriter.h"
#include "xla/service/gpu/transforms/explicit_stream_annotation_async_wrapper.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/gemm_broadcast_folding_rewriter.h"
#include "xla/service/gpu/transforms/gemm_fusion.h"
#include "xla/service/gpu/transforms/gemm_fusion_swap_operands.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/gemv_rewriter.h"
#include "xla/service/gpu/transforms/layout_assignment.h"
#include "xla/service/gpu/transforms/move_copy_to_users.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/service/gpu/transforms/pipelined_p2p_rewriter.h"
#include "xla/service/gpu/transforms/ragged_all_to_all_canonicalizer.h"
#include "xla/service/gpu/transforms/ragged_all_to_all_decomposer.h"
#include "xla/service/gpu/transforms/reduce_scatter_creator.h"
#include "xla/service/gpu/transforms/reduction_degenerate_dim_remover.h"
#include "xla/service/gpu/transforms/reduction_dimension_grouper.h"
#include "xla/service/gpu/transforms/reduction_layout_normalizer.h"
#include "xla/service/gpu/transforms/reduction_splitter.h"
#include "xla/service/gpu/transforms/rename_fusions.h"
#include "xla/service/gpu/transforms/sanitize_constant_names.h"
#include "xla/service/gpu/transforms/scatter_expander.h"
#include "xla/service/gpu/transforms/scatter_slice_simplifier.h"
#include "xla/service/gpu/transforms/softmax_rewriter_triton.h"
#include "xla/service/gpu/transforms/sort_rewriter.h"
#include "xla/service/gpu/transforms/stream_attribute_annotator.h"
#include "xla/service/gpu/transforms/stream_attribute_async_wrapper.h"
#include "xla/service/gpu/transforms/topk_specializer.h"
#include "xla/service/gpu/transforms/topk_splitter.h"
#include "xla/service/gpu/transforms/transpose_dimension_grouper.h"
#include "xla/service/gpu/transforms/tree_reduction_rewriter.h"
#include "xla/service/gpu/transforms/triton_fusion_numerics_verifier.h"
#include "xla/service/gpu/transforms/windowed_einsum_handler.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/reduce_scatter_reassociate.h"
#include "xla/service/scatter_determinism_expander.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/select_and_scatter_expander.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/service/spmd/schedule_aware_collective_ops_cse.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/while_loop_all_reduce_code_motion.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

#ifdef PLATFORM_GOOGLE
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#endif  // PLATFORM_GOOGLE

namespace xla {
namespace gpu {
namespace {

using MaybeOwningThreadPool = MaybeOwning<tsl::thread::ThreadPool>;

MaybeOwningThreadPool CreateMaybeOwningThreadPool(
    int parallelism, tsl::thread::ThreadPool* default_thread_pool,
    int default_parallelism) {
  CHECK_GE(parallelism, 0);
  CHECK_GE(default_parallelism, 1);
  // CurrentThreadId() returns -1 if the current thread does not belong to the
  // thread pool. If the current thread belongs to the thread pool, we should
  // not be using it, because it can potentially cause deadlocks.
  CHECK(default_thread_pool == nullptr ||
        default_thread_pool->CurrentThreadId() == -1);

  auto create_thread_pool = [&](int num_threads) {
    CHECK_GE(num_threads, 1);
    return std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(), "",
                                                     num_threads);
  };

  switch (parallelism) {
    case 0:
      if (default_thread_pool == nullptr && default_parallelism > 1) {
        return MaybeOwningThreadPool(create_thread_pool(default_parallelism));
      }
      return MaybeOwningThreadPool(default_thread_pool);
    case 1:
      return MaybeOwningThreadPool(nullptr);
    default:
      return MaybeOwningThreadPool(create_thread_pool(parallelism));
  }
}

absl::StatusOr<AutotuneConfig> GetAutotuneConfig(
    se::StreamExecutor* stream_exec, const DebugOptions& debug_options,
    const GpuCompiler::CompileOptions& options,
    const Compiler::TargetConfig& gpu_target_config) {
  if (stream_exec) {
    return AutotuneConfig{DeviceConfig{stream_exec, options.device_allocator},
                          debug_options};
  }
  return AutotuneConfig{DevicelessConfig{gpu_target_config.device_description},
                        debug_options};
}

se::GpuComputeCapability GetGpuVersion(const se::StreamExecutor* stream_exec) {
  return stream_exec->GetDeviceDescription().gpu_compute_capability();
}

class GpuThunkAotCompilationResult : public AotCompilationResult {
 public:
  static absl::StatusOr<std::unique_ptr<GpuThunkAotCompilationResult>>
  FromModule(const HloModule* hlo_module,
             const BufferAssignment* buffer_assignment,
             absl::string_view asm_text, absl::Span<const uint8_t> binary,
             const BinaryMap& dnn_compiled_graphs) {
    CompilationResultProto proto;
    *proto.mutable_hlo_module_with_config() = hlo_module->ToProtoWithConfig();
    *proto.mutable_buffer_assignment() = buffer_assignment->ToProto();
    proto.set_asm_text(std::string(asm_text));
    proto.set_binary(binary.data(), binary.size());
    proto.mutable_dnn_compiled_graphs()->insert(dnn_compiled_graphs.cbegin(),
                                                dnn_compiled_graphs.cend());
    return std::unique_ptr<GpuThunkAotCompilationResult>(
        new GpuThunkAotCompilationResult(hlo_module->Clone(),
                                         std::move(proto)));
  }

  static absl::StatusOr<std::unique_ptr<GpuThunkAotCompilationResult>>
  FromString(const std::string& serialized) {
    CompilationResultProto proto;
    if (!proto.ParseFromString(serialized)) {
      return Internal(
          "Failed to parse serialized GpuThunkAotCompilationResult.");
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProtoWithConfig(proto.hlo_module_with_config()));
    return std::unique_ptr<GpuThunkAotCompilationResult>(
        new GpuThunkAotCompilationResult(std::move(module), std::move(proto)));
  }

  absl::StatusOr<std::string> SerializeAsString() const override {
    return proto_.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, const se::StreamExecutor* stream_exec) const override;

  const HloModule* optimized_module() const override { return module_.get(); }
  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }

 private:
  GpuThunkAotCompilationResult(std::unique_ptr<HloModule> module,
                               CompilationResultProto proto)
      : module_(std::move(module)), proto_(std::move(proto)) {}

  std::unique_ptr<HloModule> module_;
  CompilationResultProto proto_;
};

}  // end anonymous namespace

absl::StatusOr<std::unique_ptr<Executable>>
GpuThunkAotCompilationResult::LoadExecutable(
    Compiler* compiler, const se::StreamExecutor* stream_exec) const {
  // Recreate HloModule+HloModuleConfig from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProtoWithConfig(proto_.hlo_module_with_config()));

  // Recreate BufferAssignment from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssignment::FromProto(proto_.buffer_assignment(), hlo_module.get(),
                                  compiler->BufferSizeBytesFunction(),
                                  /*can_share_buffer=*/nullptr));

  ExecutionStreamAssignment execution_stream_assignment(hlo_module.get());

  std::vector<uint8_t> binary(proto_.binary().begin(), proto_.binary().end());

  // Build the executable, which should be a thunk sequence.
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithId(compiler->PlatformId()));
  std::string platform_name = platform->Name();
  const se::DeviceDescription& gpu_device_info =
      stream_exec->GetDeviceDescription();
  mlir::DialectRegistry registry;
  auto mlir_context = std::make_unique<mlir::MLIRContext>(registry);
  llvm::LLVMContext llvm_context;
  auto* gpu_compiler = dynamic_cast<GpuCompiler*>(compiler);
  if (gpu_compiler == nullptr) {
    return Internal("Compiler is not a GpuCompiler.");
  }
  auto llvm_module = std::make_unique<llvm::Module>("", llvm_context);
  llvm_module->setTargetTriple(gpu_compiler->target_triple());
  llvm_module->setDataLayout(gpu_compiler->data_layout());
  IrEmitterContext ir_emitter_context(
      hlo_module.get(), buffer_assignment.get(), &execution_stream_assignment,
      platform_name, gpu_device_info, mlir_context.get(), llvm_module.get(),
      /*llvm_module_constants=*/nullptr,
      /*emit_kernels=*/false);

  absl::string_view cache_file_path =
      hlo_module->config().debug_options().xla_gpu_kernel_cache_file();
  if (!cache_file_path.empty() &&
      hlo_module->config()
          .debug_options()
          .xla_gpu_enable_llvm_module_compilation_parallelism()) {
    TF_RETURN_IF_ERROR(LoadCache(ir_emitter_context, cache_file_path));
  }

  auto ir_emitter = IrEmitterUnnested::Create(&ir_emitter_context);
  TF_RETURN_IF_ERROR(
      ir_emitter->EmitHloComputation(hlo_module->entry_computation()));

  // Get all other fields required by GpuExecutable.
  std::vector<GpuExecutable::ConstantInfo> constants =
      std::move(ir_emitter_context.constants());
  TF_ASSIGN_OR_RETURN(auto output_info,
                      GetOutputInfo(*hlo_module, *buffer_assignment));
  const Shape& output_shape = hlo_module->result_shape();
  int64_t debug_buffer_assignment_show_max =
      hlo_module->config()
          .debug_options()
          .xla_debug_buffer_assignment_show_max();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::Create(GpuExecutable::Params{
          /*asm_text=*/proto_.asm_text(),
          /*binary=*/binary,
          /*dnn_compiled_graphs=*/
          BinaryMap(proto_.dnn_compiled_graphs().cbegin(),
                    proto_.dnn_compiled_graphs().cend()),
          /*gpu_version=*/gpu_device_info.gpu_compute_capability(),
          /*executable=*/ir_emitter->ConsumeThunkSequence(),
          /*constants=*/std::move(constants),
          /*output_info=*/std::move(output_info),
          /*module_name=*/std::move(hlo_module->name()),
          /*output_shape=*/std::move(output_shape),
          /*mlir_allocations=*/std::nullopt,
          /*buffer_assignment=*/std::move(buffer_assignment),
          /*debug_buffer_assignment_show_max=*/debug_buffer_assignment_show_max,
          /*debug_module=*/std::move(hlo_module),
          /*enable_debug_info_manager=*/true}));
  return executable;
}

GpuCompiler::GpuCompiler(se::Platform::Id platform_id,
                         const char* target_triple, const char* data_layout)
    : platform_id_(platform_id),
      target_triple_(target_triple),
      data_layout_(data_layout),
      pointer_size_(llvm::DataLayout(data_layout)
                        .getPointerSize(0 /* default address space */)) {}

namespace {
// Adds the HloVerifier for GPU to the given pipeline.
void AddHloVerifier(HloPassPipeline* pipeline,
                    bool verify_unique_channel_ids = false,
                    HloVerifierOpts&& opts = {}, bool debug_only = false) {
  opts.verify_unique_channel_ids = verify_unique_channel_ids;
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
  if (debug_only) {
    pipeline->AddInvariantCheckerDebug<HloVerifier>(
        std::move(verifier_metadata), "hlo verifier (debug)");
  } else {
    pipeline->AddInvariantChecker<HloVerifier>(std::move(verifier_metadata),
                                               "hlo verifier");
  }
}

void CheckNotScheduled(HloModule* hlo_module) {
  if (hlo_module->has_schedule() &&
      !hlo_module->config().debug_options().xla_disable_all_hlo_passes()) {
    LOG(WARNING) << "\nThe current HLO module " << hlo_module->name()
                 << " is scheduled and optimized. \n"
                 << "It is not expected to run optimization passes again.\n"
                    "Use a test method like RunAndCompareNoHloPasses() or "
                 << "the xla_disable_all_hlo_passes flag.";
  }
}

void LogDebugOptions(HloModule* hlo_module) {
  // LOG_LINES is used instead of LOG since the message can exceed the
  // maximum line length, which results in the message being truncated.
  XLA_VLOG_LINES(
      1, absl::StrFormat("GpuCompilationEnvironment of hlo_module %s:\n%s",
                         hlo_module->name(),
                         hlo_module->config().debug_options().DebugString()));
}

AlgebraicSimplifierOptions LayoutInsensitiveAlgebraicSimplifierOptions(
    const HloModuleConfig& hlo_module_config,
    const Compiler::TargetConfig& gpu_target_config,
    AlgebraicSimplifierOptions opts_from_compiler) {
  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts =
      opts_from_compiler;
  layout_insensitive_algsimp_opts.set_conv_is_lowerable_callback(
      ConvRewriter::ConvIsLowerable);
  layout_insensitive_algsimp_opts.set_enable_dot_strength_reduction(true);

  // GPU only supports canonical convolutions.
  layout_insensitive_algsimp_opts.set_supports_non_canonical_dots(false);

  // On GPU it helps to reorder them so that the fused cuDNN kernel can be
  // used.
  layout_insensitive_algsimp_opts.set_enable_conv_add_multiply_reorder(true);

  // "slow" minmax means we propagate nan.
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(
      !hlo_module_config.debug_options().xla_gpu_enable_fast_min_max());

  // Always simplify reduce(transpose(x)) and reduce(reshape(x)), even when
  // the transpose/reshape has multiple users.  This helps int8 models, which
  // tend to have lots of transpose+reshape's (converting between NCHW and
  // NCHW_VECT_C).  Without this, those reshape+transposes can get materialized
  // out, which is really bad for perf.
  layout_insensitive_algsimp_opts
      .set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);

  if (gpu_target_config.platform_name == "ROCM") {
    layout_insensitive_algsimp_opts.set_enable_conv_operand_swap(false);
  }
  layout_insensitive_algsimp_opts
      .set_enable_unconditional_reduce_of_concat_replacement(false);
  return layout_insensitive_algsimp_opts;
}

absl::Status RunPreSPMDPartitionerPasses(HloModule* hlo_module) {
  HloPassPipeline pre_spmd_pipeline("pre-spmd-partitioner");
  // Run some IR cleanup passes before running the SPMD partitioning
  // passes.
  pre_spmd_pipeline.AddPass<CuDnnCustomCallConverter>();
  pre_spmd_pipeline.AddPass<ConvertMemoryPlacementToInternalAnnotations>();
  pre_spmd_pipeline.AddPass<CallInliner>();
  pre_spmd_pipeline.AddPass<ZeroSizedHloElimination>();
  pre_spmd_pipeline.AddPass<ConditionalCanonicalizer>();

  // The TopkDecomposer generates a compare op with type=TOTALORDER and must
  // run before the ComparisonExpander which rewrites such comparisons.
  pre_spmd_pipeline.AddPass<TopkDecomposer>([&](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kTopK;
  });

  // The SPMD partitioner would mess up the sort+slice structure, so we need to
  // rewrite Topk before that happens.
  pre_spmd_pipeline.AddPass<TopkRewriter>(
      [](const HloSortInstruction*, int64_t) { return true; });

  return pre_spmd_pipeline.Run(hlo_module).status();
}

absl::Status RunSPMDPasses(
    HloModule* hlo_module, const Compiler::TargetConfig& gpu_target_config,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts) {
  bool auto_sharding = hlo_module->config().use_auto_spmd_partitioning();
#ifndef PLATFORM_GOOGLE
  if (auto_sharding) {
    LOG(ERROR) << "GPU autosharding is not yet available in open source.";
  }
#endif

  const int64_t num_partitions = hlo_module->config().num_partitions();
  if (num_partitions > 1 && hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    AddSPMDPasses(hlo_module, layout_insensitive_algsimp_opts,
                  gpu_target_config.device_description.gpu_compute_capability(),
                  spmd_pipeline,
#ifdef PLATFORM_GOOGLE
                  [&](HloPassPipeline& pipeline) {
                    if (auto_sharding) {
                      spmd_pipeline.AddPass<AutoSharding>(
                          DefaultAutoShardingOptionFromModuleConfig(
                              hlo_module->config()));
                    }
                  });
#else
        std::nullopt);
#endif  // PLATFORM_GOOGLE
    if (hlo_module->config()
            .debug_options()
            .xla_gpu_unsafe_pipelined_loop_annotator()) {
      spmd_pipeline.AddPass<WhileLoopTripCountAnnotator>();
      spmd_pipeline.AddPass<CollectivePermuteValidIterationAnnotator>();
    }
    return spmd_pipeline.Run(hlo_module).status();
  } else {
    HloPassPipeline sharding_removal_pipeline("sharding-removal");
    // Remove redundant sharding ops when partition_count == 1.
    sharding_removal_pipeline.AddPass<ShardingRemover>();
    sharding_removal_pipeline.AddPass<HloDCE>();
    return sharding_removal_pipeline.Run(hlo_module).status();
  }
}

absl::Status RunOptimizationPasses(
    HloModule* hlo_module, const Compiler::TargetConfig& gpu_target_config,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  HloPassPipeline pipeline("optimization");
  AddHloVerifier(&pipeline, !debug_options.xla_ignore_channel_id());
  pipeline.AddPass<BatchedGatherScatterNormalizer>();
  if (debug_options.xla_gpu_multi_streamed_windowed_einsum()) {
    pipeline.AddPass<WindowedEinsumHandler>();
  }
  pipeline.AddPass<TopKSplitter>();
  pipeline.AddPass<TopkSpecializer>();
  pipeline.AddPass<TopkDecomposer>();

  HloPredicate upcaster_filter = [&](const HloInstruction* instr) {
    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
        &gpu_target_config.device_description.gpu_compute_capability());
    if (cuda_cc != nullptr &&
        !cuda_cc->IsAtLeast(se::CudaComputeCapability::kVolta)) {
      return true;
    }
    return !gpu::IsMatrixMultiplication(*instr);
  };
  pipeline.AddPass<DotDimensionSorter>();
  pipeline.AddPass<DotDecomposer>();

  pipeline.AddPass<ResultCaster>(upcaster_filter);
  pipeline.AddPass<OperandUpcaster>(upcaster_filter);

  // Add the DotOperandConverter after any potential upcasts done as part of
  // the OperandUpcaster, so that the DotOperandConverter becomes a no-op.
  pipeline.AddPass<DotOperandConverter>();

  pipeline.AddPass<SubByteNormalization>(
      SubByteNormalization::SET_ELEMENT_SIZE);

  // Expand random number generation.
  pipeline.AddPass<RngExpander>();
  pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);

  if (hlo_module->config().debug_options().xla_gpu_enable_cub_radix_sort()) {
    pipeline.AddPass<SortRewriter>();
  }

  // Comparison total order expander
  pipeline.AddPass<ComparisonExpander>(std::array{std::make_pair(BF16, F32)});

  // Remove zero-sized HLO from the input so that other passes don't have to
  // handle it.
  pipeline.AddPass<ZeroSizedHloElimination>();

  // Rewrite select-and-scatter as a scatter and a reduce-window.
  pipeline.AddPass<SelectAndScatterExpander>();

  if (RequireDeterminism(hlo_module->config())) {
    // Scatter can be indeterministic if indices are not unique or a non
    // associative combiner function is used. Eliminate these Scatter ops.
    if (debug_options.xla_gpu_enable_scatter_determinism_expander()) {
      pipeline.AddPass<ScatterDeterminismExpander>();
    }
    pipeline.AddPass<ScatterExpander>(
        ScatterExpander::kEliminateIndeterministicScatters);
  }
  // Scatters unsupported on XLA:GPU are eliminated.
  pipeline.AddPass<GpuScatterExpander>();

  // TODO(phawkins): replace QR and Eigh decompositions with calls to
  // cuSOLVER.
  pipeline.AddPass<QrExpander>();
  pipeline.AddPass<EighExpander>();

  pipeline.AddPass<DynamicIndexSplitter>();

  pipeline.AddPass<CallInliner>();

  pipeline.AddPass<StochasticConvertDecomposer>();

  pipeline.AddPass<Convolution4DExpander>();

  // Replace PRED convolutions with F16.
  pipeline.AddPass<ConvolutionPredExpander>();

  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);

  pipeline.AddPass<LogisticExpander>();
  pipeline.AddPass<ConditionalCanonicalizer>();
  pipeline.AddPass<DynamicDimensionSimplifier>();

  if (debug_options.xla_reduce_window_rewrite_base_length() != 0) {
    pipeline.AddPass<HloPassFix<ReduceWindowRewriter>>(
        debug_options.xla_reduce_window_rewrite_base_length());
  }

  DynamicPadderOptions dynamic_padder_options;

  switch (debug_options.xla_gpu_shape_checks()) {
    case DebugOptions::IGNORE:
      dynamic_padder_options.shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kIgnore;
      break;
    case DebugOptions::RUNTIME: {
      dynamic_padder_options.shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kRuntime;
      dynamic_padder_options.assertion_generator = [&](HloInstruction* inst) {
        auto created = Cast<HloCustomCallInstruction>(
            inst->parent()->AddInstruction(HloInstruction::CreateCustomCall(
                ShapeUtil::MakeTokenShape(), {inst}, kXlaGpuAssertCustomCallTag,
                "Buffers have different size at runtime",
                API_VERSION_STATUS_RETURNING)));
        created->set_custom_call_has_side_effect(true);
      };
      break;
    }
    case DebugOptions::COMPILE_TIME:
      dynamic_padder_options.shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kCompileTime;
      break;
    default:
      LOG(FATAL) << "Unreachable";
  }

  pipeline.AddPass<DynamicPadder>(dynamic_padder_options);

  // Expand the sort op to support stable sorting if required.
  pipeline.AddPass<StableSortExpander>();

  if (hlo_module->config().debug_options().xla_gpu_enable_cub_radix_sort()) {
    pipeline.AddPass<SortRewriter>();
  }

  se::GpuComputeCapability gpu_version =
      gpu_target_config.device_description.gpu_compute_capability();

  // Build simplification pipeline.  The passes in here are run to a fixed
  // point.
  [&, &pipeline =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
    AddHloVerifier(&pipeline, !debug_options.xla_ignore_channel_id(),
                   HloVerifierOpts{}, /*debug_only=*/true);

    // BatchNormExpander can create zero-sized ops, so zero-sized HLO
    // elimination has to come after that pass.
    pipeline.AddPass<ZeroSizedHloElimination>();

    pipeline.AddPass<GatherSimplifier>();
    pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
    pipeline.AddPass<ScatterSimplifier>();
    pipeline.AddPass<ScatterExpander>(
        ScatterExpander::kEliminateSimpleScatters);
    pipeline.AddPass<ScatterSliceSimplifier>();
    pipeline.AddPass<GpuAlgebraicSimplifier>(layout_insensitive_algsimp_opts,
                                             gpu_version);
    pipeline.AddPass<BitcastDtypesExpander>();
    // Only merge "smallish" dots.  This threshold defaults to 32MB today, with
    // a flag to override.
    // Do not merge dots when they are assigned different stream ids.
    std::function<bool(const HloInstruction* dot_a,
                       const HloInstruction* dot_b)>
        can_merge = [&](const HloInstruction* dot_a,
                        const HloInstruction* dot_b) -> bool {
      return dot_a->backend_config<GpuBackendConfig>()->operation_queue_id() ==
             dot_b->backend_config<GpuBackendConfig>()->operation_queue_id();
    };
    pipeline.AddPass<DotMerger>(
        /*max_size_to_merge=*/int64_t{debug_options
                                          .xla_gpu_dot_merger_threshold_mb()}
            << 20,
        can_merge);
    pipeline.AddPass<SortSimplifier>();
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<WhileLoopConstantSinking>();
    pipeline.AddPass<WhileLoopSimplifier>();
    pipeline.AddPass<SliceSinker>();

    ReshapeMoverOptions reshape_mover_options;
    reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
    pipeline.AddPass<ReshapeMover>(reshape_mover_options);
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<ConditionalSimplifier>();
    pipeline.AddPass<RealImagExpander>();
    pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot);
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    pipeline.AddPass<HloDCE>();
  }();

  // ConvertMover and ReshapeMover fight with each other: ConvertMover wants
  // to move some converts down the graph, but ReshapeMover wants to move them
  // up the graph.  As a compromise, let ReshapeMover run to a fixed point,
  // and then run ConvertMover + algsimp to a fixed point.
  [&, &pipeline =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification-2")] {
    pipeline.AddPass<ConvertMover>();
    pipeline.AddPass<GpuAlgebraicSimplifier>(layout_insensitive_algsimp_opts,
                                             gpu_version);
  }();

  pipeline.AddPass<HloComputationDeduplicator>(
      /*mark_fusion_duplications=*/false);
  return pipeline.Run(hlo_module).status();
}

absl::Status RunCollectiveOptimizationPasses(
    HloModule* hlo_module,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    se::GpuComputeCapability gpu_version) {
  // Optimize collectives generated by SPMD partitioning. Enable these passes
  // otherwise as well so that all collectives can get these optimizations.
  const HloModuleConfig& config = hlo_module->config();
  const DebugOptions& debug_options = config.debug_options();

  HloPassPipeline collectives_pipeline("collective-optimizations");
  collectives_pipeline.AddPass<RaggedAllToAllCanonicalizer>();
  if (debug_options.xla_gpu_unsupported_enable_ragged_all_to_all_decomposer()) {
    collectives_pipeline.AddPass<RaggedAllToAllDecomposer>();
  }
  collectives_pipeline.AddPass<AllReduceSimplifier>();
  collectives_pipeline.AddPass<AllReduceFolder>();
  collectives_pipeline.AddPass<AllReduceSplitter>();
  collectives_pipeline.AddPass<AllGatherOptimizer>();
  collectives_pipeline.AddPass<AllGatherDynamicSliceSimplifier>();
  collectives_pipeline.AddPass<AllReduceReassociate>(
      debug_options.xla_gpu_enable_reassociation_for_converted_ar());
  collectives_pipeline.AddPass<ReduceScatterReassociate>();

  collectives_pipeline.AddPass<WhileLoopAllReduceCodeMotion>(
      /*enable_reduce_scatter=*/debug_options
          .xla_gpu_enable_while_loop_reduce_scatter_code_motion());

  // Moves collectives' subsequent quantization before the collective to
  // minimize data transfers.
  collectives_pipeline.AddPass<CollectiveQuantizer>();
  // Remove dead computations after collective quantization.
  collectives_pipeline.AddPass<HloDCE>();

  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_all_reduce() ||
      IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(*hlo_module)) {
    CollectivePipeliner::Config config{
        /*level_to_operate_on=*/0,
        /*max_pipelining_per_loop=*/INT64_MAX,
        /*last_run=*/true,
        /*pipeline_use_tree=*/true,
        /*process_different_sized_ops=*/true,
        /*pipelining_direction=*/
        CollectivePipeliner::PipeliningDirection::kForward,
        /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>,
        /*acceptable_formatting=*/HloPredicateTrue,
        /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
        /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateFalse,
        /*should_allow_control_dependencies=*/false,
        /*postprocess_backward_peeled_op=*/std::nullopt,
        /*postprocess_backward_rotated_op=*/std::nullopt,
        /*postprocess_backward_peeled_trailing_op=*/std::nullopt,
        /*should_add_loop_invariant_op_in_chain=*/false,
        /*postprocess_pipelined_ops=*/AppendPipelinedInstruction,
    };
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }
  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_all_gather() ||
      IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(*hlo_module)) {
    CollectivePipeliner::Config config{
        /*level_to_operate_on=*/0,
        /*max_pipelining_per_loop=*/INT64_MAX,
        /*last_run=*/true,
        /*pipeline_use_tree=*/true,
        /*process_different_sized_ops=*/true,
        /*pipelining_direction=*/
        CollectivePipeliner::PipeliningDirection::kBackward,
        /*should_process=*/HloPredicateIsOp<HloOpcode::kAllGather>,
        /*acceptable_formatting=*/HloPredicateTrue,
        /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
        /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateFalse,
        /*should_allow_control_dependencies=*/false,
        /*postprocess_backward_peeled_op=*/std::nullopt,
        /*postprocess_backward_rotated_op=*/std::nullopt,
        /*postprocess_backward_peeled_trailing_op=*/std::nullopt,
        /*should_add_loop_invariant_op_in_chain=*/true,
        /*postprocess_pipelined_ops=*/AppendPipelinedInstruction,
    };
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }
  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_reduce_scatter() ||
      IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(*hlo_module)) {
    CollectivePipeliner::Config config{
        /*level_to_operate_on=*/0,
        /*max_pipelining_per_loop=*/INT64_MAX,
        /*last_run=*/true,
        /*pipeline_use_tree=*/true,
        /*process_different_sized_ops=*/true,
        /*pipelining_direction=*/
        CollectivePipeliner::PipeliningDirection::kForward,
        /*should_process=*/HloPredicateIsOp<HloOpcode::kReduceScatter>,
        /*acceptable_formatting=*/HloPredicateTrue,
        /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
        /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateFalse,
        /*should_allow_control_dependencies=*/false,
        /*postprocess_backward_peeled_op=*/std::nullopt,
        /*postprocess_backward_rotated_op=*/std::nullopt,
        /*postprocess_backward_peeled_trailing_op=*/std::nullopt,
        /*should_add_loop_invariant_op_in_chain=*/false,
        /*postprocess_pipelined_ops=*/AppendPipelinedInstruction,
    };
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }

  collectives_pipeline.AddPass<ReduceScatterCreator>();

  DebugOptions::PipelineParallelismOptLevel pipeline_parallelism_opt_level =
      debug_options.xla_gpu_experimental_pipeline_parallelism_opt_level();
  if (pipeline_parallelism_opt_level ==
          DebugOptions::
              PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE_CYCLE_DECOMPOSER ||
      debug_options.xla_gpu_enable_pipelined_p2p()) {
    collectives_pipeline.AddPass<CollectivePermuteCycleDecomposer>(
        debug_options.xla_gpu_collective_permute_decomposer_threshold());
  }

  if (pipeline_parallelism_opt_level ==
      DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE_CYCLE_DECOMPOSER) {
    collectives_pipeline.AddPass<CollectiveSelectFolder>();
  }

  if (pipeline_parallelism_opt_level !=
          DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE ||
      debug_options.xla_gpu_enable_pipelined_p2p()) {
    collectives_pipeline.AddPass<CollectivePermuteDecomposer>(
        debug_options.xla_gpu_collective_permute_decomposer_threshold(),
        pipeline_parallelism_opt_level);
  }

  bool enable_partial_send_recv_pipelining =
      pipeline_parallelism_opt_level !=
      DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE;
  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_p2p() ||
      enable_partial_send_recv_pipelining) {
    collectives_pipeline.AddPass<GpuP2PPipeliner>(
        enable_partial_send_recv_pipelining);
  }

  // Run algebraic simplifier to reshape(broadcast) into a broadcast when
  // the reshape is just adding a unit dimension. This will help with the
  // AllGatherBroadcastReorder pass.
  collectives_pipeline.AddPass<GpuAlgebraicSimplifier>(
      layout_insensitive_algsimp_opts, gpu_version);

  collectives_pipeline.AddPass<AllGatherBroadcastReorder>();

  if (debug_options.xla_gpu_experimental_collective_cse_distance_threshold() >
      0) {
    collectives_pipeline.AddPass<ScheduleAwareCollectiveOpsCSE>(
        /*distance_threshold=*/debug_options
            .xla_gpu_experimental_collective_cse_distance_threshold(),
        /*for_replicas=*/false);
  }

  // promote 16 bit integer all-reduce and reduce-scatter to 32-bit.
  const std::pair<PrimitiveType, PrimitiveType> ar_promoted_types[] = {
      {U16, U32}, {S16, S32}};
  collectives_pipeline.AddPass<AllReducePromotion>(ar_promoted_types);
  // Remove dead computations left over after ar/rs promotion.
  collectives_pipeline.AddPass<HloDCE>();

  // Run WhileLoopTripCountAnnotator after collective pipelining and before
  // layout assignment and fusion.This pass does some pattern-matching on
  // while bodies/conditions, and this is where the HLO is "nicest".
  //
  // It's important that we don't make semantic changes (e.g. unrolling) to
  // any `while` loops after this point, because otherwise the trip-count
  // annotations added by this pass may not be correct after the
  // modifications.
  collectives_pipeline.AddPass<WhileLoopTripCountAnnotator>();

  return collectives_pipeline.Run(hlo_module).status();
}

absl::Status RunLayoutAssignmentPasses(
    HloModule* hlo_module, se::GpuComputeCapability gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::DeviceDescription& device_description) {
  // Run layout assignment in a separate pipeline from
  // "post-layout-assignment" because we want everything after layout
  // assignment to have a layout-sensitive invariant-checker, but
  // HloPassPipeline also runs its invariant checker before any passes are
  // run, meaning, the pipeline that contains layout assignment cannot contain
  // a layout-sensitive verifier!
  HloPassPipeline pipeline("layout assignment");
  // Layout assignment uses alias analysis, which requires the call graph to
  // be flattened.
  pipeline.AddPass<FlattenCallGraph>();
  ChannelLayoutConstraints layout_constraints;
  pipeline.AddPass<GpuLayoutAssignment>(
      hlo_module->mutable_entry_computation_layout(), gpu_version, dnn_version,
      device_description, &layout_constraints);
  // Run SubByteNormalization because GpuLayoutAssignment may modify a
  // Layout's element_size_in_bits field.
  pipeline.AddPass<SubByteNormalization>(
      SubByteNormalization::SET_ELEMENT_SIZE);
  pipeline.AddPass<OptimizeInputOutputBufferAlias>(true);
  // Run HostOffloadLegalize before LayoutNormalization to prevent
  // the creation of invalid transpose/bitcast operations within
  // host memory offloading segments.
  pipeline.AddPass<HostOffloadLegalize>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost),
      /* after_layout= */ true);
  return pipeline.Run(hlo_module).status();
}

absl::Status RunFusionPasses(HloModule* hlo_module,
                             const Compiler::TargetConfig& gpu_target_config,
                             tsl::thread::ThreadPool* thread_pool,
                             HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  const se::DeviceDescription& gpu_device_info =
      gpu_target_config.device_description;

  TF_RETURN_IF_ERROR(FusionPipeline(hlo_module->config().debug_options(),
                                    shape_size_fn, thread_pool, gpu_device_info)
                         .Run(hlo_module)
                         .status());

  TF_RETURN_IF_ERROR(
      HorizontalFusionPipeline(gpu_device_info).Run(hlo_module).status());

  if (VLOG_IS_ON(2)) {
    HloFusionStatsVisitor stats;
    TF_RETURN_IF_ERROR(hlo_module->entry_computation()->Accept(&stats));
    VLOG(2) << stats.ToString();
  }

  return absl::OkStatus();
}

// Adds unrolling while loop optimization. Mostly to get rid of extra D2D
// copies, but also there are some performance benefits (better comm-compute
// overlap) when collectives are present within a while loop.
void AddDoubleBufferingPasses(const HloModule& module,
                              HloPassPipeline& pipeline) {
  const DebugOptions& opts = module.config().debug_options();
  std::optional<DoubleBufferLoopUnrolling::UnrollStrategy> unroll_strategy =
      std::nullopt;
  // Support old flag.
  if (opts.xla_gpu_enable_while_loop_double_buffering()) {
    unroll_strategy = DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer;
  }
  // Support new flag setting style, override the old one.
  if (opts.xla_gpu_enable_while_loop_unrolling() ==
      DebugOptions::WHILE_LOOP_UNROLLING_DOUBLE_BUFFER) {
    unroll_strategy = DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer;
  }
  if (opts.xla_gpu_enable_while_loop_unrolling() ==
      DebugOptions::WHILE_LOOP_UNROLLING_FULL_UNROLL) {
    LOG_IF(WARNING, unroll_strategy != std::nullopt)
        << "Overriding double buffering set via "
           "`xla_gpu_enable_while_loop_double_buffering` flag.";
    unroll_strategy = DoubleBufferLoopUnrolling::UnrollStrategy::kFullUnroll;
  }
  if (opts.xla_gpu_enable_while_loop_unrolling() ==
          DebugOptions::WHILE_LOOP_UNROLLING_AUTO_UNROLL &&
      IsPassEnabledAtOptimizationEffort<DoubleBufferLoopUnrolling>(module) &&
      !opts.xla_gpu_enable_while_loop_double_buffering()) {
    unroll_strategy = DoubleBufferLoopUnrolling::UnrollStrategy::kAuto;
  }
  if (unroll_strategy != std::nullopt) {
    pipeline.AddPass<WhileLoopSimplifier>();
    pipeline.AddPass<DoubleBufferLoopUnrolling>(*unroll_strategy);
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<HloDCE>();
  }
}

absl::Status RunPostFusionPasses(
    HloModule* hlo_module, const se::DeviceDescription& device_description,
    int pointer_size, const int combine_threshold_count) {
  const DebugOptions& opts = hlo_module->config().debug_options();

  HloPassPipeline pipeline("post-fusion optimization");
  pipeline.AddPass<RenameFusions>();
  pipeline.AddPass<GpuAllGatherCombiner>(
      device_description,
      /*default_combine_threshold_in_bytes=*/kDefaultAllGatherCombineThreshold,
      /*combine_threshold_in_bytes=*/
      opts.xla_gpu_all_gather_combine_threshold_bytes(),
      combine_threshold_count,
      /*combine_by_dim=*/opts.xla_gpu_enable_all_gather_combine_by_dim(),
      /*combine_different_dtypes=*/true, /*pointer_size=*/pointer_size);
  pipeline.AddPass<GpuAllReduceCombiner>(
      device_description, kDefaultAllReduceCombineThreshold,
      opts.xla_gpu_all_reduce_combine_threshold_bytes(),
      combine_threshold_count, /*pointer_size=*/pointer_size);
  pipeline.AddPass<GpuReduceScatterCombiner>(
      device_description, /*default_combine_threshold_in_bytes=*/
      kDefaultReduceScatterCombineThreshold,
      /*combine_threshold_in_bytes=*/
      opts.xla_gpu_reduce_scatter_combine_threshold_bytes(),
      combine_threshold_count,
      /*combine_by_dim=*/opts.xla_gpu_enable_reduce_scatter_combine_by_dim(),
      /*pointer_size=*/pointer_size);

  pipeline.AddPass<AllReduceContiguous>();

  int32_t blueconnect_num_devices_per_host =
      hlo_module->config()
          .debug_options()
          .xla_gpu_all_reduce_blueconnect_num_devices_per_host();
  if (blueconnect_num_devices_per_host > 0) {
    pipeline.AddPass<AllReduceBlueConnect>(blueconnect_num_devices_per_host);
  }

  AddDoubleBufferingPasses(*hlo_module, pipeline);

  return pipeline.Run(hlo_module).status();
}

absl::Status RunPostFusionSimplificationPasses(
    HloModule* hlo_module,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    se::GpuComputeCapability gpu_version,
    const Compiler::TargetConfig& gpu_target_config) {
  HloPassPipeline pipeline("post-fusion-simplification-pipeline optimization");
  AlgebraicSimplifierOptions options = layout_insensitive_algsimp_opts;
  options.set_is_layout_sensitive(true);
  pipeline.AddPass<GpuAlgebraicSimplifier>(options, gpu_version);

  // This invocation is used to populate deduplicated_name for fusions that
  // are considered duplicates according to the comparator in this pass.
  // Currently, the pass doesn't actually deduplicate the fusions.
  pipeline.AddPass<HloComputationDeduplicator>(
      /*mark_fusion_duplications=*/true);

  if (hlo_module->config()
          .debug_options()
          .xla_gpu_multi_streamed_windowed_einsum()) {
    pipeline.AddPass<StreamAttributeAnnotator>(
        gpu_target_config.device_description);
    pipeline.AddPass<StreamAttributeAsyncWrapper>();
  }
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_experimental_stream_annotation()) {
    pipeline.AddPass<ExplicitStreamAnnotationAsyncWrapper>();
  }
  return pipeline.Run(hlo_module).status();
}

absl::Status RunPostFusionVerificationPasses(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const GpuCompiler::CompileOptions& options,
    const Compiler::TargetConfig& gpu_target_config) {
  HloPassPipeline pipeline("post-fusion-verification-pipeline optimization");

  if (hlo_module->config()
          .debug_options()
          .xla_gpu_verify_triton_fusion_numerics()) {
    TF_ASSIGN_OR_RETURN(
        AutotuneConfig autotune_config,
        GetAutotuneConfig(stream_exec, hlo_module->config().debug_options(),
                          options, gpu_target_config));

    pipeline.AddPass<TritonFusionNumericsVerifier>(autotune_config);
  }

  return pipeline.Run(hlo_module).status();
}

absl::Status RunLayoutNormalizationPasses(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version) {
  HloPassPipeline layout_normalization_pipeline("layout normalization");
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  AlgebraicSimplifierOptions opts =
      GpuCompiler::GetAlgebraicSimplifierOptions(hlo_module->config());
  opts.set_supports_non_canonical_dots(false);
  opts.set_is_layout_sensitive(true);
  opts.set_enable_conv_operand_swap(false);
  opts.set_enable_conv_add_multiply_reorder(true);
  // "slow" minmax means we propagate nan.
  opts.set_minmax_propagate_nan(!debug_options.xla_gpu_enable_fast_min_max());
  opts.set_enable_unconditional_reduce_of_concat_replacement(false);

  layout_normalization_pipeline.AddPass<ReshapeDecomposer>();
  layout_normalization_pipeline.AddPass<HloPassFix<MoveCopyToUsers>>();
  layout_normalization_pipeline.AddPass<LayoutNormalization>(
      &NormalizeLayoutForGpuCustomCalls);
  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  layout_normalization_pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(
      opts, gpu_version);
  // Layout normalization will create broadcasts that are not canonical.
  layout_normalization_pipeline.AddPass<BroadcastCanonicalizer>();
  // Layout normalization will create scatters that are not simplified and
  // also have unsorted update_window_dims.
  layout_normalization_pipeline.AddPass<ScatterSimplifier>();
  return layout_normalization_pipeline.Run(hlo_module).status();
}

absl::Status RunAsyncDotPasses(HloModule* hlo_module) {
  HloPassPipeline pipeline("async-wrapper");
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  if (debug_options.xla_gpu_async_dot()) {
    pipeline.AddPass<AsyncWrapper>([](HloInstruction* instruction) {
      // TODO(b/339654953): Use a better heuristic to determine whether a
      // `dot` operation should be wrapped in an async computation.
      if (IsCublasGemm(*instruction)) {
        return true;
      }
      if (instruction->called_computations().size() == 1 &&
          IsTritonFusedComputation(
              *instruction->called_computations().front())) {
        return true;
      }
      return false;
    });
  }
  return pipeline.Run(hlo_module).status();
}

absl::Status RunDynamicSliceFusionPasses(
    HloModule* hlo_module, se::Platform::Id platform_id,
    const se::DeviceDescription& device_description, int64_t pointer_size,
    const int combine_threshold_count) {
  const DebugOptions& opts = hlo_module->config().debug_options();
  if (opts.xla_gpu_enable_dynamic_slice_fusion()) {
    HloPassPipeline pipeline("dynamic-slice");
    TF_ASSIGN_OR_RETURN(se::Platform * platform,
                        se::PlatformManager::PlatformWithId(platform_id));
    pipeline.AddPass<GpuReduceScatterCombiner>(
        device_description, /*default_combine_threshold_in_bytes=*/
        kDefaultReduceScatterCombineThreshold,
        /*combine_threshold_in_bytes=*/
        opts.xla_gpu_reduce_scatter_combine_threshold_bytes(),
        /*combine_threshold_count=*/combine_threshold_count,
        /*combine_by_dim=*/opts.xla_gpu_enable_reduce_scatter_combine_by_dim(),
        /*pointer_size=*/pointer_size);
    pipeline.AddPass<DynamicSliceFusionRewriter>(platform->Name());
    pipeline.AddPass<AsyncWrapper>([](const HloInstruction* instr) {
      if (!IsDynamicSliceFusion(instr)) {
        return false;
      }
      std::optional<const HloInstruction*> hero_op = HloBfsFindIf(
          {instr->fused_instructions_computation()->root_instruction()},
          [](const HloInstruction* instr) -> bool {
            return IsCollective(instr);
          });
      return hero_op.has_value();
    });
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  return absl::OkStatus();
}
}  // namespace

absl::Status GpuCompiler::RunCollectiveScheduleLinearizerPasses(
    HloModule* hlo_module, se::StreamExecutor* stream_exec) {
  HloPassPipeline pipeline("collective-schedule-linearizer");
  pipeline.AddPass<CollectivesScheduleLinearizer>(
      [this, stream_exec](const HloModule* module) {
        return RequiresCollectiveScheduleLinearizer(module, stream_exec);
      });
  return pipeline.Run(hlo_module).status();
}

// Runs optimization passes on the given HLO module.
absl::Status GpuCompiler::OptimizeHloModule(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config) {
  tsl::profiler::TraceMe traceme("GpuCompiler::OptimizeHloModule");
  const se::DeviceDescription& device_description =
      gpu_target_config.device_description;

  CheckNotScheduled(hlo_module);
  LogDebugOptions(hlo_module);

  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      /*parallelism=*/hlo_module->config()
          .debug_options()
          .xla_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/options.thread_pool,
      /*default_parallelism=*/tsl::port::MaxParallelism());

  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts =
      LayoutInsensitiveAlgebraicSimplifierOptions(
          hlo_module->config(), gpu_target_config,
          GetAlgebraicSimplifierOptions(hlo_module->config()));

  TF_RETURN_IF_ERROR(RunPreSPMDPartitionerPasses(hlo_module));
  TF_RETURN_IF_ERROR(RunSPMDPasses(hlo_module, gpu_target_config,
                                   layout_insensitive_algsimp_opts));
  TF_RETURN_IF_ERROR(RunOptimizationPasses(hlo_module, gpu_target_config,
                                           layout_insensitive_algsimp_opts));
  se::GpuComputeCapability gpu_version =
      device_description.gpu_compute_capability();
  TF_RETURN_IF_ERROR(RunCollectiveOptimizationPasses(
      hlo_module, layout_insensitive_algsimp_opts, gpu_version));

  // Run target-specific HLO optimization passes for convolution
  // canonicalization.
  se::dnn::VersionInfo dnn_version = gpu_target_config.dnn_version_info;
  if (stream_exec != nullptr) {
    gpu_version = GetGpuVersion(stream_exec);
    TF_ASSIGN_OR_RETURN(dnn_version, GetDnnVersionInfo(stream_exec));
  }

  TF_RETURN_IF_ERROR(OptimizeHloConvolutionCanonicalization(
      hlo_module, gpu_version, dnn_version,
      device_description.runtime_version()));

  TF_RETURN_IF_ERROR(RunLayoutAssignmentPasses(
      hlo_module, gpu_version, dnn_version, device_description));

  TF_RETURN_IF_ERROR(RunLayoutNormalizationPasses(hlo_module, gpu_version));

  // Run target-specific HLO optimization passes after layout assignment.
  TF_RETURN_IF_ERROR(OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config,
      thread_pool.get_mutable()));

  const int combine_threshold_count = 256;

  // This is a "low effort, high impact" fusion that should be run first.
  TF_RETURN_IF_ERROR(RunDynamicSliceFusionPasses(
      hlo_module, /*platform_id=*/PlatformId(),
      /*device_description=*/gpu_target_config.device_description,
      /*pointer_size=*/pointer_size_, combine_threshold_count));

  TF_RETURN_IF_ERROR(RunFusionPasses(hlo_module, gpu_target_config,
                                     thread_pool.get_mutable(),
                                     ShapeSizeBytesFunction()));
  TF_RETURN_IF_ERROR(RunPostFusionPasses(
      hlo_module, device_description, pointer_size_, combine_threshold_count));
  TF_RETURN_IF_ERROR(RunAsyncCollectivesConversionPasses(hlo_module));
  TF_RETURN_IF_ERROR(RunPostFusionSimplificationPasses(
      hlo_module, layout_insensitive_algsimp_opts, gpu_version,
      gpu_target_config));

  TF_RETURN_IF_ERROR(RunPostFusionVerificationPasses(
      hlo_module, stream_exec, options, gpu_target_config));

  TF_RETURN_IF_ERROR(
      RunCollectiveScheduleLinearizerPasses(hlo_module, stream_exec));

  TF_RETURN_IF_ERROR(RunAsyncDotPasses(hlo_module));

  return absl::OkStatus();
}  // NOLINT(readability/fn_size)

AlgebraicSimplifierOptions GpuCompiler::GetAlgebraicSimplifierOptions(
    const HloModuleConfig& config) {
  AlgebraicSimplifierOptions opts;
  opts.set_enable_dot_strength_reduction(true);
  return opts;
}

absl::Status GpuCompiler::RunPreSchedulingCopyInsertion(
    HloModule& hlo_module, const se::DeviceDescription& device_description) {
  return PreSchedulingCopyInsertionPipeline(
             hlo_module.config(), GetCanShareBuffer(device_description),
             device_description)
      .Run(&hlo_module)
      .status();
}

namespace {
void AddGemmRewriterPasses(HloPassPipeline& pipeline,
                           const DebugOptions& debug_options,
                           const se::GpuComputeCapability gpu_version,
                           const se::SemanticVersion& toolkit_version) {
  // Adding bias to GEMMs is helpful for skipping kernel launches for `add`
  // operations. However, the bias term can add dependencies between the GEMMs
  // that could otherwise be parallelized. Because of this, we disable bias
  // addition when async dot is enabled.
  GemmRewriterOptions::BiasMode bias_mode =
      GemmRewriterOptions::BiasMode::kBias;
  if (debug_options.xla_gpu_async_dot()) {
    bias_mode = GemmRewriterOptions::BiasMode::kNoBias;
  }

  // Rewrite dots with the algorithms that cannot be handled by cublas directly.
  // I.e. transform single dot into a chain of dots with the default algorithm
  // that cublas can handle. These dots were inlined by the CallInliner pass
  // above.
  pipeline.AddPass<DotAlgorithmRewriter>();

  pipeline.AddPass<GemmRewriter>(
      gpu_version, toolkit_version,
      GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only, bias_mode});
  pipeline.AddPass<GemmRewriter>(
      gpu_version, toolkit_version,
      GemmRewriterOptions{GemmRewriterOptions::DType::kNonFp8Only, bias_mode});
}
}  // namespace

absl::Status GpuCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config,
    tsl::thread::ThreadPool* thread_pool) {
  // Constants:
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  const se::GpuComputeCapability gpu_version =
      gpu_target_config.device_description.gpu_compute_capability();
  const AlgebraicSimplifierOptions simplifier_options = [&] {
    AlgebraicSimplifierOptions opts =
        GetAlgebraicSimplifierOptions(hlo_module->config());
    opts.set_supports_non_canonical_dots(false);
    opts.set_is_layout_sensitive(true);
    opts.set_enable_conv_operand_swap(false);
    opts.set_enable_conv_add_multiply_reorder(true);
    // "slow" minmax means we propagate nan.
    opts.set_minmax_propagate_nan(!debug_options.xla_gpu_enable_fast_min_max());
    opts.set_enable_unconditional_reduce_of_concat_replacement(false);
    return opts;
  }();
  TF_ASSIGN_OR_RETURN(AutotuneConfig autotune_config,
                      GetAutotuneConfig(stream_exec, debug_options, options,
                                        gpu_target_config));
  // Lambdas and related constants:
  const GpuFloatSupport bf16_support(gpu_version, BF16);
  const GpuFloatSupport f8e5m2_support(gpu_version, F8E5M2, F16);
  const GpuFloatSupport f8e4m3_support(gpu_version, F8E4M3, F16);
  const GpuFloatSupport f8e4m3fn_support(gpu_version, F8E4M3FN, F16);
  const FloatSupport f8e4m3b11fnuz_support(F8E4M3B11FNUZ, F16);
  const GpuFloatSupport f8e5m2fnuz_support(gpu_version, F8E5M2FNUZ, F16);
  const GpuFloatSupport f8e4m3fnuz_support(gpu_version, F8E4M3FNUZ, F16);
  const GpuFloatSupport f8e3m4_support(gpu_version, F8E3M4, F16);
  const GpuFloatSupport s4_support(gpu_version, S4, S8);
  const GpuFloatSupport u4_support(gpu_version, U4, U8);
  const GpuFloatSupport f4e2m1fn_support(gpu_version, F4E2M1FN, F16);
  const GpuFloatSupport f8e8m0fnu_support(gpu_version, F8E8M0FNU, F32);
  auto add_float_normalization = [&](HloPassPipeline& pipeline) {
    auto& sub_pipeline =
        pipeline.AddPass<HloPassPipeline>("float_normalization");
    sub_pipeline.AddPass<FloatNormalization>(&bf16_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e5m2_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3fn_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3b11fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e5m2fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e3m4_support);
    sub_pipeline.AddPass<FloatNormalization>(&s4_support);
    sub_pipeline.AddPass<FloatNormalization>(&u4_support);
    sub_pipeline.AddPass<FloatNormalization>(&f4e2m1fn_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e8m0fnu_support);
    // Remove `f32 -> bf16 -> f32` casts inserted by bf16 normalization.
    if (debug_options.xla_allow_excess_precision()) {
      sub_pipeline.AddPass<SimplifyFPConversions>();
    }
  };

  {
    HloPassPipeline pipeline("hlo normalization");

    // The LayoutAssignment pass may leave behind kCopy instructions which are
    // duplicate or NOPs, so remove them with algebraic simplification and CSE.
    pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                         gpu_version);

    // GemmRewriter assumes that all transposes are folded into gemms, but,
    // since commit 7d529df, this is not always true at this point.
    // Therefore, rerun transpose folding.
    pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot,
                                       TransposeFolding::NeverFoldTranspose);

    pipeline.AddPass<ReshapeDecomposer>();
    pipeline.AddPass<ReduceDecomposer>([&](const HloInstruction* r) {
      return IsReductionFromOrToContiguousDimensions(
          *r, gpu_target_config.device_description);
    });

    // Greedy pattern matching for custom kernel fusions. We run it before
    // Triton rewriter or a regular Gemm rewriter to be able to match compatible
    // GEMMs before they matched into Triton gemm or a cuBLAS custom call.
    if (debug_options.xla_gpu_enable_custom_fusions()) {
      pipeline.AddPass<SimplifyFPConversions>();
      pipeline.AddPass<CustomKernelFusionRewriter>(
          &gpu_target_config.device_description);
      pipeline.AddPass<CustomKernelFusionAutotuner>(autotune_config);
    }

    // Rewrite GEMMs into custom calls.
    se::GpuComputeCapability gpu_version =
        gpu_target_config.device_description.gpu_compute_capability();
    pipeline.AddPass<AlgorithmChecker>(gpu_version);
    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version);
    const auto* rocm_cc = std::get_if<se::RocmComputeCapability>(&gpu_version);

    // Make sure that dots have at least 1 contracting dimension in the
    // operands. Needs to happen shortly before the dot rewrite, as otherwise
    // AlgebraicSimplifier will simplify it away again.
    // TODO(b/375566188): Figure out whether we can get rid of this pass.
    pipeline.AddPass<DotNormalizer>();
    if (debug_options.xla_gpu_enable_triton_gemm() &&
        ((cuda_cc != nullptr &&
          cuda_cc->IsAtLeast(se::CudaComputeCapability::kAmpere)) ||
         rocm_cc != nullptr)) {
      pipeline.AddPass<GemvRewriter>();
      pipeline.AddPass<GemmFusion>(gpu_version);
      pipeline.AddPass<GemmFusionSwapOperands>();
    } else if (cuda_cc != nullptr &&
               cuda_cc->major == se::CudaComputeCapability::kVolta) {
      // Greedy pattern matching for custom kernel fusions.
      pipeline.AddPass<SimplifyFPConversions>();
      pipeline.AddPass<CustomKernelFusionRewriter>(
          &gpu_target_config.device_description);
      pipeline.AddPass<CustomKernelFusionAutotuner>(autotune_config);
    }

    // Rewrite GEMMs into custom calls.
    AddGemmRewriterPasses(
        pipeline, debug_options, gpu_version,
        gpu_target_config.device_description.runtime_version());

    // Rewrite GEMMs with broadcasted inputs as strided GEMMs.
    pipeline.AddPass<GemmBroadcastFoldingRewriter>();

    pipeline.AddPass<LayoutNormalization>(&NormalizeLayoutForGpuCustomCalls);
    // Remove any redundant operations (such as bitcasts) introduced by layout
    // normalization.
    pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                         gpu_version);
    // Layout normalization will create scatters that are not simplified and
    // also have unsorted update_window_dims.
    pipeline.AddPass<ScatterSimplifier>();
    pipeline.AddPass<BroadcastCanonicalizer>();

    pipeline.AddPass<TransposeDimensionGrouper>();
    pipeline.AddPass<ReductionDegenerateDimRemover>();
    pipeline.AddPass<ReductionLayoutNormalizer>();
    // Run Softmax fusion after layout normalization. We expect a default layout
    // in the softmax codegen pipeline. However we should run before
    // ReductionDimensionGrouper, as that makes matching the softmax pattern
    // harder.
    if ((cuda_cc != nullptr &&
         cuda_cc->IsAtLeast(se::CudaComputeCapability::kAmpere)) ||
        rocm_cc != nullptr) {
      pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                           gpu_version);
      pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
      pipeline.AddPass<HloConstantFolding>();
      pipeline.AddPass<HloDCE>();
      pipeline.AddPass<SoftmaxRewriterTriton>(
          gpu_target_config.device_description, ShapeSizeBytesFunction(),
          /*only_fuse_if_profitable=*/true);
    }

    pipeline.AddPass<ReductionDimensionGrouper>();
    pipeline.AddPass<HloPassFix<ReductionSplitter>>(
        gpu_target_config.device_description,
        /*ignore_small_reduce_dims=*/false);
    pipeline.AddPass<HloPassFix<TreeReductionRewriter>>(
        gpu_target_config.device_description);
    // Normalization passes might have introduced s4 tensors without bit width
    // annotations, this pass will add the annotations.
    pipeline.AddPass<SubByteNormalization>(
        SubByteNormalization::SET_ELEMENT_SIZE);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  HloPassPipeline pipeline("post-layout_assignment");
  AddHloVerifier(&pipeline, !debug_options.xla_ignore_channel_id(),
                 HloVerifierOpts{}
                     .MakeLayoutSensitive()
                     .WithInstructionCanChangeLayout(
                         LayoutAssignment::InstructionCanChangeLayout)
                     .VerifyBroadcastDimensionsOrder()
                     .VerifyReshapeIsBitcast(),
                 /*debug_only=*/true);

  // Triton compilation needs normalized operations on bf16 (i.e. converted to
  // f32).
  add_float_normalization(pipeline);

  TF_RETURN_IF_ERROR(AddGemmFusionAutotuningPasses(
      &pipeline, hlo_module, autotune_config, thread_pool,
      options.key_value_store,
      gpu_target_config.device_description.runtime_version()));

  if (debug_options
          .xla_gpu_unsupported_enable_generic_triton_emitter_for_gemms()) {
    pipeline.AddPass<NestGemmFusion>();
  }
  // Inline back the calls which have better performance with cuBLAS.
  pipeline.AddPass<CallInliner>();
  // TODO(tdanyluk): Apply CublasPadForGemms to the cuBLAS GEMMs generated
  // here for possibly better cuBLAS performance.

  AddGemmRewriterPasses(pipeline, debug_options, gpu_version,
                        gpu_target_config.device_description.runtime_version());

  // Rewrite GEMMs with broadcasted inputs as strided GEMMs.
  pipeline.AddPass<GemmBroadcastFoldingRewriter>();

  // Recover host-offloader invariants (such as the single-use broadcast buffer
  // initialization before loops) by re-running the offload legalizer.
  pipeline.AddPass<HostOffloadLegalize>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost),
      /* after_layout= */ true);

  pipeline.AddPass<LayoutNormalization>(&NormalizeLayoutForGpuCustomCalls);

  // Layout normalization will create scatters that are not simplified and
  // also have unsorted update_window_dims.
  pipeline.AddPass<ScatterSimplifier>();

  // Verify the host memory space before the host offloader pass
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(
          HloVerifierOpts{}.VerifyNoHostMemorySpace());
  pipeline.AddPass<HloVerifier>(std::move(verifier_metadata));

  pipeline.AddPass<HostOffloader>();

  TF_RETURN_IF_ERROR(
      AddConvAndGemmAutotuningPasses(&pipeline, gpu_version, options,
                                     hlo_module, autotune_config, thread_pool));

  // The GEMM fusion autotuner can insert new bf16 reductions that need to be
  // normalized again.
  add_float_normalization(pipeline);

  // Clean up new_tuple described above.
  pipeline.AddPass<TupleSimplifier>();

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                       gpu_version);

  if (debug_options.xla_allow_excess_precision()) {
    // This pass cleans up chains of compiler-generated converts
    // (i.e. f32 -> bf16 -> f32) that have been produced by the algebraic
    // simplifier by rearranging ops (i.e. by pushing broadcasts towards the
    // root).
    pipeline.AddPass<SimplifyFPConversions>();
  }

  {
    // Because of an issue with JAX remat and `SimplifyFPConversions` (see PR:
    // https://github.com/jax-ml/jax/pull/22244), we can only eliminate the
    // no-op reduce-precision operations after the last call to
    // `SimplifyFPConversions`. We are creating a sub-pipeline here because that
    // allows us to test this order in a unit test.
    HloPassPipeline& remove_no_op_reduce_precision_pipeline =
        pipeline.AddPass<HloPassPipeline>(
            "remove-no-op-reduce-precision-algebraic-simplifier");
    AlgebraicSimplifierOptions simplifier_options_{simplifier_options};
    simplifier_options_.set_enable_remove_no_op_reduce_precision(true);
    remove_no_op_reduce_precision_pipeline
        .AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options_,
                                                     gpu_version);
  }

  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);

  pipeline.AddPass<HostMemoryTransferAsyncifier>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost));

#ifdef NDEBUG
  // Verify the module in non-debug builds. For debug builds, the verifier
  // already runs after every pass.
  HloVerifierOpts opts = HloVerifierOpts{}
                             .MakeLayoutSensitive()
                             .WithInstructionCanChangeLayout(
                                 LayoutAssignment::InstructionCanChangeLayout)
                             .VerifyBroadcastDimensionsOrder()
                             .VerifyReshapeIsBitcast();
  opts.verify_unique_channel_ids = !debug_options.xla_ignore_channel_id();
  pipeline.AddPass<HloVerifier>(
      std::make_unique<DefaultVerifierMetadata>(std::move(opts)),
      "end-of-post-layout_assignment");
#endif  // NDEBUG

  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  return absl::OkStatus();
}

// Returns the TargetConfig, either from the module debug options, or from the
// CompilationOptions, or if both of those are absent, from the attached GPU.
/*static*/ absl::StatusOr<Compiler::TargetConfig> GpuCompiler::GetTargetConfig(
    const Compiler::CompileOptions& options, const DebugOptions& debug_opts,
    se::StreamExecutor* executor) {
  if (options.target_config.has_value()) {
    return *options.target_config;
  }
  if (!debug_opts.xla_gpu_target_config_filename().empty()) {
    std::string gpu_target_config_string;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(), debug_opts.xla_gpu_target_config_filename(),
        &gpu_target_config_string));
    stream_executor::GpuTargetConfigProto gpu_target_config_proto;
    if (!tsl::protobuf::TextFormat::ParseFromString(gpu_target_config_string,
                                                    &gpu_target_config_proto)) {
      return absl::FailedPreconditionError(
          "Failed to parse GpuTargetConfigProto");
    }

    return Compiler::TargetConfig{gpu_target_config_proto};
  }
  if (executor) {
    Compiler::TargetConfig target_config = Compiler::TargetConfig{executor};
    int64_t device_memory_size =
        target_config.device_description.device_memory_size();
    // Checking for device_memory_size == -1 is how we detect that we are
    // running on Nvidia's software simulator. When running on simulation,
    // the config from StreamExecutor is inaccurate, so we must load the
    // hard-coded config from a file.
    if (device_memory_size == -1) {
      return absl::FailedPreconditionError(
          "When running on an NVIDIA simulation device, you must use "
          "--xla_gpu_target_config_filename to pass in target information. "
          "The target config from StreamExecutor is inaccurate.");
    }
    return target_config;
  }
  return absl::InternalError(
      "Either GPU has to be attached, or --xla_gpu_target_config_filename "
      "has to be specified to specify the target to compile for.");
}

absl::StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  const DebugOptions debug_opts = module->config().debug_options();
  TF_RETURN_IF_ERROR(LoadAutotuneResultsFromFile(debug_opts));
  bool is_deviceless = options.target_config.has_value() ||
                       !debug_opts.xla_gpu_target_config_filename().empty();

  TF_ASSIGN_OR_RETURN(TargetConfig gpu_target_config,
                      GetTargetConfig(options, debug_opts, stream_exec));
  const std::optional<std::string> unoptimized_fingerprint =
      MaybeUploadUnoptimizedGpuSymbols(module.get(),
                                       gpu_target_config.ToProto());

  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("GpuCompiler::RunHloPasses for ", module->name()),
      !options.is_autotuning_compilation);
  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tsl::profiler::TraceMeLevel::kInfo);

  TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(),
                                       is_deviceless ? nullptr : stream_exec,
                                       options, gpu_target_config));

  const se::DeviceDescription& device_description =
      gpu_target_config.device_description;
  TF_RETURN_IF_ERROR(
      RunPreSchedulingCopyInsertion(*module, device_description));

  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
      &device_description.gpu_compute_capability());
  if (cuda_cc != nullptr && cuda_cc->IsAtLeastAmpere()) {
    // This needs to run after every pass affecting fusions, which includes
    // `CopyFusion`, which runs just before.
    TF_RETURN_IF_ERROR(
        FusionDispatchPipeline(device_description, ShapeSizeBytesFunction())
            .Run(module.get())
            .status());
  }

  uint64_t end_usecs = tsl::Env::Default()->NowMicros();

  // This won't record values for calls that error out (because if they error
  // out we have no way of telling how far through the process we got).
  RecordHloPassesDuration(end_usecs - start_usecs);

  DumpHloModuleMetadataIfEnabled({module.get()});

  AutotuneResults autotune_results;
  TF_ASSIGN_OR_RETURN(
      AutotuneConfig autotune_config,
      GetAutotuneConfig(stream_exec, debug_opts, options, gpu_target_config));
  if (!is_deviceless) {
    TF_RETURN_IF_ERROR(
        AutotunerUtil::SerializeAutotuneResults(&autotune_results));
    TF_RETURN_IF_ERROR(SerializeAutotuneResultsToFile(debug_opts));
  }
  const std::optional<std::string> optimized_fingerprint =
      MaybeUploadOptimizedGpuSymbols(module.get(), autotune_results);
  if (unoptimized_fingerprint.has_value() &&
      optimized_fingerprint.has_value()) {
    MaybeUploadGpuSymbolMapping(*unoptimized_fingerprint,
                                *optimized_fingerprint);
  }

  if (DumpingEnabledForHloModule(*module)) {
    TF_ASSIGN_OR_RETURN(
        std::string autotune_results,
        AutotunerUtil::SerializeAutotuneResults(/*as_textproto=*/true));
    DumpToFileInDirOrStdout(*module, "", "autotune_results.pbtxt",
                            autotune_results);
  }

  return std::move(module);
}

namespace {
absl::Status RunPostSchedulingCopyInsertion(
    HloModule* module,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer) {
  // We run a separate pass of copy elision here because the sequential ordering
  // from the HLO schedule potentially allows for more copies to be eliminated.
  constexpr int64_t kRegionBasedLiveRangeAnalysisLimit = -1;
  const int64_t kUseRegionBasedLiveRangeAnalysis =
      module->config()
              .debug_options()
              .xla_gpu_copy_insertion_use_region_analysis()
          ? kRegionBasedLiveRangeAnalysisLimit
          : 0;
  CopyInsertion copy_insertion(can_share_buffer,
                               kUseRegionBasedLiveRangeAnalysis);
  TF_RETURN_IF_ERROR(copy_insertion.RemoveUnnecessaryCopies(module));

  // Stash away the schedule during copy insertion, to avoid validation failures
  // while the module is in flux.
  HloSchedule saved_schedule = module->schedule();
  module->clear_schedule();

  // RemoveUnnecessaryCopies only considers interference when determining
  // whether it is legal to remove a copy. However, copies in the graph may be
  // necessary for other reason such as preventing a constant from being live
  // out of the graph. So run AddSpecialCaseCopies to re-insert these copies.
  TF_RETURN_IF_ERROR(
      copy_insertion.CopyInsertion::AddSpecialCaseCopies(module));

  TF_RETURN_IF_ERROR(HloDCE().Run(module).status());

  // The passes above can add and remove copies, update the schedule to
  // account for these transformations. Newly added instructions will be
  // placed ASAP in the schedule.

  // Update and restore the schedule. The saved schedule has a reference to the
  // updated HLO module. The saved schedule needs to be updated before restoring
  // it to the module to avoid validation failures.
  TF_RETURN_IF_ERROR(saved_schedule.Update());
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(saved_schedule)));

  return absl::OkStatus();
}
}  // namespace

using OutputInfoMap =
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;

static void NullDiagnosticHandler(const llvm::DiagnosticInfo* diag_info,
                                  void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info->print(diagnostic_printer);

  VLOG(5) << error_string;
}

namespace {

std::unique_ptr<llvm::Module> CopyToContext(const llvm::Module& module,
                                            llvm::LLVMContext& context) {
  // We are setting llvm::SmallString's InternalLen to 0, because we want to
  // allocate its buffer on the heap. We use llvm::SmallString instead of
  // std::string, because llvm::raw_svector_ostream is a bit faster than
  // llvm::raw_string_ostream.
  llvm::SmallString<0> bitcode;
  llvm::raw_svector_ostream bitcode_ostream(bitcode);
  llvm::WriteBitcodeToFile(module, bitcode_ostream);

  llvm::Expected<std::unique_ptr<llvm::Module>> new_module =
      llvm::parseBitcodeFile(
          llvm::MemoryBufferRef(llvm::StringRef(bitcode.data(), bitcode.size()),
                                "split_module"),
          context);
  CHECK(new_module) << "Failed to parse bitcode "
                    << llvm::toString(new_module.takeError());

  return std::move(new_module.get());
}

}  // namespace

absl::StatusOr<GpuCompiler::BackendCompileResult>
GpuCompiler::CompileSingleModule(
    const HloModuleConfig& module_config,
    const stream_executor::DeviceDescription& device_description,
    const HloModule* debug_module, llvm::Module* llvm_module, bool relocatable,
    const CompileOptions& options, std::optional<int> shard_number) {
  {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    XLA_SCOPED_LOGGING_TIMER_IF(
        absl::StrCat(
            "GpuCompiler::RunBackend - Running LLVM verifier for ",
            (debug_module != nullptr ? debug_module->name() : "(unknown)")),
        VLOG_IS_ON(4) && !options.is_autotuning_compilation);

    llvm_module->getContext().setDiagnosticHandlerCallBack(
        NullDiagnosticHandler, nullptr);

    std::string err;
    llvm::raw_string_ostream err_stream(err);

    // verifyModule() returns true if the module is broken.
    TF_RET_CHECK(!llvm::verifyModule(*llvm_module, &err_stream))
        << "Invalid LLVM IR before optimizations:\n"
        << err_stream.str()
        << "\nThis probably indicates a bug in the HLO -> LLVM IR "
           "lowering. Rerun with --xla_dump_to to get the IR"
        << (debug_module
                ? absl::StrCat(" and looks for files with name containing: *",
                               FilenameFor(*debug_module, "", ""), "*")
                : ".");
  }

  TF_ASSIGN_OR_RETURN(
      BackendCompileResult result,
      CompileTargetBinary(module_config, llvm_module, device_description,
                          relocatable, debug_module, options, shard_number));

  const bool should_dump = DumpingEnabledForHloModule(
      debug_module ? debug_module->name() : "", module_config.debug_options());

  if (should_dump) {
    if (debug_module) {
      llvm_ir::DumpIrIfEnabled(
          *debug_module, *llvm_module,
          /*optimized=*/true,
          shard_number.has_value() ? std::to_string(*shard_number) : "");
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
  }

  if (user_post_optimization_hook_) {
    user_post_optimization_hook_(*llvm_module);
  }

  return result;
}

namespace {
int CountFunctions(const llvm::Module& module) {
  int num_functions = 0;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      ++num_functions;
    }
  }
  return num_functions;
}

// Returns the name of the single function in the module or empty string if it's
// not a single-function module.
std::string SingleFunctionName(const llvm::Module& module) {
  std::string name;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      if (name.empty()) {
        // First function in a module: name the module with it.
        name = func.getName().str();
      } else {
        // Not the first function - the module is not cacheable.
        return "";
      }
    }
  }
  return name;
}
}  // namespace

absl::StatusOr<GpuCompiler::BackendCompileResult> GpuCompiler::CompileAndLink(
    const HloModuleConfig& module_config,
    CompileModuleResults& compile_module_results,
    const se::DeviceDescription& device_description,
    se::StreamExecutor* stream_exec, const CompileOptions& options,
    const HloModule* debug_module) {
  llvm::Module* llvm_module = &*compile_module_results.llvm_module;

  bool force_module_split =
      module_config.debug_options().xla_llvm_force_inline_before_split();
  if (force_module_split) {
    for (llvm::Function& func : llvm_module->functions()) {
      if (func.getNumUses() > 0 && !func.isDeclaration()) {
        VLOG(4) << absl::StrFormat("Inlining function %s with %d users.\n",
                                   func.getName().str(), func.getNumUses());
        std::vector<llvm::CallInst*> calls_to_inline;
        for (auto* user : func.users()) {
          if (auto* call = llvm::dyn_cast<llvm::CallInst>(user)) {
            calls_to_inline.push_back(call);
          }
        }
        for (auto* call_to_inline : calls_to_inline) {
          llvm::InlineFunctionInfo inline_function_info;
          if (!llvm::InlineFunction(*call_to_inline, inline_function_info)
                   .isSuccess()) {
            return absl::InternalError("Can not inline function " +
                                       func.getName().str());
          };
        }
      }
    }
  }

  // Record the name of some constant global variables and their initializers.
  // We'll change the linkage type of these variables from external to internal
  // to ensure constant-folding works properly after calling llvm::SplitModule.
  llvm::DenseMap<llvm::StringRef, llvm::Constant*> const_initializer_map;
  llvm::Module& module_with_constants =
      (compile_module_results.llvm_module_constants == nullptr)
          ? *llvm_module
          : *compile_module_results.llvm_module_constants;
  for (llvm::GlobalVariable& gv : module_with_constants.globals()) {
    if (gv.hasName() && gv.isConstant() && gv.hasInitializer() &&
        gv.hasExternalLinkage()) {
      llvm::Constant* initializer = gv.getInitializer();
      unsigned int num_elements = 0;
      if (auto* caz =
              llvm::dyn_cast<llvm::ConstantAggregateZero>(initializer)) {
        num_elements = caz->getElementCount().getFixedValue();
      } else if (auto* cds = llvm::dyn_cast<llvm::ConstantDataSequential>(
                     initializer)) {
        num_elements = cds->getNumElements();
      }
      if (num_elements > 0) {
        const_initializer_map[gv.getName()] = initializer;
      }
    }
  }

  llvm_ir::DumpIrIfEnabled(*debug_module, *llvm_module,
                           /*optimized=*/false, "inlined");

  absl::string_view cache_path =
      module_config.debug_options().xla_gpu_kernel_cache_file();
  const bool use_cache = !cache_path.empty();

  struct NamedModule {
    // The string is the function name for single-function modules (used to
    // cache them), empty for all other modules.
    std::string name;
    std::unique_ptr<llvm::Module> module;
  };
  std::vector<NamedModule> llvm_modules;
  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      /*parallelism=*/module_config.debug_options()
          .xla_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/options.thread_pool,
      /*default_parallelism=*/1);
  // Only single-function module are cacheable -> for caching try to get 1
  // function per module. If caching is not used limit the number of modules to
  // the number of threads.
  int num_modules = CountFunctions(*llvm_module);
  if (thread_pool.get() != nullptr && !use_cache) {
    num_modules = std::max(1, std::min(thread_pool->NumThreads(), num_modules));
  }
  if (compile_module_results.llvm_module_constants != nullptr) {
    llvm_modules.reserve(num_modules + 1);
    llvm_modules.push_back(
        {"", std::move(compile_module_results.llvm_module_constants)});
  } else {
    llvm_modules.reserve(num_modules);
  }
  int single_function_module_count = 0;
  llvm::SplitModule(
      *llvm_module, num_modules,
      [&](std::unique_ptr<llvm::Module> module) {
        // Change the linkage type of some global constant variables to internal
        for (llvm::GlobalVariable& gv : module->globals()) {
          if (gv.hasName() && gv.isConstant() && !gv.hasInitializer() &&
              const_initializer_map.count(gv.getName()) != 0) {
            gv.setInitializer(const_initializer_map[gv.getName()]);
            gv.setLinkage(llvm::GlobalValue::InternalLinkage);
          }
        }
        const std::string name = SingleFunctionName(*module);
        if (!name.empty()) {
          ++single_function_module_count;
        }
        llvm_modules.push_back({name, std::move(module)});
      },
      /*PreserveLocals=*/true, /*RoundRobin=*/true);
  VLOG(2) << "Single-function cacheable modules: "
          << single_function_module_count << " / " << llvm_modules.size();

  struct NamedCompileResult {
    // Single function name or empty just like for llvm_modules.
    std::string name;
    absl::StatusOr<BackendCompileResult> result;
  };
  std::vector<NamedCompileResult> compile_results(llvm_modules.size());
  if (thread_pool.get() != nullptr) {
    absl::BlockingCounter counter(llvm_modules.size());
    for (int i = 0; i < llvm_modules.size(); ++i) {
      thread_pool.get_mutable()->Schedule(
          [&compile_results, i, &llvm_modules, &counter, this, &module_config,
           &device_description, &debug_module, &options] {
            // Each thread has its own context to avoid race conditions.
            llvm::LLVMContext new_context;
            std::unique_ptr<llvm::Module> new_module =
                CopyToContext(*llvm_modules.at(i).module, new_context);
            compile_results.at(i) = {
                llvm_modules.at(i).name,
                CompileSingleModule(module_config, device_description,
                                    debug_module, new_module.get(),
                                    /*relocatable=*/true, options,
                                    /*shard_number=*/i)};
            counter.DecrementCount();
          });
    }
    counter.Wait();
  } else {
    for (int i = 0; i < llvm_modules.size(); ++i) {
      compile_results.at(i) = {
          llvm_modules.at(i).name,
          CompileSingleModule(module_config, device_description, debug_module,
                              &*llvm_modules.at(i).module,
                              /*relocatable=*/true, options,
                              /*shard_number=*/i)};
    }
  }

  std::string ptx_snippets;
  std::vector<std::vector<uint8_t>> binaries_to_link;
  binaries_to_link.reserve(compile_results.size());
  std::vector<KernelReuseCache::NamedBinary> binaries_to_cache;
  binaries_to_cache.reserve(single_function_module_count);
  for (const auto& [name, maybe_result] : compile_results) {
    TF_ASSIGN_OR_RETURN(auto result, maybe_result);
    if (result.binary.empty()) {
      continue;
    }
    ptx_snippets += result.asm_text;
    ptx_snippets += "\n";
    binaries_to_link.push_back(result.binary);
    if (!name.empty()) {
      binaries_to_cache.push_back({name, result.binary});
    }
  }

  if (use_cache) {
    std::string resolved_path;
    if (!tsl::io::ResolveTestPrefixes(cache_path, resolved_path)) {
      return FailedPrecondition("File path can not be resolved: %s",
                                cache_path);
    }
    // current_cache contains new kernels from the current compilation and
    // kernels to reuse from previous compilations if some were loaded from the
    // cache file.
    const CompilationCacheProto& current_cache =
        compile_module_results.kernel_compilation_cache;
    const bool cache_file_exists =
        tsl::Env::Default()->FileExists(resolved_path).ok();
    if (cache_file_exists) {
      // Pick reused binaries from previous compilations needed to link the
      // current executable.
      int loaded_kernel_count = 0;
      for (const auto& [name, entry] : current_cache.entries()) {
        if (llvm_module->getFunction(name) != nullptr) {
          VLOG(5) << "Using the just compiled kernel for " << name;
          TF_RET_CHECK(entry.binary().empty())
              << name
              << " is a just compiled kernel and is not expected to have a "
                 "binary yet.";
          continue;
        }
        const uint8_t* binary =
            reinterpret_cast<const uint8_t*>(entry.binary().data());
        binaries_to_link.push_back(
            std::vector<uint8_t>(binary, binary + entry.binary().size()));
        VLOG(5) << "Using " << name << " from cache: " << entry.binary().size();
        ++loaded_kernel_count;
      }
      VLOG(2) << "Using " << loaded_kernel_count << " / "
              << current_cache.entries_size() << " cached kernels.";
    }
    if (!binaries_to_cache.empty()) {
      TF_RETURN_IF_ERROR(
          UpdateDiskKernelCache(resolved_path, /*do_append=*/cache_file_exists,
                                current_cache, binaries_to_cache));
    }
  }

  auto maybe_backend_result =
      LinkModules(device_description, stream_exec, std::move(binaries_to_link),
                  module_config.debug_options());
  if (!maybe_backend_result.ok()) {
    LOG(ERROR) << "The CUDA linking API did not work. Please use XLA_FLAGS="
                  "--xla_gpu_enable_llvm_module_compilation_parallelism=false "
                  "to bypass it, but expect to get longer compilation time due "
                  "to the lack of multi-threading. Original error: "
               << maybe_backend_result.status();
    return maybe_backend_result.status();
  }
  VLOG(4) << "Binary size after linking [B]: " << maybe_backend_result->size();
  compile_module_results.kernel_compilation_cache.Clear();
  return BackendCompileResult{ptx_snippets, std::move(*maybe_backend_result)};
}

absl::StatusOr<GpuCompiler::CompileResultWithMetadata>
GpuCompiler::CompileToBackendResult(
    HloModule* module, llvm::LLVMContext* llvm_context,
    se::StreamExecutor* executor, const CompileOptions& options,
    const se::DeviceDescription& gpu_device_info) {
  tsl::profiler::TraceMe traceme("GpuCompiler::CompileToBackendResult");

  TF_RETURN_IF_ERROR(RunPreSchedulingPasses(module, executor, gpu_device_info));
  TF_ASSIGN_OR_RETURN(
      ScheduleMetadata schedule_metadata,
      ScheduleGpuModule(module, pointer_size_, gpu_device_info));
  TF_RETURN_IF_ERROR(RunPostSchedulingPipelines(
      module, schedule_metadata.scheduler_mem_limit, gpu_device_info));

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithId(PlatformId()));

  // Test whether LinkModules is supported.
  bool can_use_link_modules = (executor != nullptr);
  if (can_use_link_modules) {
    TF_ASSIGN_OR_RETURN(can_use_link_modules,
                        CanUseLinkModules(module->config(), gpu_device_info));
  }
  const bool split_modules =
      can_use_link_modules &&
      module->config()
          .debug_options()
          .xla_gpu_enable_llvm_module_compilation_parallelism();
  const bool use_cache =
      split_modules &&
      !module->config().debug_options().xla_gpu_kernel_cache_file().empty();

  CompileModuleResults compile_module_results;

  {
    xla::llvm_ir::LLVMCommandLineOptionsLock llvm_options_lock(
        GetLLVMCommandLineOptions(module->config().debug_options()));
    // Compile the module to thnks and llvm IR.
    TF_ASSIGN_OR_RETURN(
        compile_module_results,
        CompileModuleToLlvmIr(module, llvm_context, target_triple_,
                              data_layout_, platform, gpu_device_info,
                              GetCanShareBuffer(gpu_device_info),
                              BufferSizeBytesFunction(),
                              /*split_constants_module=*/use_cache));
  }

  if (user_pre_optimization_hook_) {
    user_pre_optimization_hook_(*compile_module_results.llvm_module);
    if (compile_module_results.llvm_module_constants != nullptr) {
      user_pre_optimization_hook_(
          *compile_module_results.llvm_module_constants);
    }
  }

  llvm_ir::DumpIrIfEnabled(*module, *compile_module_results.llvm_module,
                           /*optimized=*/false);
  if (compile_module_results.llvm_module_constants != nullptr) {
    llvm_ir::DumpIrIfEnabled(*module,
                             *compile_module_results.llvm_module_constants,
                             /*optimized=*/false, "constants");
  }

  BackendCompileResult backend_result;
  // Disable multi-threading during deviceless AOT compilation.
  // TODO(anlunx): Enable multi-threading once deviceless AOT compilation is
  // enabled.
  if (split_modules) {
    TF_ASSIGN_OR_RETURN(
        backend_result,
        CompileAndLink(module->config(), compile_module_results,
                       gpu_device_info, executor, options, module));
  } else {
    CHECK(compile_module_results.llvm_module_constants == nullptr);
    TF_ASSIGN_OR_RETURN(
        backend_result,
        CompileSingleModule(module->config(), gpu_device_info, module,
                            &*compile_module_results.llvm_module,
                            /*relocatable=*/false, options,
                            /*shard_number=*/std::nullopt));
  }
  RecordXlaDeviceBinarySize(backend_result.binary.size());
  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(
        *module, "", "thunk_sequence.txt",
        compile_module_results.executable->ToString(/*indent=*/0));
  }

  return CompileResultWithMetadata{std::move(backend_result),
                                   std::move(compile_module_results)};
}

absl::StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  tsl::profiler::ScopedAnnotation backend_annotation{[&] {
    return absl::StrFormat("XlaCompileBackend:#module=%s,program_id=%d#",
                           module->name(), module->unique_id());
  }};

  RecordGpuCompilerStacktrace();

  BinaryMap dnn_compiled_graphs;
  if (stream_exec) {
    TF_RETURN_IF_ERROR(RunCudnnCompilerPasses(module.get(), stream_exec,
                                              &dnn_compiled_graphs));
  }

  const DebugOptions& debug_opts = module->config().debug_options();
  TF_ASSIGN_OR_RETURN(TargetConfig gpu_target_config,
                      GetTargetConfig(options, debug_opts, stream_exec));

  if (DumpingEnabledForHloModule(*module)) {
    std::string textproto;
    tsl::protobuf::TextFormat::PrintToString(gpu_target_config.ToProto(),
                                             &textproto);
    DumpToFileInDirOrStdout(*module, "", "gpu_target_config.pbtxt", textproto);
  }

  if (!options.is_autotuning_compilation) {
    VLOG(1) << "Starting to compile HLO module " << module->name();
  }

  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("GpuCompiler::RunBackend for ", module->name()),
      !options.is_autotuning_compilation);
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

  if (options.is_autotuning_compilation) {
    if (module->config().debug_options().xla_embed_ir_in_executable()) {
      LOG(WARNING) << "Doing autotuning compilations with "
                      "xla_embed_ir_in_executable wastes memory!";
    }
  }

  llvm::LLVMContext llvm_context;
  const se::DeviceDescription& gpu_device_info =
      gpu_target_config.device_description;

  if (module->config().hlo_profiling_enabled() || VLOG_IS_ON(1)) {
    HloCostAnalysis::Options cost_analysis_options{ShapeSizeBytesFunction()};
    cost_analysis_options.set_bytes_per_second(
        gpu_device_info.memory_bandwidth());
    GpuHloCostAnalysis cost_analysis(cost_analysis_options, gpu_device_info);
    TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&cost_analysis));
    if (!options.is_autotuning_compilation) {
      VLOG(1) << "HLO memory read+written: "
              << tsl::strings::HumanReadableNumBytes(
                     cost_analysis.bytes_accessed());
    }
    if (module->config().hlo_profiling_enabled()) {
      LOG(ERROR) << "--xla_hlo_profile for GPU is unsupported.";
    }
  }

  TF_ASSIGN_OR_RETURN(
      CompileResultWithMetadata res,
      CompileToBackendResult(module.get(), &llvm_context, stream_exec, options,
                             gpu_device_info));

  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(
        *module, "", "thunk_sequence.txt",
        res.compile_module_results.executable->ToString(/*indent=*/0));
  }

  // The module is being moved into the GpuExecutable below and we need to
  // read a few config values from the module, before it becomes invalid.
  bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  int64_t debug_buffer_assignment_show_max =
      module->config().debug_options().xla_debug_buffer_assignment_show_max();

  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaCreateGpuExecutable:#module=%s#",
                           module->name());
  });
  TF_ASSIGN_OR_RETURN(
      auto gpu_executable,
      GpuExecutable::Create(GpuExecutable::Params{
          /*asm_text=*/(options.is_autotuning_compilation &&
                        !res.backend_result.binary.empty())
              ? std::string()
              : std::move(res.backend_result.asm_text),
          /*binary=*/std::move(res.backend_result.binary),
          /*dnn_compiled_graphs=*/
          std::move(dnn_compiled_graphs),
          /*gpu_version=*/gpu_device_info.gpu_compute_capability(),
          /*executable=*/std::move(res.compile_module_results.executable),
          /*constants=*/std::move(res.compile_module_results.constants),
          /*output_info=*/std::move(res.compile_module_results.output_info),
          /*module_name=*/std::move(res.compile_module_results.module_name),
          /*output_shape=*/std::move(res.compile_module_results.output_shape),
          /*mlir_allocations=*/
          (res.compile_module_results.use_original_allocations
               ? std::optional<std::vector<BufferAllocation>>()
               : std::move(res.compile_module_results.allocations)),
          /*buffer_assignment=*/
          std::move(res.compile_module_results.buffer_assignment),
          /*debug_buffer_assignment_show_max=*/
          debug_buffer_assignment_show_max,
          /*debug_module=*/options.is_autotuning_compilation
              ? std::unique_ptr<HloModule>()
              : std::move(module),
          /*enable_debug_info_manager=*/!options.is_autotuning_compilation}));

  if (embed_ir_in_executable) {
    std::string ir_module_string_before_opt =
        llvm_ir::DumpToString(res.compile_module_results.llvm_module.get());
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
    DCHECK_NE("", ir_module_string_before_opt);
  }

  IncrementCompiledProgramsCount();

  if (!options.is_autotuning_compilation && gpu_executable->has_module()) {
    // Dump computation proto state and buffer assignment for
    // CompiledMemoryAnalysis.
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_buffer_assignment() =
        gpu_executable->buffer_assignment()->ToProto();
    gpu_executable->set_hlo_proto(std::move(hlo_proto));
    gpu_executable->set_debug_info(
        gpu_executable->buffer_assignment()->StatsString(
            /*report_total_fragmentation=*/true));
  }

  return static_cast<std::unique_ptr<Executable>>(std::move(gpu_executable));
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
GpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& options) {
  // Check that we are on the platform (CUDA or ROCm) that was chosen for AOT
  // compilation.
  CHECK_EQ(options.PlatformId(), PlatformId());

  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();

  std::vector<std::unique_ptr<HloModule>> optimized_modules;
  optimized_modules.reserve(modules.size());

  for (std::unique_ptr<HloModule>& module : modules) {
    if (!module->has_schedule()) {
      tsl::profiler::ScopedAnnotation annotation{[&] {
        return absl::StrFormat("XlaCompile:#module=%s,program_id=%d#",
                               module->name(), module->unique_id());
      }};
      CompileOptions compile_options;
      compile_options.device_allocator = options.device_allocator();
      compile_options.target_config = options.target_config();
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<HloModule> optimized_module,
          RunHloPasses(std::move(module), options.executor(), compile_options));
      optimized_modules.push_back(std::move(optimized_module));
    } else {
      optimized_modules.push_back(std::move(module));
    }
  }

  modules = std::move(optimized_modules);

  std::vector<std::unique_ptr<AotCompilationResult>> results;

  const std::optional<Compiler::TargetConfig>& target_config =
      options.target_config();
  CHECK(target_config.has_value() || options.executor() != nullptr);
  const se::DeviceDescription& gpu_device_info =
      target_config.has_value() ? target_config->device_description
                                : options.executor()->GetDeviceDescription();
  for (const std::unique_ptr<HloModule>& module : modules) {
    llvm::LLVMContext llvm_context;
    TF_ASSIGN_OR_RETURN(
        CompileResultWithMetadata res,
        CompileToBackendResult(module.get(), &llvm_context, options.executor(),
                               {options.device_allocator()}, gpu_device_info));

    // Create GpuThunkAotCompilationResult if thunk runtime is enabled.
    TF_ASSIGN_OR_RETURN(
        results.emplace_back(),
        GpuThunkAotCompilationResult::FromModule(
            module.get(), res.compile_module_results.buffer_assignment.get(),
            res.backend_result.asm_text, res.backend_result.binary,
            res.backend_result.dnn_compiled_graphs));
  }

  return std::move(results);
}

HloCostAnalysis::ShapeSizeFunction GpuCompiler::ShapeSizeBytesFunction() const {
  // Capture just the pointer size, not the entire GpuCompiler object.
  return gpu::ShapeSizeBytesFunction(pointer_size_);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>> GpuCompiler::Export(
    Executable* executable) const {
  auto* gpu_executable = tensorflow::down_cast<GpuExecutable*>(executable);
  if (!gpu_executable) return Internal("GpuExecutable is null");

  return GpuThunkAotCompilationResult::FromModule(
      &gpu_executable->module(), gpu_executable->buffer_assignment(),
      gpu_executable->text(), gpu_executable->binary(),
      gpu_executable->dnn_compiled_graphs());
}

absl::Status GpuCompiler::RunPreSchedulingPasses(
    HloModule* module, se::StreamExecutor* stream_exec,
    const se::DeviceDescription& gpu_device_info) {
  HloPassPipeline pipeline("pre-scheduling-passes");
  pipeline.AddPass<FusionWrapper>(gpu_device_info);
  if (module->config().debug_options().xla_gpu_collect_cost_model_stats()) {
    GpuHloCostAnalysis::Options cost_analysis_options{
        ShapeSizeBytesFunction(),
        /*per_second_rates=*/{},
        /*min_latencies_seconds=*/{},
        /*count_multiple_input_accesses=*/true};
    // Cost model analysis for compute.
    pipeline.AddPass<GpuCostModelStatsCollection>(gpu_device_info,
                                                  cost_analysis_options);
    // Cost model analysis for collectives.
    pipeline.AddPass<SolGpuCostModelStatsCollection>(gpu_device_info,
                                                     ShapeSizeBytesFunction());
  }
  return pipeline.Run(module).status();
}

HloCostAnalysis::Options CreateHloAnalysisOpts(
    const HloModule& module, const se::DeviceDescription& gpu_device_info,
    ShapeSizeFn shape_size_fn) {
  HloCostAnalysis::Options hlo_cost_analysis_options;
  hlo_cost_analysis_options.shape_size = shape_size_fn;
  std::optional<HloRematerialization::HostMemoryOffloadConfig>
      offloading_config = std::nullopt;
  if (module.config().debug_options().xla_gpu_enable_host_memory_offloading()) {
    constexpr float kGiga = 1e+9;
    // Fused multiply-add means that these two instructions are computed as
    // one, so for this case the maximum flops is doubled.
    constexpr float kFma = 2;
    float flops_per_sec = gpu_device_info.core_count() *
                          gpu_device_info.fpus_per_core() *
                          gpu_device_info.clock_rate_ghz() * kGiga * kFma;
    int64_t host_memory_space_color =
        static_cast<int64_t>(se::MemoryType::kHost);
    hlo_cost_analysis_options.set_flops_per_second(flops_per_sec);
    hlo_cost_analysis_options.set_transcendentals_per_second(flops_per_sec);
    offloading_config =
        std::make_optional<HloRematerialization::HostMemoryOffloadConfig>(
            /*host_memory_space=*/host_memory_space_color,
            /*bandwidth_to_host_bytes_per_second=*/
            gpu_device_info.memory_bandwidth(),
            /*bandwidth_from_host_bytes_per_second=*/
            gpu_device_info.memory_bandwidth());
  }
  return hlo_cost_analysis_options;
}

HloRematerialization::Options CreateRematOpts(
    const HloModule& module, const se::DeviceDescription& gpu_device_info,
    HloCostAnalysis& hlo_cost_analysis, int64_t scheduler_mem_limit) {
  bool enable_offloading =
      module.config().debug_options().xla_gpu_enable_host_memory_offloading();
  std::optional<HloRematerialization::HostMemoryOffloadConfig>
      offloading_config = std::nullopt;
  if (enable_offloading) {
    int64_t host_memory_space_color =
        static_cast<int64_t>(se::MemoryType::kHost);
    offloading_config =
        std::make_optional<HloRematerialization::HostMemoryOffloadConfig>(
            /*host_memory_space=*/host_memory_space_color,
            /*bandwidth_to_host_bytes_per_second=*/
            gpu_device_info.memory_bandwidth(),
            /*bandwidth_from_host_bytes_per_second=*/
            gpu_device_info.memory_bandwidth());
  }
  HloRematerialization::RematerializationModeConfig
      rematerialization_mode_config(/*recompute=*/true, /*compress=*/true,
                                    /*host_offload=*/enable_offloading);
  HloRematerialization::Options options(
      hlo_cost_analysis, rematerialization_mode_config,
      // Assume 75% of the total device memory is available for XLA.
      /*memory_limit_bytes=*/scheduler_mem_limit,
      /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
      /*min_remat_size=*/0, /*compact_shape_function=*/nullptr,
      /*host_memory_offload_config=*/offloading_config);
  return options;
}

absl::Status GpuCompiler::RunPostSchedulingPipelines(
    HloModule* module, int64_t scheduler_mem_limit,
    const se::DeviceDescription& gpu_device_info) const {
  TF_RETURN_IF_ERROR(RunPostSchedulingCopyInsertion(
      module, GetCanShareBuffer(gpu_device_info)));
  HloPassPipeline main_pipeline("post-scheduling-passes");

  // Pipeline for async -> sync conversion on for non-overlapped async ops.
  {
    HloPassPipeline& pipeline =
        main_pipeline.AddPass<HloPassPipeline>("async-to-sync-converter");

    if (module->config()
                .debug_options()
                .xla_gpu_experimental_pipeline_parallelism_opt_level() ==
            DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE &&
        (module->config()
             .debug_options()
             .xla_gpu_enable_pipelined_collectives() ||
         module->config().debug_options().xla_gpu_enable_pipelined_p2p())) {
      pipeline.AddPass<PipelinedP2PRewriter>();
    }
    pipeline.AddPass<GpuConvertAsyncCollectivesToSync>();
  }

  // Pipeline rematerialization passes with optional host offloading.
  HloRematerialization::RematerializationSizes sizes;
  // `HloCostAnalysis` initialization.
  HloCostAnalysis::Options hlo_cost_analysis_opts =
      CreateHloAnalysisOpts(*module, gpu_device_info, ShapeSizeBytesFunction());
  HloCostAnalysis hlo_cost_analysis(hlo_cost_analysis_opts);
  // `HloRematerialization` options initialization.
  HloRematerialization::Options remat_opts = CreateRematOpts(
      *module, gpu_device_info, hlo_cost_analysis, scheduler_mem_limit);
  {
    HloPassPipeline& pipeline =
        main_pipeline.AddPass<HloPassPipeline>("remat-pipeline");

    pipeline.AddPass<HloRematerialization>(remat_opts, sizes);
    pipeline.AddPass<StreamAttributeAnnotator>(gpu_device_info);
    pipeline.AddPass<OptimizationBarrierExpander>();
  }

  // Wrap remaining unfused ops that have no LHLO equivalent in single-op
  // fusions. This needs to happen after rematerialization, because that
  // will insert additional copies.
  {
    HloPassPipeline& pipeline =
        main_pipeline.AddPass<HloPassPipeline>("fusion-wrapper");
    pipeline.AddPass<FusionWrapper>(gpu_device_info);
  }

  // Pipeline with passes which wrap a scheduled module into command buffers.
  {
    HloPassPipeline& pipeline =
        main_pipeline.AddPass<HloPassPipeline>("command-buffer-scheduling");
    pipeline.AddPass<CommandBufferScheduling>(gpu_device_info);
    pipeline.AddPass<SanitizeConstantNames>();
  }

  if (module->config().debug_options().xla_gpu_pgle_accuracy_checker() ==
      DebugOptions::PGLE_STRICTNESS_LEVEL_ERROR) {
    AddHloVerifier(&main_pipeline,
                   module->config().debug_options().xla_ignore_channel_id(),
                   HloVerifierOpts{}.VerifyInstructionNameUnchanged());
  }
  return main_pipeline.Run(module).status();
}

absl::Status GpuCompiler::LoadAutotuneResultsFromFile(
    const DebugOptions& debug_options) {
  // We are doing this before the timer is started.
  if (absl::string_view file_path =
          debug_options.xla_gpu_load_autotune_results_from();
      !file_path.empty()) {
    static absl::once_flag once;
    absl::Status status = absl::OkStatus();
    absl::call_once(once, [&file_path, &status] {
      status = AutotunerUtil::LoadAutotuneResultsFromFile(file_path);
    });
    TF_RETURN_IF_ERROR(status);
  }
  return absl::OkStatus();
}

absl::Status GpuCompiler::SerializeAutotuneResultsToFile(
    const DebugOptions& debug_options) {
  // We are doing this after the timer is finished.
  if (absl::string_view file_path =
          debug_options.xla_gpu_dump_autotune_results_to();
      !file_path.empty()) {
    // Warning: This writes the autotune results at every compilation,
    // possibly multiple times per process.
    TF_RETURN_IF_ERROR(
        AutotunerUtil::SerializeAutotuneResultsToFile(file_path));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
GpuCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  return LoadAotCompilationResultStatic(serialized_aot_result);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
GpuCompiler::LoadAotCompilationResultStatic(
    const std::string& serialized_aot_result) {
  return GpuThunkAotCompilationResult::FromString(serialized_aot_result);
}

}  // namespace gpu
}  // namespace xla

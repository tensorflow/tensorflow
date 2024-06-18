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
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/maybe_owning.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/all_gather_broadcast_reorder.h"
#include "xla/service/all_gather_combiner.h"
#include "xla/service/all_reduce_combiner.h"
#include "xla/service/all_reduce_contiguous.h"
#include "xla/service/all_reduce_folder.h"
#include "xla/service/all_reduce_promotion.h"
#include "xla/service/all_reduce_reassociate.h"
#include "xla/service/all_reduce_splitter.h"
#include "xla/service/async_collective_creator.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/broadcast_canonicalizer.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_inliner.h"
#include "xla/service/collective_permute_decomposer.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/collectives_schedule_linearizer.h"
#include "xla/service/comparison_expander.h"
#include "xla/service/compiler.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/convert_memory_placement_to_internal_annotations.h"
#include "xla/service/convert_mover.h"
#include "xla/service/convolution_4d_expander.h"
#include "xla/service/convolution_pred_expander.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/dot_merger.h"
#include "xla/service/dump.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_dimension_simplifier.h"
#include "xla/service/dynamic_index_splitter.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/eigh_expander.h"
#include "xla/service/executable.h"
#include "xla/service/export_hlo.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gather_simplifier.h"
#include "xla/service/gpu/algorithm_checker.h"
#include "xla/service/gpu/all_reduce_blueconnect.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/collective_permute_cycle_decomposer.h"
#include "xla/service/gpu/command_buffer_scheduling.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/conv_layout_normalization.h"
#include "xla/service/gpu/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/dot_dimension_sorter.h"
#include "xla/service/gpu/dot_operand_converter.h"
#include "xla/service/gpu/double_buffer_loop_unrolling.h"
#include "xla/service/gpu/dynamic_slice_fusion_rewriter.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/fusion_wrapper.h"
#include "xla/service/gpu/gemm_broadcast_folding_rewriter.h"
#include "xla/service/gpu/gemm_fusion.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/gpu/gemv_rewriter.h"
#include "xla/service/gpu/gpu_algebraic_simplifier.h"
#include "xla/service/gpu/gpu_all_gather_optimizer.h"
#include "xla/service/gpu/gpu_async_collective_annotator.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/gpu_convert_async_collectives_to_sync.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/gpu_layout_assignment.h"
#include "xla/service/gpu/gpu_p2p_pipeliner.h"
#include "xla/service/gpu/gpu_reduce_scatter_creator.h"
#include "xla/service/gpu/gpu_sanitize_constant_names.h"
#include "xla/service/gpu/gpu_scatter_expander.h"
#include "xla/service/gpu/gpu_spmd_pipeline.h"
#include "xla/service/gpu/gpu_windowed_einsum_handler.h"
#include "xla/service/gpu/hlo_fusion_stats.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_unnested.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/model/gpu_cost_model_stats_collection.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/move_copy_to_users.h"
#include "xla/service/gpu/pipelined_p2p_rewriter.h"
#include "xla/service/gpu/prepare_hlo_for_ir_emitting_pipeline.h"
#include "xla/service/gpu/reduction_degenerate_dim_remover.h"
#include "xla/service/gpu/reduction_dimension_grouper.h"
#include "xla/service/gpu/reduction_layout_normalizer.h"
#include "xla/service/gpu/reduction_splitter.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/gpu/rename_fusions.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/runtime_intrinsics.h"
#include "xla/service/gpu/scatter_slice_simplifier.h"
#include "xla/service/gpu/softmax_rewriter_triton.h"
#include "xla/service/gpu/stream_attribute_annotator.h"
#include "xla/service/gpu/stream_attribute_async_wrapper.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/topk_specializer.h"
#include "xla/service/gpu/topk_splitter.h"
#include "xla/service/gpu/tree_reduction_rewriter.h"
#include "xla/service/gpu/triton_fusion_numerics_verifier.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_computation_deduplicator.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_rematerialization.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/host_memory_transfer_asyncifier.h"
#include "xla/service/host_offload_legalize.h"
#include "xla/service/host_offloader.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logistic_expander.h"
#include "xla/service/operand_upcaster.h"
#include "xla/service/optimization_barrier_expander.h"
#include "xla/service/optimize_input_output_buffer_alias.h"
#include "xla/service/qr_expander.h"
#include "xla/service/real_imag_expander.h"
#include "xla/service/reduce_decomposer.h"
#include "xla/service/reduce_scatter_combiner.h"
#include "xla/service/reduce_scatter_reassociate.h"
#include "xla/service/reduce_window_rewriter.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/result_caster.h"
#include "xla/service/rng_bit_generator_expander.h"
#include "xla/service/rng_expander.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/simplify_fp_conversions.h"
#include "xla/service/slice_sinker.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/stable_sort_expander.h"
#include "xla/service/stochastic_convert_decomposer.h"
#include "xla/service/sub_byte_normalization.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_all_reduce_code_motion.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/while_loop_trip_count_annotator.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/traceme.h"

#ifdef PLATFORM_GOOGLE
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
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
  return AutotuneConfig{
      DevicelessConfig{gpu_target_config.device_description_str},
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
             std::string_view asm_text, absl::Span<const uint8_t> binary,
             const Thunk::BinaryMap& dnn_compiled_graphs) {
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
  auto ir_emitter = IrEmitterUnnested::Create(&ir_emitter_context);
  TF_RETURN_IF_ERROR(
      ir_emitter->EmitHloComputation(hlo_module->entry_computation()));
  std::unique_ptr<ThunkSequence> thunk_sequence =
      ir_emitter->ConsumeThunkSequence();

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
          Thunk::BinaryMap(proto_.dnn_compiled_graphs().cbegin(),
                           proto_.dnn_compiled_graphs().cend()),
          /*gpu_version=*/gpu_device_info.gpu_compute_capability(),
          /*executable=*/std::move(thunk_sequence),
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
void AddHloVerifier(HloPassPipeline* pipeline, HloVerifierOpts&& opts = {},
                    bool debug_only = false) {
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
      GpuConvRewriter::ConvIsLowerable);
  layout_insensitive_algsimp_opts.set_enable_dot_strength_reduction(
      hlo_module_config.debug_options()
          .xla_gpu_enable_dot_strength_reduction());

  // GPU only supports canonical convolutions.
  layout_insensitive_algsimp_opts.set_supports_non_canonical_dots(false);

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
  if (num_partitions > 1) {
    if (!hlo_module->config().use_spmd_partitioning()) {
      return InvalidArgument(
          "num_partitions=%d but SPMD partitioning not enabled.",
          num_partitions);
    }
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    AddSPMDPasses(
        hlo_module, layout_insensitive_algsimp_opts,
        gpu_target_config.device_description.gpu_compute_capability(),
        spmd_pipeline,
#ifdef PLATFORM_GOOGLE
        [&](HloPassPipeline& pipeline) {
          if (auto_sharding) {
            AutoShardingOption option;
            option.enable = true;
            if (!hlo_module->config()
                     .auto_spmd_partitioning_mesh_shape()
                     .empty()) {
              option.device_mesh_shape =
                  hlo_module->config().auto_spmd_partitioning_mesh_shape();
            } else {
              // Use a simple mesh shape if not specified.
              option.device_mesh_shape = {
                  gpu_target_config.device_description.core_count(), 1};
            }
            if (!hlo_module->config()
                     .auto_spmd_partitioning_mesh_ids()
                     .empty()) {
              option.device_mesh_ids =
                  hlo_module->config().auto_spmd_partitioning_mesh_ids();
            }
            option.memory_budget_per_device =
                hlo_module->config()
                    .debug_options()
                    .xla_gpu_auto_spmd_partitioning_memory_budget_gb() *
                1024 * 1024 * 1024;
            option.memory_budget_ratio =
                hlo_module->config()
                    .debug_options()
                    .xla_gpu_auto_spmd_partitioning_memory_budget_ratio();
            spmd_pipeline.AddPass<AutoSharding>(option);
          }
        });
#else
        std::nullopt);
#endif  // PLATFORM_GOOGLE
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
  AddHloVerifier(&pipeline);
  if (debug_options.xla_gpu_multi_streamed_windowed_einsum()) {
    pipeline.AddPass<GpuWindowedEinsumHandler>();
  }
  pipeline.AddPass<TopKSplitter>();
  pipeline.AddPass<TopkSpecializer>();
  pipeline.AddPass<TopkDecomposer>();

  HloPredicate upcaster_filter = [&](const HloInstruction* instr) {
    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
        &gpu_target_config.device_description.gpu_compute_capability());
    if (cuda_cc != nullptr &&
        !cuda_cc->IsAtLeast(se::CudaComputeCapability::VOLTA)) {
      return true;
    }
    return !gpu::IsMatrixMultiplication(*instr);
  };
  pipeline.AddPass<DotDimensionSorter>();
  pipeline.AddPass<DotDecomposer>();

  pipeline.AddPass<OperandUpcaster>(upcaster_filter);
  pipeline.AddPass<ResultCaster>(upcaster_filter);

  // Add the DotOperandConverter after any potential upcasts done as part of
  // the OperandUpcaster, so that the DotOperandConverter becomes a no-op.
  pipeline.AddPass<DotOperandConverter>();

  pipeline.AddPass<SubByteNormalization>(
      SubByteNormalization::SET_ELEMENT_SIZE);

  // Expand random number generation.
  pipeline.AddPass<RngExpander>();
  pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);

  // Comparison total order expander
  pipeline.AddPass<ComparisonExpander>(std::array{std::make_pair(BF16, F32)});

  // Remove zero-sized HLO from the input so that other passes don't have to
  // handle it.
  pipeline.AddPass<ZeroSizedHloElimination>();

  if (debug_options.xla_gpu_deterministic_ops() ||
      debug_options.xla_gpu_exclude_nondeterministic_ops()) {
    // Scatter can be indeterministic if indices are not unique or a non
    // associative combiner function is used. Eliminate these Scatter ops.
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

  // TODO(b/64094172): make Call work on GPU instead of inlining.
  pipeline.AddPass<CallInliner>();

  pipeline.AddPass<StochasticConvertDecomposer>();

  pipeline.AddPass<Convolution4DExpander>();

  // Replace PRED convolutions with F16.
  pipeline.AddPass<ConvolutionPredExpander>();

  // Expand the sort op to support stable sorting if required.
  pipeline.AddPass<StableSortExpander>();

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
  se::GpuComputeCapability gpu_version =
      gpu_target_config.device_description.gpu_compute_capability();

  // Build simplification pipeline.  The passes in here are run to a fixed
  // point.
  [&, &pipeline =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
    AddHloVerifier(&pipeline, HloVerifierOpts{}, /*debug_only=*/true);

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
    // AlgebraicSimplifier may add contracting dimensions to a dot.
    pipeline.AddPass<DotDimensionSorter>();
    pipeline.AddPass<DotDecomposer>();
    // Only merge "smallish" dots.  This threshold was not set carefully, but
    // so far we know that 1mb is too small.
    pipeline.AddPass<DotMerger>(/*max_size_to_merge=*/int64_t{32} << 20);
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
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  HloPassPipeline collectives_pipeline("collective-optimizations");
  collectives_pipeline.AddPass<AllReduceFolder>();
  if (debug_options.xla_gpu_enable_all_reduce_splitter()) {
    collectives_pipeline.AddPass<AllReduceSplitter>();
  }
  collectives_pipeline.AddPass<ReduceScatterCreator>();
  collectives_pipeline.AddPass<AllGatherOptimizer>();
  collectives_pipeline.AddPass<AllReduceReassociate>(
      debug_options.xla_gpu_enable_reassociation_for_converted_ar());
  collectives_pipeline.AddPass<ReduceScatterReassociate>();

  collectives_pipeline.AddPass<WhileLoopAllReduceCodeMotion>(
      /*enable_reduce_scatter=*/debug_options
          .xla_gpu_enable_while_loop_reduce_scatter_code_motion());

  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_all_reduce()) {
    CollectivePipeliner::Config config{
        /*level_to_operate_on=*/0,
        /*max_pipelining_per_loop=*/INT64_MAX,
        /*last_run=*/true,
        /*pipeline_use_tree=*/false,
        /*process_different_sized_ops=*/true,
        /*pipelining_direction=*/
        CollectivePipeliner::PipeliningDirection::kForward,
        /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>,
        /*acceptable_formatting=*/HloPredicateTrue,
        /*reuse_pipelined_op_buffer=*/HloPredicateFalse};
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }
  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_all_gather()) {
    // TODO(b/346702380): This constraint relaxation breaks some near-optimal
    // schedules for async LHS. This is just the mitigation, the proper fix is
    // to add a heuristic to the LHS scheduler which would prefer paths with
    // more costly collectives first.
    bool acceptable_loop_invariant_op_in_chain =
        !debug_options.xla_gpu_enable_approx_costly_collectives();
    CollectivePipeliner::Config config{
        /*level_to_operate_on=*/0,
        /*max_pipelining_per_loop=*/INT64_MAX,
        /*last_run=*/true,
        /*pipeline_use_tree=*/false,
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
        acceptable_loop_invariant_op_in_chain,
    };
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }
  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_reduce_scatter()) {
    CollectivePipeliner::Config config{
        /*level_to_operate_on=*/0,
        /*max_pipelining_per_loop=*/INT64_MAX,
        /*last_run=*/true,
        /*pipeline_use_tree=*/false,
        /*process_different_sized_ops=*/true,
        /*pipelining_direction=*/
        CollectivePipeliner::PipeliningDirection::kForward,
        /*should_process=*/HloPredicateIsOp<HloOpcode::kReduceScatter>,
        /*acceptable_formatting=*/HloPredicateTrue,
        /*reuse_pipelined_op_buffer=*/HloPredicateFalse};
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }

  collectives_pipeline.AddPass<CollectivePermuteCycleDecomposer>(
      hlo_module->config()
          .debug_options()
          .xla_gpu_collective_permute_decomposer_threshold());

  collectives_pipeline.AddPass<CollectivePermuteDecomposer>(
      hlo_module->config()
          .debug_options()
          .xla_gpu_collective_permute_decomposer_threshold());

  if (hlo_module->config()
          .debug_options()
          .xla_gpu_enable_pipelined_collectives() ||
      hlo_module->config().debug_options().xla_gpu_enable_pipelined_p2p()) {
    AddP2PPipeliner(collectives_pipeline);
  }

  // Run algebraic simplifier to reshape(broadcast) into a broadcast when
  // the reshape is just adding a unit dimension. This will help with the
  // AllGatherBroadcastReorder pass.
  collectives_pipeline.AddPass<GpuAlgebraicSimplifier>(
      layout_insensitive_algsimp_opts, gpu_version);

  collectives_pipeline.AddPass<AllGatherBroadcastReorder>();

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

absl::Status RunLayoutAssignmentPasses(HloModule* hlo_module,
                                       se::GpuComputeCapability gpu_version,
                                       se::dnn::VersionInfo dnn_version) {
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
      &layout_constraints);
  // Run SubByteNormalization because GpuLayoutAssignment may modify a
  // Layout's element_size_in_bits field.
  pipeline.AddPass<SubByteNormalization>(
      SubByteNormalization::SET_ELEMENT_SIZE);
  pipeline.AddPass<OptimizeInputOutputBufferAlias>(true);
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

  if (hlo_module->config().debug_options().xla_gpu_collect_cost_model_stats()) {
    GpuHloCostAnalysis::Options cost_analysis_options{
        shape_size_fn,
        /*per_second_rates=*/{},
        /*count_multiple_input_accesses=*/true};

    HloPassPipeline post_fusion_analysis("post_fusion_analysis");
    post_fusion_analysis.AddPass<GpuCostModelStatsCollection>(
        gpu_device_info, cost_analysis_options);
    TF_RETURN_IF_ERROR(post_fusion_analysis.Run(hlo_module).status());
  }

  TF_RETURN_IF_ERROR(
      HorizontalFusionPipeline(gpu_device_info).Run(hlo_module).status());

  if (VLOG_IS_ON(2)) {
    HloFusionStatsVisitor stats;
    TF_RETURN_IF_ERROR(hlo_module->entry_computation()->Accept(&stats));
    VLOG(2) << stats.ToString();
  }

  return absl::OkStatus();
}

absl::Status RunPostFusionPasses(
    HloModule* hlo_module,
    std::function<absl::Status(HloPassPipeline*, const DebugOptions&)>
        add_custom_kernel_replacement_passes) {
  const DebugOptions& opts = hlo_module->config().debug_options();

  HloPassPipeline pipeline("post-fusion optimization");
  pipeline.AddPass<RenameFusions>();
  pipeline.AddPass<AllGatherCombiner>(
      opts.xla_gpu_all_gather_combine_threshold_bytes(),
      /*combine_threshold_count=*/256,
      opts.xla_gpu_enable_all_gather_combine_by_dim());
  pipeline.AddPass<AllReduceCombiner>(
      opts.xla_gpu_all_reduce_combine_threshold_bytes(),
      /*combine_threshold_count=*/256);
  pipeline.AddPass<ReduceScatterCombiner>(
      opts.xla_gpu_reduce_scatter_combine_threshold_bytes(),
      /*combine_threshold_count=*/256,
      opts.xla_gpu_enable_reduce_scatter_combine_by_dim());

  pipeline.AddPass<AllReduceContiguous>();

  TF_RETURN_IF_ERROR(add_custom_kernel_replacement_passes(&pipeline, opts));

  int32_t blueconnect_num_devices_per_host =
      hlo_module->config()
          .debug_options()
          .xla_gpu_all_reduce_blueconnect_num_devices_per_host();
  if (blueconnect_num_devices_per_host > 0) {
    pipeline.AddPass<AllReduceBlueConnect>(blueconnect_num_devices_per_host);
  }

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
  if (unroll_strategy != std::nullopt) {
    pipeline.AddPass<DoubleBufferLoopUnrolling>(*unroll_strategy);
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<HloDCE>();
  }

  return pipeline.Run(hlo_module).status();
}

absl::Status RunPostFusionCollectiveOptimizationPasses(HloModule* hlo_module) {
  HloPassPipeline pipeline("post-fusion-collectives optimization");

  // Convert all collectives to their async form, and then annotate the ones
  // that actually need to run asynchronously with a GPU specific backend
  // config.
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_reduce = HloPredicateTrue;
  config.convert_collective_broadcast = HloPredicateTrue;
  config.convert_collective_permute = HloPredicateTrue;
  config.convert_all_gather = HloPredicateTrue;
  config.convert_reduce_scatter = HloPredicateTrue;
  config.convert_all_to_all = HloPredicateTrue;
  pipeline.AddPass<AsyncCollectiveCreator>(std::move(config));

  absl::flat_hash_set<DebugOptions::CollectiveOpType> disabled_async_ops;
  for (auto collective_op_type : hlo_module->config()
                                     .debug_options()
                                     .xla_gpu_disable_async_collectives()) {
    disabled_async_ops.insert(
        static_cast<DebugOptions::CollectiveOpType>(collective_op_type));
  }
  auto convert_to_async = [&disabled_async_ops](const HloInstruction* inst) {
    switch (inst->opcode()) {
      case HloOpcode::kAllReduceStart:
        return !disabled_async_ops.contains(DebugOptions::ALLREDUCE);
      case HloOpcode::kCollectivePermuteStart:
        return !disabled_async_ops.contains(DebugOptions::COLLECTIVEPERMUTE);
      case HloOpcode::kAllGatherStart:
        return !disabled_async_ops.contains(DebugOptions::ALLGATHER);
      case HloOpcode::kAsyncStart: {
        auto async_inst = Cast<HloAsyncInstruction>(inst);
        switch (async_inst->async_wrapped_opcode()) {
          case HloOpcode::kCollectiveBroadcast:
            return !disabled_async_ops.contains(
                DebugOptions::COLLECTIVEBROADCAST);
          case HloOpcode::kReduceScatter:
            return !disabled_async_ops.contains(DebugOptions::REDUCESCATTER);
          case HloOpcode::kAllToAll:
            return !disabled_async_ops.contains(DebugOptions::ALLTOALL);
          default:
            return false;
        }
      }
      default:
        return false;
    }
  };
  pipeline.AddPass<GpuAsyncCollectiveAnnotator>(convert_to_async);

  return pipeline.Run(hlo_module).status();
}

absl::Status RunPostFusionSimplificationPasses(
    HloModule* hlo_module,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    se::GpuComputeCapability gpu_version) {
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
    pipeline.AddPass<StreamAttributeAnnotator>();
    pipeline.AddPass<StreamAttributeAsyncWrapper>();
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

}  // namespace

// Runs optimization passes on the given HLO module.
absl::Status GpuCompiler::OptimizeHloModule(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config) {
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
      gpu_target_config.device_description.gpu_compute_capability();
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
      hlo_module, gpu_version, dnn_version, options.device_allocator));

  TF_RETURN_IF_ERROR(
      RunLayoutAssignmentPasses(hlo_module, gpu_version, dnn_version));

  // TODO(b/328264715): Add tests to ensure that layout normalization pass is
  // run before any fusion pass.
  HloPassPipeline layout_normalization_pipeline("layout normalization");
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  const AlgebraicSimplifierOptions simplifier_options = [&] {
    AlgebraicSimplifierOptions opts =
        GetAlgebraicSimplifierOptions(hlo_module->config());
    opts.set_supports_non_canonical_dots(false);
    opts.set_is_layout_sensitive(true);
    opts.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    opts.set_minmax_propagate_nan(!debug_options.xla_gpu_enable_fast_min_max());
    opts.set_enable_unconditional_reduce_of_concat_replacement(false);
    return opts;
  }();
  layout_normalization_pipeline.AddPass<ReshapeDecomposer>();
  layout_normalization_pipeline.AddPass<HloPassFix<MoveCopyToUsers>>();
  layout_normalization_pipeline.AddPass<LayoutNormalization>(
      &NormalizeLayoutForGpuCustomCalls);
  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  layout_normalization_pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(
      simplifier_options, gpu_version);
  // Layout normalization will create broadcasts that are not canonical.
  layout_normalization_pipeline.AddPass<BroadcastCanonicalizer>();
  // Layout normalization will create scatters that are not simplified and
  // also have unsorted update_window_dims.
  layout_normalization_pipeline.AddPass<ScatterSimplifier>();
  TF_RETURN_IF_ERROR(layout_normalization_pipeline.Run(hlo_module).status());
  // Run target-specific HLO optimization passes after layout assignment.
  TF_RETURN_IF_ERROR(OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config,
      thread_pool.get_mutable()));

  // This is a "low effort, high impact" fusion that should be run first.
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_enable_address_computation_fusion()) {
    HloPassPipeline pipeline("dynamic-slice");
    TF_ASSIGN_OR_RETURN(se::Platform * platform,
                        se::PlatformManager::PlatformWithId(PlatformId()));
    pipeline.AddPass<DynamicSliceFusionRewriter>(platform->Name());
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  TF_RETURN_IF_ERROR(RunFusionPasses(hlo_module, gpu_target_config,
                                     thread_pool.get_mutable(),
                                     ShapeSizeBytesFunction()));
  TF_RETURN_IF_ERROR(RunPostFusionPasses(
      hlo_module,
      [this](HloPassPipeline* pipeline, const DebugOptions& debug_options) {
        return AddCustomKernelReplacementPasses(pipeline, debug_options);
      }));
  TF_RETURN_IF_ERROR(RunPostFusionCollectiveOptimizationPasses(hlo_module));
  TF_RETURN_IF_ERROR(RunPostFusionSimplificationPasses(
      hlo_module, layout_insensitive_algsimp_opts, gpu_version));

  TF_RETURN_IF_ERROR(RunPostFusionVerificationPasses(
      hlo_module, stream_exec, options, gpu_target_config));

  return RunPreSchedulingPasses(hlo_module, stream_exec);
}  // NOLINT(readability/fn_size)

AlgebraicSimplifierOptions GpuCompiler::GetAlgebraicSimplifierOptions(
    const HloModuleConfig& config) {
  AlgebraicSimplifierOptions opts;
  opts.set_enable_dot_strength_reduction(
      config.debug_options().xla_gpu_enable_dot_strength_reduction());
  return opts;
}

// Modifies the given HLO module so that it will be accepted by IrEmitter.
// Unlike optimization passes, the passes are necessary for correctness.
absl::Status GpuCompiler::PrepareHloModuleForIrEmitting(HloModule* hlo_module) {
  return PrepareHloModuleForIrEmittingPipeline(*hlo_module, GetCanShareBuffer())
      .Run(hlo_module)
      .status();
}

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
  const GpuFloatSupport f8e4m3fn_support(gpu_version, F8E4M3FN, F16);
  const FloatSupport f8e4m3b11fnuz_support(F8E4M3B11FNUZ, F16);
  const GpuFloatSupport f8e5m2fnuz_support(gpu_version, F8E5M2FNUZ, F16);
  const GpuFloatSupport f8e4m3fnuz_support(gpu_version, F8E4M3FNUZ, F16);
  auto add_float_normalization = [&](HloPassPipeline& pipeline) {
    auto& sub_pipeline =
        pipeline.AddPass<HloPassPipeline>("float_normalization");
    sub_pipeline.AddPass<FloatNormalization>(&bf16_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e5m2_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3fn_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3b11fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e5m2fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3fnuz_support);
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
      return IsReductionFromOrToContiguousDimensions(*r);
    });

    // Greedy pattern matching for custom kernel fusions. We run it before
    // Triton rewriter or a regular Gemm rewriter to be able to match compatible
    // GEMMs before they matched into Triton gemm or a cuBLAS custom call.
    //
    // TODO(ezhulenev): This should be plugged into the cost model and fusion
    // heuristic, so we can mix and match various Gemm implementations based
    // on projected (measured) performance.
    if (debug_options.xla_gpu_enable_custom_fusions()) {
      pipeline.AddPass<CustomKernelFusionRewriter>(
          &gpu_target_config.device_description);
    }

    // Rewrite GEMMs into custom calls.
    se::GpuComputeCapability gpu_version =
        gpu_target_config.device_description.gpu_compute_capability();
    pipeline.AddPass<AlgorithmChecker>(gpu_version);
    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version);

    // Rewrite FP8 GEMMs ahead of Triton which currently lacks support for FP8
    // and may rewrite quantized FP8 GEMMs as higher-precision GEMMs.
    pipeline.AddPass<GemmRewriter>(gpu_version, GetToolkitVersion(),
                                   /*f8_rewrite=*/true);
    if (debug_options.xla_gpu_enable_triton_gemm() && cuda_cc != nullptr &&
        cuda_cc->IsAtLeast(se::CudaComputeCapability::AMPERE)) {
      pipeline.AddPass<GemvRewriter>();
      pipeline.AddPass<GemmFusion>(gpu_version);
    }
    // Rewrite non-FP8 GEMMs.
    pipeline.AddPass<GemmRewriter>(gpu_version, GetToolkitVersion(),
                                   /*f8_rewrite=*/false);

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

    pipeline.AddPass<ReductionDegenerateDimRemover>();
    pipeline.AddPass<ReductionLayoutNormalizer>();
    // Run Softmax fusion after layout normalization. We expect a default layout
    // in the softmax codegen pipeline. However we should run before
    // ReductionDimensionGrouper, as that makes matching the softmax pattern
    // harder.
    if (debug_options.xla_gpu_enable_triton_softmax_fusion() &&
        cuda_cc != nullptr &&
        cuda_cc->IsAtLeast(se::CudaComputeCapability::AMPERE)) {
      pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                           gpu_version);
      pipeline.AddPass<SoftmaxRewriterTriton>(gpu_version);
    }

    pipeline.AddPass<ReductionDimensionGrouper>();
    // Do not split small reduction dimensions unless priority fusion is
    // enabled, which handles such cases well.
    bool ignore_small_reduce_dims =
        !debug_options.xla_gpu_enable_priority_fusion();
    pipeline.AddPass<HloPassFix<ReductionSplitter>>(ignore_small_reduce_dims);
    pipeline.AddPass<HloPassFix<GpuTreeReductionRewriter>>(gpu_version);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  HloPassPipeline pipeline("post-layout_assignment");
  AddHloVerifier(&pipeline,
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

  TF_RETURN_IF_ERROR(AddGemmFusionAutotuningPasses(&pipeline, hlo_module,
                                                   autotune_config, thread_pool,
                                                   options.key_value_store));
  // Inline back the calls which have better performance with cuBLAS.
  pipeline.AddPass<CallInliner>();
  // TODO(tdanyluk): Apply CublasPadForGemms to the cuBLAS GEMMs generated
  // here for possibly better cuBLAS performance.
  pipeline.AddPass<GemmRewriter>(gpu_version, GetToolkitVersion());
  // Rewrite GEMMs with broadcasted inputs as strided GEMMs.
  pipeline.AddPass<GemmBroadcastFoldingRewriter>();

  pipeline.AddPass<HostOffloadLegalize>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost),
      /* after_layout= */ true);
  pipeline.AddPass<HostOffloader>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost));

  TF_RETURN_IF_ERROR(AddConvAndGemmAutotuningPasses(
      &pipeline, hlo_module, autotune_config, thread_pool));

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

  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);

  pipeline.AddPass<HostMemoryTransferAsyncifier>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost));

#ifdef NDEBUG
  // Verify the module in non-debug builds. For debug builds, the verifier
  // already runs after every pass.
  pipeline.AddPass<HloVerifier>(
      std::make_unique<DefaultVerifierMetadata>(
          HloVerifierOpts{}
              .MakeLayoutSensitive()
              .WithInstructionCanChangeLayout(
                  LayoutAssignment::InstructionCanChangeLayout)
              .VerifyBroadcastDimensionsOrder()
              .VerifyReshapeIsBitcast()),
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

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

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
GpuCompiler::CompileSingleModule(const HloModuleConfig& module_config,
                                 se::GpuComputeCapability gpu_version,
                                 const HloModule* debug_module,
                                 llvm::Module* llvm_module, bool relocatable,
                                 const CompileOptions& options,
                                 std::optional<int> shard_number) {
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
      CompileTargetBinary(module_config, llvm_module, gpu_version, relocatable,
                          debug_module, options));

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

  // Write PTX to IR dump directory, if IR dumping was requested.
  if (should_dump) {
    absl::string_view ptx = result.asm_text;
    if (debug_module) {
      DumpToFileInDirOrStdout(*debug_module, "",
                              shard_number.has_value()
                                  ? (std::to_string(*shard_number) + ".ptx")
                                  : "ptx",
                              ptx);
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
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
    se::GpuComputeCapability gpu_version, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const HloModule* debug_module) {
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
      /*PreserveLocals=*/true);
  VLOG(2) << "Single-function cacheable modules: "
          << single_function_module_count << " / " << llvm_modules.size();

  struct NamedCompileResult {
    // Single function name or empty just like for llvm_modules.
    std::string name;
    absl::StatusOr<BackendCompileResult> result;
  };
  std::vector<NamedCompileResult> compile_results(llvm_modules.size());
  if (thread_pool.get() != nullptr) {
    tsl::BlockingCounter counter(llvm_modules.size());
    for (int i = 0; i < llvm_modules.size(); ++i) {
      thread_pool.get_mutable()->Schedule(
          [&compile_results, i, &llvm_modules, &counter, this, &module_config,
           &gpu_version, &debug_module, &options] {
            // Each thread has its own context to avoid race conditions.
            llvm::LLVMContext new_context;
            std::unique_ptr<llvm::Module> new_module =
                CopyToContext(*llvm_modules.at(i).module, new_context);
            compile_results.at(i) = {
                llvm_modules.at(i).name,
                CompileSingleModule(module_config, gpu_version, debug_module,
                                    new_module.get(),
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
          CompileSingleModule(module_config, gpu_version, debug_module,
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

  auto maybe_backend_result = LinkModules(
      stream_exec, std::move(binaries_to_link), module_config.debug_options());
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
                        CanUseLinkModules(module->config()));
  }
  const bool split_modules =
      can_use_link_modules &&
      module->config()
          .debug_options()
          .xla_gpu_enable_llvm_module_compilation_parallelism();
  const bool use_cache =
      split_modules &&
      !module->config().debug_options().xla_gpu_kernel_cache_file().empty();

  // Compile the module
  TF_ASSIGN_OR_RETURN(
      CompileModuleResults compile_module_results,
      CompileModuleToLlvmIr(module, llvm_context, target_triple_, data_layout_,
                            platform->Name(), platform->id(), gpu_device_info,
                            GetCanShareBuffer(), BufferSizeBytesFunction(),
                            /*split_constants_module=*/use_cache));

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
    TF_ASSIGN_OR_RETURN(backend_result,
                        CompileAndLink(module->config(), compile_module_results,
                                       gpu_device_info.gpu_compute_capability(),
                                       executor, options, module));
  } else {
    CHECK(compile_module_results.llvm_module_constants == nullptr);
    TF_ASSIGN_OR_RETURN(
        backend_result,
        CompileSingleModule(module->config(),
                            gpu_device_info.gpu_compute_capability(), module,
                            &*compile_module_results.llvm_module,
                            /*relocatable=*/false, options,
                            /*shard_number=*/std::nullopt));
  }
  RecordXlaDeviceBinarySize(backend_result.binary.size());
  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(*module, "", "thunk_sequence.txt",
                            compile_module_results.executable->ToString());
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
  Thunk::BinaryMap dnn_compiled_graphs;
  if (stream_exec) {
    TF_RETURN_IF_ERROR(RunCudnnFusionCompilerPass(module.get(), stream_exec,
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
    GpuHloCostAnalysis cost_analysis(cost_analysis_options, &gpu_device_info);
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
    DumpToFileInDirOrStdout(*module, "", "thunk_sequence.txt",
                            res.compile_module_results.executable->ToString());
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
        gpu_executable->buffer_assignment()->GetStats().ToString());
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
  return [pointer_size = pointer_size_](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
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
    HloModule* module, se::StreamExecutor* stream_exec) {
  HloPassPipeline pipeline("pre-scheduling-passes");
  pipeline.AddPass<CollectivesScheduleLinearizer>(
      [this, stream_exec](const HloModule* module) {
        return RequiresCollectiveScheduleLinearizer(module, stream_exec);
      });
  return pipeline.Run(module).status();
}

absl::Status GpuCompiler::RunPostSchedulingPipelines(
    HloModule* module, int64_t scheduler_mem_limit,
    const se::DeviceDescription& gpu_device_info) const {
  TF_RETURN_IF_ERROR(
      RunPostSchedulingCopyInsertion(module, GetCanShareBuffer()));
  {
    HloPassPipeline pipeline("post-scheduling-passes");

    if (module->config()
            .debug_options()
            .xla_gpu_enable_pipelined_collectives() ||
        module->config().debug_options().xla_gpu_enable_pipelined_p2p()) {
      pipeline.AddPass<PipelinedP2PRewriter>();
    }
    HloPredicate is_nop =
        HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant,
                         HloOpcode::kBitcast, HloOpcode::kGetTupleElement>;
    pipeline.AddPass<GpuConvertAsyncCollectivesToSync>(is_nop);

    TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  }

  {
    HloPassPipeline pipeline("remat-pipeline");

    const bool enable_offloading = module->config()
                                       .debug_options()
                                       .xla_gpu_enable_host_memory_offloading();
    HloRematerialization::RematerializationModeConfig
        rematerialization_mode_config(/*recompute=*/true, /*compress=*/true,
                                      /*host_offload=*/enable_offloading);
    HloCostAnalysis::Options hlo_cost_analysis_options;
    hlo_cost_analysis_options.shape_size = ShapeSizeBytesFunction();
    std::optional<HloRematerialization::HostMemoryOffloadConfig>
        offloading_config = std::nullopt;
    if (enable_offloading) {
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
    HloCostAnalysis hlo_cost_analysis(hlo_cost_analysis_options);
    HloRematerialization::Options options(
        hlo_cost_analysis, rematerialization_mode_config,
        // Assume 75% of the total device memory is available for XLA.
        /*memory_limit_bytes=*/scheduler_mem_limit,
        /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
        /*min_remat_size=*/0, /*compact_shape_function=*/nullptr,
        /*host_memory_offload_config=*/offloading_config);
    HloRematerialization::RematerializationSizes sizes;
    pipeline.AddPass<HloRematerialization>(options, sizes);
    pipeline.AddPass<StreamAttributeAnnotator>();
    pipeline.AddPass<OptimizationBarrierExpander>();

    TF_ASSIGN_OR_RETURN(bool changed, pipeline.Run(module));
    if (changed) {
      VLOG(1) << "HloRematerialization saved "
              << sizes.before_bytes - sizes.after_bytes << " bytes";
    }
  }

  {
    HloPassPipeline pipeline("fusion-wrapper");
    pipeline.AddPass<FusionWrapper>();
    // Wrap remaining unfused ops that have no LHLO equivalent in single-op
    // fusions. This needs to happen after rematerialization, because that
    // will insert additional copies.
    TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  }

  // After we have a scheduled module and all operations wrapped into fusions
  // we can decide how to wrap them into command buffers.
  {
    HloPassPipeline pipeline("command-buffer-scheduling");
    auto driver_version = se::gpu::GpuDriver::GetDriverVersion();
    const int32_t toolkit_version = GetToolkitVersion();
    pipeline.AddPass<CommandBufferScheduling>(
        gpu_device_info, toolkit_version,
        driver_version.value_or(toolkit_version));
    pipeline.AddPass<GpuSanitizeConstantNames>();
    TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  }

  return absl::OkStatus();
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"

#include <stdlib.h>

#include <algorithm>
#include <any>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/transforms/hlo_constant_splitter.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "tensorflow/compiler/xla/mlir_hlo/transforms/gpu_passes.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/all_gather_broadcast_reorder.h"
#include "tensorflow/compiler/xla/service/all_gather_combiner.h"
#include "tensorflow/compiler/xla/service/all_reduce_combiner.h"
#include "tensorflow/compiler/xla/service/all_reduce_contiguous.h"
#include "tensorflow/compiler/xla/service/all_reduce_folder.h"
#include "tensorflow/compiler/xla/service/all_reduce_promotion.h"
#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"
#include "tensorflow/compiler/xla/service/all_to_all_decomposer.h"
#include "tensorflow/compiler/xla/service/async_collective_creator.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"
#include "tensorflow/compiler/xla/service/broadcast_canonicalizer.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/collectives_schedule_linearizer.h"
#include "tensorflow/compiler/xla/service/comparison_expander.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/convert_async_collectives_to_sync.h"
#include "tensorflow/compiler/xla/service/convert_mover.h"
#include "tensorflow/compiler/xla/service/convolution_4d_expander.h"
#include "tensorflow/compiler/xla/service/convolution_pred_expander.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dot_merger.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_simplifier.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/dynamic_padder.h"
#include "tensorflow/compiler/xla/service/eigh_expander.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/float_normalization.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/gather_simplifier.h"
#include "tensorflow/compiler/xla/service/gpu/alias_passthrough_params.h"
#include "tensorflow/compiler/xla/service/gpu/all_reduce_blueconnect.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/conv_layout_normalization.h"
#include "tensorflow/compiler/xla/service/gpu/dot_dimension_sorter.h"
#include "tensorflow/compiler/xla/service/gpu/for_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_broadcast_folding_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_reduce_scatter_creator.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_sanitize_constant_names.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_scatter_expander.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_serializable_autotuner.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_shape_verifier.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_fusion_stats.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_input_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/metrics.h"
#include "tensorflow/compiler/xla/service/gpu/move_copy_to_users.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_dimension_grouper.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_splitter.h"
#include "tensorflow/compiler/xla/service/gpu/runtime_intrinsics.h"
#include "tensorflow/compiler/xla/service/gpu/scatter_slice_simplifier.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/tree_reduction_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/variadic_op_splitter.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_computation_deduplicator.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_rematerialization.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/layout_normalization.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/logistic_expander.h"
#include "tensorflow/compiler/xla/service/loop_schedule_linearizer.h"
#include "tensorflow/compiler/xla/service/operand_upcaster.h"
#include "tensorflow/compiler/xla/service/optimization_barrier_expander.h"
#include "tensorflow/compiler/xla/service/qr_expander.h"
#include "tensorflow/compiler/xla/service/real_imag_expander.h"
#include "tensorflow/compiler/xla/service/reduce_decomposer.h"
#include "tensorflow/compiler/xla/service/reduce_scatter_combiner.h"
#include "tensorflow/compiler/xla/service/reduce_scatter_reassociate.h"
#include "tensorflow/compiler/xla/service/reshape_decomposer.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/result_caster.h"
#include "tensorflow/compiler/xla/service/rng_bit_generator_expander.h"
#include "tensorflow/compiler/xla/service/rng_expander.h"
#include "tensorflow/compiler/xla/service/scatter_simplifier.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/sharding_remover.h"
#include "tensorflow/compiler/xla/service/simplify_fp_conversions.h"
#include "tensorflow/compiler/xla/service/slice_sinker.h"
#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/spmd/collective_permute_motion.h"
#include "tensorflow/compiler/xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/stable_sort_expander.h"
#include "tensorflow/compiler/xla/service/stochastic_convert_decomposer.h"
#include "tensorflow/compiler/xla/service/topk_rewriter.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_all_reduce_code_motion.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_trip_count_annotator.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.pb.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/location_exporter.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla/mhlo_to_lhlo_with_xla.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/blocking_counter.h"
#include "tensorflow/tsl/platform/casts.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/threadpool.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/triton_autotuner.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {
namespace {

class GpuFloatSupport : public FloatSupport {
 public:
  explicit GpuFloatSupport(PrimitiveType low_precision_type)
      : FloatSupport(low_precision_type) {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return FloatSupport::SupportsLowPrecisionOperand(hlo, operand_index) ||
           IsSupported(hlo);
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return FloatSupport::SupportsLowPrecisionOutput(hlo) || IsSupported(hlo);
  }

 private:
  bool IsSupported(const HloInstruction& hlo) const {
    switch (hlo.opcode()) {
      // Collective ops.
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllReduceStart:
      case HloOpcode::kAllReduceDone:
      case HloOpcode::kAllToAll:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kReduceScatter:
      // Handled by Triton GEMM.
      case HloOpcode::kDot:
        return LowPrecisionType() == BF16;
      // Data movement only ops.
      case HloOpcode::kBroadcast:
      case HloOpcode::kConcatenate:
      case HloOpcode::kCopy:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kGather:
      case HloOpcode::kPad:
      case HloOpcode::kReshape:
      case HloOpcode::kReverse:
      case HloOpcode::kScatter:
      case HloOpcode::kSelect:
      case HloOpcode::kSelectAndScatter:
      case HloOpcode::kSlice:
      case HloOpcode::kTranspose:
      // Other special ops.
      case HloOpcode::kBitcast:
        return true;
      default:
        return false;
    }
  }
};

bool ConvIsLowerable(HloInstruction* conv) {
  return GpuConvRewriter::ConvIsLowerable(conv);
}

// CollectivesScheduleLinearizer enforces a total ordering between collectives
// to work around (1) divergence in initial HLOs across executables that are
// communicating with each other using HLO collectives, and (2) divergence in
// executables introduced due to auto tuning, specifically the use of extra
// scratch space for convolutions.
// We always apply this pass when not using SPMD (where initial HLO divergence
// may be possible). This function decided whether to apply this pass when using
// SPMD partitioning. When using SPMD, if convolutions are present in the code
// and we are using "online" autotuning (i.e., not AOT) we need to use the pass,
// else we do not need to enable the pass.
bool RequiresCollectiveScheduleLinearizer(const HloModule* module) {
  for (const HloComputation* comp : module->MakeNonfusionComputations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (GpuConvAlgorithmPicker::IsCandidate(inst)) {
        return true;
      }
    }
  }
  // No convolution auto-tuning candidates found in the module.
  return false;
}
}  // end anonymous namespace

using OwnedThunkSequence = GpuExecutable::OwnedThunkSequence;
using OwnedGpuRuntimeProgram = GpuExecutable::OwnedGpuRuntimeProgram;

StatusOr<std::unique_ptr<Executable>>
GpuXlaRuntimeAotCompilationResult::LoadExecutable(
    Compiler* compiler, se::StreamExecutor* executor) const {
  XlaRuntimeExecutableProto xla_runtime_executable =
      xla_runtime_gpu_executable_.xla_runtime_executable();
  TF_ASSIGN_OR_RETURN(HloModuleConfig hlo_module_config,
                      HloModule::CreateModuleConfigFromProto(
                          xla_runtime_executable.hlo_module_proto(),
                          GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(xla_runtime_executable.hlo_module_proto(),
                                 hlo_module_config));
  auto gpu_compiler = tensorflow::down_cast<GpuCompiler*>(compiler);

  std::vector<GpuExecutable::ConstantInfo> constants;
  for (auto& cst : xla_runtime_gpu_executable_.constants()) {
    GpuExecutable::ConstantInfo constant = {
        cst.symbol_name(),
        {cst.content().begin(), cst.content().end()},
        cst.allocation_index()};
    constants.push_back(std::move(constant));
  }

  return GpuExecutable::LoadFromObjFile(
      std::move(hlo_module), xla_runtime_executable.obj_file(),
      xla_runtime_executable.mlir_module(),
      xla_runtime_gpu_executable_.entry_func_attrs(),
      GetDebugOptionsFromFlags(), xla_runtime_gpu_executable_.gpu_asm_text(),
      xla_runtime_gpu_executable_.gpu_binary(), std::move(constants),
      gpu_compiler->GetGpuVersion(executor), executor);
}

GpuTargetConfig::GpuTargetConfig(const se::GpuTargetConfigProto& proto)
    : gpu_device_info(proto.gpu_device_info()),
      platform_name(proto.platform_name()),
      dnn_version_info(proto.dnn_version_info()) {
  if (proto.has_cuda_compute_capability()) {
    stream_executor::CudaComputeCapability cuda_compute_capability(
        proto.cuda_compute_capability());
    gpu_version = cuda_compute_capability;
  } else {
    CHECK(proto.has_rocm_compute_capability());
    stream_executor::RocmComputeCapability rocm_compute_capability(
        proto.rocm_compute_capability());
    gpu_version = rocm_compute_capability;
  }

  device_description_str = proto.device_description_str();
}

se::GpuTargetConfigProto GpuTargetConfig::ToProto() const {
  se::GpuTargetConfigProto proto;
  *proto.mutable_gpu_device_info() = gpu_device_info.ToProto();

  if (std::holds_alternative<se::CudaComputeCapability>(gpu_version)) {
    auto cuda_compute_capability =
        std::get<se::CudaComputeCapability>(gpu_version);
    *proto.mutable_cuda_compute_capability() =
        cuda_compute_capability.ToProto();
  } else {
    auto rocm_compute_capability =
        std::get<se::RocmComputeCapability>(gpu_version);
    *proto.mutable_rocm_compute_capability() =
        rocm_compute_capability.ToProto();
  }

  proto.set_platform_name(platform_name);
  *proto.mutable_dnn_version_info() = dnn_version_info.ToProto();
  proto.set_device_description_str(device_description_str);
  return proto;
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
      std::make_unique<GpuVerifierMetadata>(std::move(opts));
  if (debug_only) {
    pipeline->AddInvariantCheckerDebug<HloVerifier>(
        std::move(verifier_metadata), "hlo verifier (debug)");
  } else {
    pipeline->AddInvariantChecker<HloVerifier>(std::move(verifier_metadata),
                                               "hlo verifier");
  }
}
}  // namespace

// Runs optimization passes on the given HLO module.
Status GpuCompiler::OptimizeHloModule(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator,
    const GpuTargetConfig& gpu_target_config,
    const AutotuneResults* autotune_results) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts({},
                                                             ConvIsLowerable);

  // GPU only supports canonical convolutions.
  layout_insensitive_algsimp_opts.set_supports_non_canonical_dots(false);

  // "slow" minmax means we propagate nan.
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(
      !debug_options.xla_gpu_enable_fast_min_max());

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

  const int64_t num_partitions = hlo_module->config().num_partitions();
  if (num_partitions > 1) {
    if (!hlo_module->config().use_spmd_partitioning()) {
      return InvalidArgument(
          "num_partitions=%d but SPMD partitioning not enabled.",
          num_partitions);
    }
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    // Run some IR cleanup passes before running the SPMD partitioning
    // passes.
    spmd_pipeline.AddPass<CallInliner>();
    spmd_pipeline.AddPass<ZeroSizedHloElimination>();
    spmd_pipeline.AddPass<ConditionalCanonicalizer>();
    spmd_pipeline.AddPass<TopkRewriter>(
        // We're only rewriting TopK to prevent SPMD partitioning from blowing
        // it up. Always allow it.
        [](const HloSortInstruction*, int64_t) { return true; });

    HloPassPipeline& spmd_simplify =
        spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

    spmd_simplify.AddPass<AlgebraicSimplifier>(layout_insensitive_algsimp_opts);

    spmd_simplify.AddPass<SortSimplifier>();
    spmd_simplify.AddPass<TupleSimplifier>();
    spmd_simplify.AddPass<ScatterSimplifier>();
    spmd_simplify.AddPass<ScatterExpander>(
        ScatterExpander::kEliminateSimpleScatters);
    spmd_simplify.AddPass<GatherSimplifier>();
    spmd_simplify.AddPass<GatherExpander>(
        GatherExpander::kEliminateSimpleGathers);
    spmd_simplify.AddPass<WhileLoopConstantSinking>();
    spmd_simplify.AddPass<WhileLoopSimplifier>();

    spmd_simplify.AddPass<ReshapeMover>();
    spmd_simplify.AddPass<HloConstantFolding>();
    spmd_simplify.AddPass<ConditionalSimplifier>();
    spmd_simplify.AddPass<HloDCE>();

    spmd_pipeline.AddPass<HloConstantSplitter>();
    spmd_pipeline.AddPass<ShardingPropagation>(
        /*is_spmd=*/true, /*propagate_metadata=*/false,
        hlo_module->config().allow_spmd_sharding_propagation_to_output());
    spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
        num_partitions, hlo_module->config().replica_count());
    spmd_pipeline.AddPass<CollectivePermuteMotion>();
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  } else {
    HloPassPipeline sharding_removal_pipeline("sharding-removal");
    // Remove redundant sharding ops when partition_count == 1.
    sharding_removal_pipeline.AddPass<ShardingRemover>();
    sharding_removal_pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(sharding_removal_pipeline.Run(hlo_module).status());
  }

  {
    HloPassPipeline pipeline("optimization");
    AddHloVerifier(&pipeline);
    pipeline.AddPass<TopkDecomposer>();
    pipeline.AddPass<AllToAllDecomposer>();

    HloPredicate upcaster_filter = [&](const HloInstruction* instr) {
      return !stream_exec->GetDeviceDescription()
                  .cuda_compute_capability()
                  .IsAtLeast(se::CudaComputeCapability::VOLTA) ||
             !gpu::IsMatrixMultiplication(*instr);
    };

    pipeline.AddPass<OperandUpcaster>(upcaster_filter);
    pipeline.AddPass<ResultCaster>(upcaster_filter);

    // Expand random number generation.
    pipeline.AddPass<RngExpander>();
    pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);

    // Comparison total order expander
    pipeline.AddPass<ComparisonExpander>();

    // Remove zero-sized HLO from the input so that other passes don't have to
    // handle it.
    pipeline.AddPass<ZeroSizedHloElimination>();

    if (debug_options.xla_gpu_deterministic_ops()) {
      // Scatter can be indeterministic if indices are not unique or a non
      // associative combiner function is used. Eliminate these Scatter ops.
      pipeline.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateIndeterminisitcScatters);
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

    pipeline.AddPass<DotDimensionSorter>();
    pipeline.AddPass<DotDecomposer>();

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

    pipeline.AddPass<LogisticExpander>(
        /*expansion_type=*/LogisticExpansionType::kExp);
    pipeline.AddPass<ConditionalCanonicalizer>();
    pipeline.AddPass<DynamicDimensionSimplifier>();

    DynamicPadderOptions dynamic_padder_options;

    switch (hlo_module->config().debug_options().xla_gpu_shape_checks()) {
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
                  ShapeUtil::MakeTokenShape(), {inst},
                  kXlaGpuAssertCustomCallTag,
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
      pipeline.AddPass<AlgebraicSimplifier>(layout_insensitive_algsimp_opts);
      pipeline.AddPass<BitcastDtypesExpander>();
      // AlgebraicSimplifier may add contracting dimensions to a dot.
      pipeline.AddPass<DotDimensionSorter>();
      pipeline.AddPass<DotDecomposer>();
      // Only merge "smallish" dots.  This threshold was not set carefully, but
      // so far we know that 1mb is too small.
      pipeline.AddPass<DotMerger>(/*max_size_to_merge=*/int64_t{16} << 20);
      pipeline.AddPass<SortSimplifier>();
      pipeline.AddPass<TupleSimplifier>();
      pipeline.AddPass<WhileLoopConstantSinking>();
      pipeline.AddPass<WhileLoopSimplifier>();
      pipeline.AddPass<SliceSinker>();
      pipeline.AddPass<ReshapeMover>();
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
      pipeline.AddPass<AlgebraicSimplifier>(layout_insensitive_algsimp_opts);
    }();

    // Run WhileLoopTripCountAnnotator at the end of the simplification
    // pipeline, before layout assignment and fusion.  This pass does some
    // pattern-matching on while bodies/conditions, and this is where the HLO is
    // "nicest".
    //
    // It's important that we don't make semantic changes (e.g. unrolling) to
    // any `while` loops after this point, because otherwise the trip-count
    // annotations added by this pass may not be correct after the
    // modifications.
    pipeline.AddPass<WhileLoopTripCountAnnotator>();
    pipeline.AddPass<HloComputationDeduplicator>(
        /*mark_fusion_duplications=*/false);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  // Optimize collectives generated by SPMD partitioning. Enable these passes
  // otherwise as well so that all collectives can get these optimizations.
  {
    HloPassPipeline collectives_pipeline("collective-optimizations");
    collectives_pipeline.AddPass<AllReduceFolder>();
    collectives_pipeline.AddPass<WhileLoopAllReduceCodeMotion>();
    collectives_pipeline.AddPass<ReduceScatterCreator>();
    collectives_pipeline.AddPass<AllReduceReassociate>();
    collectives_pipeline.AddPass<ReduceScatterReassociate>();

    // Run algebraic simplifier to reshape(broadcast) into a broadcast when
    // the reshape is just adding a unit dimension. This will help with the
    // AllGatherBroadcastReorder pass.
    collectives_pipeline.AddPass<AlgebraicSimplifier>(
        layout_insensitive_algsimp_opts);

    collectives_pipeline.AddPass<AllGatherBroadcastReorder>();

    // promote 16 bit integer all-reduce and reduce-scatter to 32-bit.
    const std::pair<PrimitiveType, PrimitiveType> ar_promoted_types[] = {
        {U16, U32}, {S16, S32}};
    collectives_pipeline.AddPass<AllReducePromotion>(ar_promoted_types);
    // Remove dead computations left over after ar/rs promotion.
    collectives_pipeline.AddPass<HloDCE>();

    TF_RETURN_IF_ERROR(collectives_pipeline.Run(hlo_module).status());
  }

  // Run target-specific HLO optimization passes for convolution
  // canonicalization.
  GpuVersion gpu_version = gpu_target_config.gpu_version;
  se::dnn::VersionInfo dnn_version = gpu_target_config.dnn_version_info;
  if (stream_exec != nullptr) {
    gpu_version = GetGpuVersion(stream_exec);
    se::dnn::DnnSupport* dnn = stream_exec->AsDnn();
    if (dnn == nullptr) {
      return tsl::errors::FailedPrecondition(
          "DNN library initialization failed."
          " Look at the errors above for more details.");
    }
    TF_ASSIGN_OR_RETURN(dnn_version, dnn->GetVersion());
  }

  TF_RETURN_IF_ERROR(OptimizeHloConvolutionCanonicalization(
      hlo_module, gpu_version, dnn_version, device_allocator));

  {
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
        hlo_module->mutable_entry_computation_layout(), stream_exec,
        &layout_constraints);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  // Run target-specific HLO optimization passes after layout assignment.
  TF_RETURN_IF_ERROR(
      OptimizeHloPostLayoutAssignment(hlo_module, stream_exec, device_allocator,
                                      gpu_target_config, autotune_results));

  const GpuDeviceInfo& gpu_device_info = gpu_target_config.gpu_device_info;

  {
    HloPassFix<HloPassPipeline> fusion("fusion");
    // We try to split variadic ops with many parameters into several such ops
    // to avoid exceeding the parameter space.
    fusion.AddPass<VariadicOpSplitter>();
    AddHloVerifier(
        &fusion,
        HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
            LayoutAssignment::InstructionCanChangeLayout),
        /*debug_only=*/true);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/false,
                                         gpu_device_info);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true,
                                         gpu_device_info);
    fusion.AddPass<FusionMerger>(gpu_device_info, ShapeSizeBytesFunction());
    fusion.AddPass<GpuMultiOutputFusion>(gpu_device_info,
                                         ShapeSizeBytesFunction());
    fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                           /*only_fusion_computations=*/true);
    fusion.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(fusion.Run(hlo_module).status());
  }

  {
    HloPassFix<HloPassPipeline> horizontal_fusion("horizontal fusion");
    horizontal_fusion.AddPass<GpuHorizontalLoopFusion>();
    horizontal_fusion.AddPass<GpuHorizontalInputFusion>(gpu_device_info);
    horizontal_fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                                      /*only_fusion_computations=*/true);
    horizontal_fusion.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(horizontal_fusion.Run(hlo_module).status());
  }

  if (VLOG_IS_ON(2)) {
    HloFusionStatsVisitor stats;
    TF_RETURN_IF_ERROR(hlo_module->entry_computation()->Accept(&stats));
    VLOG(2) << stats.ToString();
  }

  {
    HloPassPipeline pipeline("post-fusion optimization");
    pipeline.AddPass<AllGatherCombiner>(
        /*combine_threshold_in_bytes=*/1024 * 1024 * 1024,
        /*combine_threshold_count=*/256);
    pipeline.AddPass<AllReduceCombiner>(
        debug_options.xla_gpu_all_reduce_combine_threshold_bytes(),
        /*combine_threshold_count=*/256);
    pipeline.AddPass<ReduceScatterCombiner>(
        /*combine_threshold_in_bytes=*/30 * 1024 * 1024,
        /*combine_threshold_count=*/256);

    if (debug_options.xla_gpu_all_reduce_contiguous()) {
      pipeline.AddPass<AllReduceContiguous>();
    }

    int32_t blueconnect_num_devices_per_host =
        debug_options.xla_gpu_all_reduce_blueconnect_num_devices_per_host();
    if (blueconnect_num_devices_per_host > 0) {
      pipeline.AddPass<AllReduceBlueConnect>(blueconnect_num_devices_per_host);
    }

    {
      bool async_all_reduce = debug_options.xla_gpu_enable_async_all_reduce();
      bool async_collective_permute =
          debug_options.xla_gpu_enable_async_collective_permute();
      bool async_all_gather = debug_options.xla_gpu_enable_async_all_gather();
      bool async_reduce_scatter =
          debug_options.xla_gpu_enable_async_reduce_scatter();
      bool async_all_to_all = debug_options.xla_gpu_enable_async_all_to_all();

      if (async_all_reduce || async_collective_permute || async_all_gather ||
          async_reduce_scatter || async_all_to_all) {
        AsyncCollectiveCreator::CollectiveCreatorConfig config;
        auto convert_op = [](bool flag) {
          return [=](const HloInstruction*) { return flag; };
        };
        config.convert_all_reduce = convert_op(async_all_reduce);
        config.convert_collective_permute =
            convert_op(async_collective_permute);
        config.convert_all_gather = convert_op(async_all_gather);
        config.convert_reduce_scatter = convert_op(async_reduce_scatter);
        config.convert_all_to_all = convert_op(async_all_to_all);
        pipeline.AddPass<AsyncCollectiveCreator>(std::move(config));
      }
    }

    if (!hlo_module->config().use_spmd_partitioning()) {
      pipeline.AddPass<CollectivesScheduleLinearizer>();
    }

    AlgebraicSimplifierOptions options = layout_insensitive_algsimp_opts;
    options.set_is_layout_sensitive(true);
    pipeline.AddPass<AlgebraicSimplifier>(options);

    // This invocation is used to populate deduplicated_name for fusions that
    // are considered duplicates according to the comparator in this pass.
    // Currently, the pass doesn't actually deduplicate the fusions.
    pipeline.AddPass<HloComputationDeduplicator>(
        /*mark_fusion_duplications=*/true);

    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  return OkStatus();
}

// Modifies the given HLO module so that it will be accepted by IrEmitter.
// Unlike optimization passes, the passes are necessary for correctness.
Status GpuCompiler::PrepareHloModuleForIrEmitting(HloModule* hlo_module) {
  // In some cases, we have to place the result of an instruction in a temporary
  // buffer. For instance, the buffer that holds an external parameter is
  // assumed immutable at this point, and should not be reused for output
  // (b/27180329). Therefore, in that case, we set the output to be a copy of
  // the parameter.
  HloPassPipeline pipeline("GPU-ir-emit-prepare");
  AddHloVerifier(
      &pipeline,
      HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout),
      /*debug_only=*/true);

  // Copy insertion should be performed immediately before IR emission to avoid
  // inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes an
  // instruction which materializes a value). DCE must be run immediately before
  // (and sometime after) copy insertion, to avoid dead code from interfering
  // with the rewrites.
  pipeline.AddPass<HloDCE>();
  if (hlo_module->config().alias_passthrough_params()) {
    pipeline.AddPass<AliasPassthroughParams>();
  }
  pipeline.AddPass<LoopScheduleLinearizer>(GetCanShareBuffer());
  pipeline.AddPass<CopyInsertion>(GetCanShareBuffer());
  // We are using a sub-pipeline here, so that the verifier only runs after both
  // GpuHorizontalLoopFusion and HloDCE.
  auto& sub_pipeline =
      pipeline.AddPass<HloPassPipeline>("horizontal-loop-fusion-for-copy");
  // To fuse the copy.
  sub_pipeline.AddPass<GpuHorizontalLoopFusion>("copy_");
  sub_pipeline.AddPass<HloDCE>();
  pipeline.AddPass<GpuSanitizeConstantNames>();
  return pipeline.Run(hlo_module).status();
}

Status GpuCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator,
    const GpuTargetConfig& gpu_target_config,
    const AutotuneResults* autotune_results) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  {
    HloPassPipeline pipeline("hlo normalization");

    // The LayoutAssignment pass may leave behind kCopy instructions which are
    // duplicate or NOPs, so remove them with algebraic simplification and CSE.
    AlgebraicSimplifierOptions options;
    options.set_supports_non_canonical_dots(false);
    options.set_is_layout_sensitive(true);
    options.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    options.set_minmax_propagate_nan(
        !debug_options.xla_gpu_enable_fast_min_max());
    pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

    // GemmRewriter assumes that all transposes are folded into gemms, but,
    // since commit 7d529df, this is not always true at this point.
    // Therefore, rerun transpose folding.
    pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot,
                                       TransposeFolding::NeverFoldTranspose);

    pipeline.AddPass<ReshapeDecomposer>();
    pipeline.AddPass<ReduceDecomposer>([&](const HloInstruction* r) {
      return IsReductionFromOrToContiguousDimensions(*r);
    });
    pipeline.AddPass<HloPassFix<MoveCopyToUsers>>();

    const stream_executor::CudaComputeCapability& compute_capability =
        std::get<se::CudaComputeCapability>(gpu_target_config.gpu_version);

    // Rewrite GEMMs into custom calls.
    if (debug_options.xla_gpu_enable_triton_gemm() &&
        compute_capability.IsAtLeast(se::CudaComputeCapability::VOLTA)) {
      pipeline.AddPass<GemmRewriterTriton>(compute_capability);
    }
    pipeline.AddPass<GemmRewriter>(compute_capability);

    // Rewrite GEMMs with broadcasted inputs as strided GEMMs.
    pipeline.AddPass<GemmBroadcastFoldingRewriter>();

    if (debug_options.xla_gpu_normalize_layouts()) {
      pipeline.AddPass<LayoutNormalization>(&NormalizeLayoutForGpuCustomCalls);
      pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);
    }
    pipeline.AddPass<BroadcastCanonicalizer>();

    pipeline.AddPass<ReductionDegenerateDimRemover>();
    pipeline.AddPass<ReductionLayoutNormalizer>();
    pipeline.AddPass<ReductionDimensionGrouper>();
    pipeline.AddPass<HloPassFix<ReductionSplitter>>();
    pipeline.AddPass<HloPassFix<GpuTreeReductionRewriter>>(compute_capability);
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

  AutotuningConfig autotune_config =
      stream_exec
          ? AutotuningConfig{DeviceConfig{stream_exec, device_allocator}}
          : AutotuningConfig{
                DevicelessConfig{gpu_target_config.device_description_str}};

  // Linearize collective schedule under SPMD partitioning if online autotuning
  // of convolutions is enabled.
  const bool enable_collecive_schedule_linearizer_for_spmd =
      hlo_module->config().use_spmd_partitioning() &&
      autotune_config.is_online() &&
      GpuConvAlgorithmPicker::IsEnabled(hlo_module);

  if (enable_collecive_schedule_linearizer_for_spmd) {
    pipeline.AddPass<CollectivesScheduleLinearizer>(
        RequiresCollectiveScheduleLinearizer);
  }

  if (autotune_config.is_offline()) {
    GpuConvAlgorithmPicker::ClearAutotuneResults();
    TF_RETURN_IF_ERROR(
        GpuConvAlgorithmPicker::LoadAutotuneResults(*autotune_results));
#if GOOGLE_CUDA
    GemmAlgorithmPicker::ClearAutotuneResults();
    TF_RETURN_IF_ERROR(
        GemmAlgorithmPicker::LoadAutotuneResults(*autotune_results));
    TritonAutotuner::ClearAutotuneResults();
    TF_RETURN_IF_ERROR(TritonAutotuner::LoadAutotuneResults(*autotune_results));
#endif  // GOOGLE_CUDA
  }
  if (GpuConvAlgorithmPicker::IsEnabled(hlo_module)) {
    pipeline.AddPass<GpuConvAlgorithmPicker>(autotune_config);
  }
#if GOOGLE_CUDA
  pipeline.AddPass<GemmAlgorithmPicker>(autotune_config);
  pipeline.AddPass<TritonAutotuner>(
      autotune_config,
      debug_options.xla_gpu_force_compilation_parallelism()
          ? debug_options.xla_gpu_force_compilation_parallelism()
          : tsl::port::MaxParallelism());
#endif  // GOOGLE_CUDA

  GpuFloatSupport bf16_support(BF16);
  pipeline.AddPass<FloatNormalization>(&bf16_support);
  GpuFloatSupport f8e5m2_support(F8E5M2);
  pipeline.AddPass<FloatNormalization>(&f8e5m2_support);
  GpuFloatSupport f8e4m3fn_support(F8E4M3FN);
  pipeline.AddPass<FloatNormalization>(&f8e4m3fn_support);

  // Remove `f32 -> bf16 -> f32` casts inserted by bf16 normalization.
  if (debug_options.xla_gpu_simplify_all_fp_conversions()) {
    pipeline.AddPass<SimplifyFPConversions>();
  }

  // Clean up new_tuple described above.
  pipeline.AddPass<TupleSimplifier>();

  {
    // The LayoutAssignment pass may leave behind kCopy instructions which are
    // duplicate or NOPs, so remove them with algebraic simplification and CSE.
    AlgebraicSimplifierOptions options;
    options.set_supports_non_canonical_dots(false);
    options.set_is_layout_sensitive(true);
    options.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    options.set_minmax_propagate_nan(
        !hlo_module->config().debug_options().xla_gpu_enable_fast_min_max());
    pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);
  }

  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return OkStatus();
}

StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GpuCompiler::RunHloPasses for ", module->name()));
  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tsl::profiler::TraceMeLevel::kInfo);

  GpuTargetConfig gpu_target_config = GetGpuTargetConfig(stream_exec);
  TF_RETURN_IF_ERROR(
      OptimizeHloModule(module.get(), stream_exec, options.device_allocator,
                        gpu_target_config, /*autotune_results=*/nullptr));

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

  uint64_t end_usecs = tsl::Env::Default()->NowMicros();

  // This won't record values for calls that error out (because if they error
  // out we have no way of telling how far through the process we got).
  RecordHloPassesDuration(end_usecs - start_usecs);

  return std::move(module);
}

StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPassesWithoutDevice(
    std::unique_ptr<HloModule> module, const CompileOptions& options,
    const GpuTargetConfig& gpu_target_config,
    const AutotuneResults& autotune_results) {
  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GpuCompiler::RunHloPasses for ", module->name()));
  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tsl::profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(), nullptr,
                                       options.device_allocator,
                                       gpu_target_config, &autotune_results));

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

  uint64_t end_usecs = tsl::Env::Default()->NowMicros();

  // This won't record values for calls that error out (because if they error
  // out we have no way of telling how far through the process we got).
  RecordHloPassesDuration(end_usecs - start_usecs);

  return std::move(module);
}

static std::optional<bool> DummyCanShareBufferFunction(const HloInstruction*,
                                                       const HloInstruction*,
                                                       const ShapeIndex&) {
  return std::nullopt;
}

StatusOr<std::unique_ptr<BufferAssignment>> GpuCompiler::AssignBuffers(
    HloModule* hlo_module, se::StreamExecutor* stream_exec) {
  const GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);
  TF_RETURN_IF_ERROR(
      ScheduleGpuModule(hlo_module, pointer_size_, gpu_device_info));

  auto buffer_size_bytes_function =
      [this](const BufferValue& buffer_value) -> int64_t {
    return GetSizeOfShape(buffer_value.shape(), pointer_size_);
  };

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          hlo_module,
          std::make_unique<SequentialHloOrdering>(hlo_module->schedule()),
          buffer_size_bytes_function,
          /*color_alignment=*/
          [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
          /*allocate_buffers_for_constants=*/true,
          /*colorer=*/BufferAssigner::DefaultColorer(),
          /*must_not_live_out=*/{}, GetCanShareBuffer()));

  return std::move(assignment);
}

// Lowers MLIR module to the XLA Gpu runtime custom calls.
static Status LowerToXlaGpuRuntime(mlir::ModuleOp module,
                                   llvm::StringRef entry_function_name,
                                   llvm::ArrayRef<int64_t> buffer_sizes,
                                   ThunkSequence* thunk_sequence,
                                   const DebugOptions& debug_options) {
  if (!module) {
    return InternalError("No MLIR module to lower.");
  }

  mlir::PassManager pm(module->getName(), mlir::PassManager::Nesting::Implicit);

  GpuPipelineOpts opts;
  opts.cuda_graph_level = debug_options.xla_gpu_cuda_graph_level();
  populateXlaGpuRuntimePasses(pm, thunk_sequence, opts);

  if (pm.run(module).failed()) {
    return InternalError("Failed to lower LMHLO to Gpu runtime custom calls.");
  }

  return OkStatus();
}

static StatusOr<OwnedGpuRuntimeProgram> LowerToJitRt(
    mlir::ModuleOp mlir_module, llvm::StringRef entry_function_name,
    llvm::ArrayRef<int64_t> buffer_sizes, const HloModuleConfig& module_config,
    std::unique_ptr<ThunkSequence> thunk_sequence,
    const HloModule* hlo_module_for_dump = nullptr) {
  // Forward collective (NCCL) attributes for use by the lowering pipeline.
  mlir::OpBuilder builder(mlir_module.getContext());
  mlir::IntegerAttr replica_count_attr =
      builder.getI64IntegerAttr(module_config.replica_count());
  mlir::IntegerAttr num_partitions_attr =
      builder.getI64IntegerAttr(module_config.num_partitions());
  mlir::func::FuncOp func =
      mlir_module.lookupSymbol<mlir::func::FuncOp>(entry_function_name);
  func->setAttr("replica_count", replica_count_attr);
  func->setAttr("num_partitions", num_partitions_attr);

  // Lower LMHLO operations to the JitRt compatible custom calls.
  TF_RETURN_IF_ERROR(LowerToXlaGpuRuntime(
      mlir_module, {entry_function_name.data(), entry_function_name.size()},
      buffer_sizes, thunk_sequence.get(), module_config.debug_options()));

  // TODO(b/232033540): Pass MLIR module directly to Gpu runtime executable
  // without forcing serialization.
  std::string module_str = llvm_ir::DumpToString(mlir_module);

  if (hlo_module_for_dump != nullptr) {
    DumpToFileInDirOrStdout(*hlo_module_for_dump, "gpu_rt_host", "mlir",
                            module_str);
  }

  return std::make_unique<GpuRuntimeProgram>(
      entry_function_name.str(), std::move(module_str), buffer_sizes.vec(),
      module_config.debug_options());
}

using OutputInfoMap =
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;
static Status GetMlirAllocationInfo(mlir::func::FuncOp func,
                                    std::vector<BufferAllocation>* allocations,
                                    OutputInfoMap* output_info,
                                    Shape* output_shape,
                                    EntryFunctionAttributes* entry_func_attrs);

namespace {
// Removes all globals from the given module that are both uninitialized and
// have no uses within that module.
void RemoveUnusedAndUninitializedGlobals(
    llvm::Module* llvm_module,
    const std::vector<GpuExecutable::ConstantInfo>& constants) {
  for (const auto& info : constants) {
    // Empty content means the constant is initialized in the LLVM IR, so we
    // must not remove it.
    if (!info.content.empty()) {
      llvm::GlobalVariable* global =
          llvm_module->getGlobalVariable(info.symbol_name);
      CHECK(global != nullptr);
      if (global->use_empty()) {
        global->eraseFromParent();
      }
    }
  }
}
}  // namespace

struct CompileModuleResults {
  std::unique_ptr<llvm::Module> llvm_module;
  std::unique_ptr<BufferAssignment> buffer_assignment;
  std::vector<BufferAllocation> allocations;
  std::variant<OwnedThunkSequence, OwnedGpuRuntimeProgram> executable;
  EntryFunctionAttributes entry_func_attrs;
  std::vector<GpuExecutable::ConstantInfo> constants;
  OutputInfoMap output_info;
  Shape output_shape;
  std::string module_name;
};

static void ForAllThunks(const std::function<void(Thunk*)>& fn,
                         ThunkSequence* thunk_sequence) {
  for (std::unique_ptr<Thunk>& thunk : *thunk_sequence) {
    if (thunk->kind() == Thunk::kConditional) {
      auto* cond_thunk = static_cast<ConditionalThunk*>(thunk.get());
      for (const std::unique_ptr<SequentialThunk>& branch_thunks :
           cond_thunk->branch_thunks()) {
        ForAllThunks(fn, &branch_thunks->thunks());
      }
    } else if (thunk->kind() == Thunk::kFor) {
      auto* for_thunk = static_cast<ForThunk*>(thunk.get());
      ForAllThunks(fn, &for_thunk->body_thunk_sequence()->thunks());
    } else if (thunk->kind() == Thunk::kSequential) {
      auto* sequential_thunk = static_cast<SequentialThunk*>(thunk.get());
      ForAllThunks(fn, &sequential_thunk->thunks());
    } else if (thunk->kind() == Thunk::kWhile) {
      auto* while_thunk = static_cast<WhileThunk*>(thunk.get());
      ForAllThunks(fn, &while_thunk->condition_thunk_sequence()->thunks());
      ForAllThunks(fn, &while_thunk->body_thunk_sequence()->thunks());
    } else {
      fn(thunk.get());
    }
  }
}

static bool HasFp8(const HloModule& hlo_module) {
  for (const HloComputation* computation : hlo_module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (ShapeUtil::HasPrimitiveType(instruction->shape(), F8E5M2) ||
          ShapeUtil::HasPrimitiveType(instruction->shape(), F8E4M3FN)) {
        return true;
      }
    }
  }
  return false;
}

// Prints mlir diagnostic messages to VLOG level 2.
static mlir::LogicalResult DiagnosticHandler(mlir::Diagnostic& diag) {
  VLOG(2) << diag.str();
  return mlir::failure();
}

// The order of `thunk_sequence` corresponds to
// `hlo_schedule->ThunkLaunchOrder()`.
static Status CompileModuleToLlvmIrImpl(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    int pointer_size, CompileModuleResults* results,
    se::StreamExecutor* stream_exec = nullptr) {
  results->llvm_module = std::make_unique<llvm::Module>("", *llvm_context);
  results->llvm_module->setTargetTriple(target_triple);
  results->llvm_module->setDataLayout(data_layout);

  TF_RETURN_IF_ERROR(
      ScheduleGpuModule(hlo_module, pointer_size, gpu_device_info));
  {
    HloPassPipeline pipeline("post-scheduling-passes");

    HloPredicate is_nop =
        HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant,
                         HloOpcode::kBitcast, HloOpcode::kGetTupleElement>;
    pipeline.AddPass<ConvertAsyncCollectivesToSync>(is_nop);
    pipeline.AddPass<OptimizationBarrierExpander>();

    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  auto buffer_size_bytes_function =
      [pointer_size](const BufferValue& buffer_value) -> int64_t {
    return GetSizeOfShape(buffer_value.shape(), pointer_size);
  };

  HloRematerialization::RematerializationSizes sizes;
  HloRematerialization remat(
      [pointer_size](const Shape& shape) {
        return GetSizeOfShape(shape, pointer_size);
      },
      // Assume 75% of the total device memory is available for XLA.
      /*memory_limit_bytes=*/gpu_device_info.device_memory_size * 0.75,
      /*sizes=*/&sizes,
      HloRematerialization::RematerializationPass::kPostFusion,
      /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
      /*compact_shape_function=*/nullptr,
      HloRematerialization::RematerializationMode::kRecomputeAndCompress);
  TF_ASSIGN_OR_RETURN(bool changed, remat.Run(hlo_module));
  if (changed) {
    VLOG(1) << "HloRematerialization saved "
            << sizes.before_bytes - sizes.after_bytes << " bytes";
  }

  TF_ASSIGN_OR_RETURN(
      results->buffer_assignment,
      BufferAssigner::Run(
          hlo_module,
          std::make_unique<SequentialHloOrdering>(hlo_module->schedule()),
          buffer_size_bytes_function,
          /*color_alignment=*/
          [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
          /*allocate_buffers_for_constants=*/true,
          /*colorer=*/BufferAssigner::DefaultColorer(),
          /*must_not_live_out=*/{}, can_share_buffer_function));

  VLOG(1) << "Buffer Assignment Stats for " << hlo_module->name() << "\n"
          << results->buffer_assignment->GetStats().ToString();
  DumpHloModuleIfEnabled(*hlo_module, *results->buffer_assignment,
                         absl::StrCat("sm_", cuda_compute_capability.ToString(),
                                      "_gpu_", kAfterOptimizationsDumpName));

  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  mlir::DialectRegistry registry;
  IrEmitterUnnested::GetDependentDialects(registry);
  mlir::MLIRContext mlir_context(registry);
  mlir_context.getDiagEngine().registerHandler(DiagnosticHandler);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::ModuleOp::create(mlir::Builder(&mlir_context).getUnknownLoc());

  TF_RETURN_IF_ERROR(
      HloToLhloModule(*results->buffer_assignment, *hlo_module, *mlir_module));

  results->module_name =
      mlir::mhlo::GetDebugNameFromLocation(mlir_module->getLoc());

  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "lmhlo", mlir_module.get());
  }

  auto entry_function = mlir::cast<mlir::func::FuncOp>(
      mlir_module->lookupSymbol(hlo_module->entry_computation()->name()));

  TF_RETURN_IF_ERROR(GetMlirAllocationInfo(
      entry_function, &results->allocations, &results->output_info,
      &results->output_shape, &results->entry_func_attrs));

  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr, platform_name,
      gpu_device_info, cuda_compute_capability, rocm_compute_capability,
      &mlir_context, results->llvm_module.get());

  ir_emitter_context.set_allocations(results->allocations);

  TF_ASSIGN_OR_RETURN(
      auto ir_emitter,
      IrEmitterUnnested::Create(hlo_module->config(), &ir_emitter_context));

  {
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
        "GpuCompiler::RunBackend - IR emission for ", hlo_module->name()));

    TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(&entry_function.getBody()));

    bool supports_runtime_managed_constants =
        // TODO(b/218907125): Implement this feature for ROCm as well.
        platform_id != se::rocm::kROCmPlatformId &&
        hlo_module->config().debug_options().xla_gpu_enable_shared_constants();
    if (supports_runtime_managed_constants) {
      // Remove these globals from the generated code to indicate that XLA is
      // responsible for allocating and initializing them.
      RemoveUnusedAndUninitializedGlobals(ir_emitter_context.llvm_module(),
                                          ir_emitter_context.constants());
    }

    results->constants = std::move(ir_emitter_context.constants());
    uint64_t end_usecs = tsl::Env::Default()->NowMicros();

    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    RecordHloToLlvmDuration(end_usecs - start_usecs);
  }

  // TODO(ezhulenev): Remove the FP8 check once https://reviews.llvm.org/D140088
  // is submitted. Currently we can't emit LLVM IR with fp8 types.
  if (IsXlaRuntimeExecutableEnabled(hlo_module->config()) &&
      !HasFp8(*hlo_module)) {
    std::vector<int64_t> buffer_sizes;
    llvm::transform(
        results->allocations, std::back_inserter(buffer_sizes),
        [](const BufferAllocation& allocation) { return allocation.size(); });
    TF_ASSIGN_OR_RETURN(
        results->executable,
        LowerToJitRt(*mlir_module, entry_function.getName(), buffer_sizes,
                     hlo_module->config(), ir_emitter->ConsumeThunkSequence(),
                     /*hlo_module_for_dump=*/hlo_module));
    return OkStatus();
  }

  auto thunk_sequence = ir_emitter->ConsumeThunkSequence();
  ForAllThunks([](Thunk* thunk) { thunk->ClearCompileTimeInfo(); },
               thunk_sequence.get());
  results->executable = std::move(thunk_sequence);
  return OkStatus();
}

static void NullDiagnosticHandler(const llvm::DiagnosticInfo& diag_info,
                                  void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info.print(diagnostic_printer);

  VLOG(5) << error_string;
}

StatusOr<std::pair<std::string, std::vector<uint8_t>>>
GpuCompiler::CompileToTargetBinary(const HloModuleConfig& module_config,
                                   std::unique_ptr<llvm::Module> llvm_module,
                                   GpuVersion gpu_version,
                                   se::StreamExecutor* stream_exec,
                                   const CompileOptions& options,
                                   const HloModule* debug_module) {
  using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;

  const auto compile_single_module =
      [this, gpu_version, &module_config, debug_module](
          llvm::Module* llvm_module, bool relocatable,
          std::optional<int> shard_number) -> StatusOr<BackendCompileResult> {
    {
      XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
          "GpuCompiler::RunBackend - Running LLVM verifier for ",
          (debug_module != nullptr ? debug_module->name() : "(unknown)")));

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
    StatusOr<std::pair<std::string, std::vector<uint8_t>>> result =
        CompileTargetBinary(module_config, llvm_module, gpu_version,
                            relocatable, debug_module);

    if (!result.ok()) {
      return result;
    }

    const bool should_dump =
        DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                   module_config.debug_options());

    if (should_dump) {
      if (debug_module) {
        if (shard_number.has_value()) {
          llvm_ir::DumpIrIfEnabled(*debug_module, *llvm_module,
                                   /*optimized=*/true,
                                   std::to_string(*shard_number));
        } else {
          llvm_ir::DumpIrIfEnabled(*debug_module, *llvm_module,
                                   /*optimized=*/true);
        }
      } else {
        LOG(ERROR)
            << "Dumping is not implemented since the file name cannot be "
               "inferred. Please implement (potentially MLIR) module -> "
               "filename heuristic.";
      }
    }

    if (user_post_optimization_hook_) {
      user_post_optimization_hook_(*llvm_module);
    }

    // Write PTX to IR dump directory, if IR dumping was requested.
    if (should_dump) {
      absl::string_view ptx = result->first;
      if (debug_module) {
        if (shard_number.has_value()) {
          DumpToFileInDirOrStdout(*debug_module, "",
                                  std::to_string(*shard_number) + ".ptx", ptx);
        } else {
          DumpToFileInDirOrStdout(*debug_module, "", "ptx", ptx);
        }
      } else {
        LOG(ERROR)
            << "Dumping is not implemented since the file name cannot be "
               "inferred. Please implement (potentially MLIR) module -> "
               "filename heuristic.";
      }
    }

    return result;
  };

  tsl::thread::ThreadPool* thread_pool;
  std::optional<tsl::thread::ThreadPool> overriding_thread_pool;
  switch (
      module_config.debug_options().xla_gpu_force_compilation_parallelism()) {
    case 0:
      thread_pool = options.thread_pool;
      break;
    case 1:
      thread_pool = nullptr;
      break;
    default:
      overriding_thread_pool.emplace(
          tsl::Env::Default(), "",
          module_config.debug_options()
              .xla_gpu_force_compilation_parallelism());
      thread_pool = &*overriding_thread_pool;
      break;
  }

  if (!thread_pool) {
    return compile_single_module(llvm_module.get(), /*relocatable=*/false,
                                 /*shard_number=*/std::nullopt);
  }

  // Test whether LinkModules is supported.
  TF_ASSIGN_OR_RETURN(bool can_use_link_modules,
                      CanUseLinkModules(module_config));
  if (!can_use_link_modules) {
    return compile_single_module(llvm_module.get(), /*relocatable=*/false,
                                 /*shard_number=*/std::nullopt);
  }

  std::vector<std::unique_ptr<llvm::Module>> llvm_modules;
  int num_functions = 0;
  for (llvm::Function& func : llvm_module->functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      num_functions++;
    }
  }

  // Record the name of some constant global variables and their initializers.
  // We'll change the linkage type of these variables from external to internal
  // to ensure constant-folding works properly after calling llvm::SplitModule.
  llvm::DenseMap<llvm::StringRef, llvm::Constant*> const_initializer_map;
  for (llvm::GlobalVariable& gv : llvm_module->globals()) {
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

  llvm::SplitModule(
      *llvm_module,
      std::max<unsigned>(
          1, std::min<unsigned>(thread_pool->NumThreads(), num_functions)),
      [&](std::unique_ptr<llvm::Module> module) {
        // Change the linkage type of some global constant variables to internal
        for (llvm::GlobalVariable& gv : module->globals()) {
          if (gv.hasName() && gv.isConstant() && !gv.hasInitializer() &&
              const_initializer_map.count(gv.getName()) != 0) {
            gv.setInitializer(const_initializer_map[gv.getName()]);
            gv.setLinkage(llvm::GlobalValue::InternalLinkage);
          }
        }
        llvm_modules.push_back(std::move(module));
      },
      /*PreserveLocals=*/true);

  std::vector<StatusOr<BackendCompileResult>> compile_results(
      llvm_modules.size());
  tsl::BlockingCounter counter(llvm_modules.size());
  for (int i = 0; i < llvm_modules.size(); i++) {
    thread_pool->Schedule(
        [&compile_results, compile_single_module, i, &llvm_modules, &counter] {
          llvm::Module* original_module = llvm_modules[i].get();
          llvm::LLVMContext context;

          std::unique_ptr<llvm::Module> new_llvm_module;
          // Switch to a new context by dumping and re-parsing LLVM IR. Each
          // thread has its own context to avoid race conditions.
          {
            std::string ir = llvm_ir::DumpToString(original_module);
            llvm::SMDiagnostic err;
            new_llvm_module = llvm::parseAssemblyString(ir, err, context);
            if (!new_llvm_module) {
              std::string err_string;
              llvm::raw_string_ostream os(err_string);
              err.print(/*ProgName=*/nullptr, os, /*ShowColors=*/false);
              LOG(FATAL) << "Failed to parse IR: " << err_string;
            }
          }

          compile_results[i] = compile_single_module(
              new_llvm_module.get(), /*relocatable=*/true, /*shard_number=*/i);
          counter.DecrementCount();
        });
  }
  counter.Wait();

  std::string ptx_snippets;
  std::vector<std::vector<uint8_t>> submodule_compile_results;
  for (auto& maybe_result : compile_results) {
    TF_ASSIGN_OR_RETURN(auto result, maybe_result);
    if (result.second.empty()) {
      continue;
    }
    ptx_snippets += result.first;
    ptx_snippets += "\n";
    submodule_compile_results.push_back(result.second);
  }

  auto maybe_backend_result =
      this->LinkModules(stream_exec, std::move(submodule_compile_results),
                        module_config.debug_options());
  if (!maybe_backend_result.ok()) {
    LOG(ERROR) << "The CUDA linking API did not work. Please use "
                  "XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 to "
                  "bypass it, but expect to get longer compilation time due to "
                  "the lack of multi-threading. Original error: "
               << maybe_backend_result.status();
    return maybe_backend_result.status();
  }

  return std::make_pair(ptx_snippets, std::move(*maybe_backend_result));
}

StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  VLOG(1) << "Starting to compile HLO module " << module->name();
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GpuCompiler::RunBackend for ", module->name()));
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

  TF_RET_CHECK(stream_exec != nullptr);

  llvm::LLVMContext llvm_context;

  const GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);

  if (module->config().hlo_profiling_enabled() || VLOG_IS_ON(1)) {
    HloCostAnalysis::Options options{ShapeSizeBytesFunction()};
    options.set_bytes_per_second(
        stream_exec->GetDeviceDescription().memory_bandwidth());
    GpuHloCostAnalysis cost_analysis(options);
    TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&cost_analysis));
    VLOG(1) << "HLO memory read+written: "
            << tsl::strings::HumanReadableNumBytes(
                   cost_analysis.bytes_accessed());
    if (module->config().hlo_profiling_enabled()) {
      LOG(ERROR) << "--xla_hlo_profile for GPU is unsupported.";
    }
  }

  CompileModuleResults compile_module_results;
  TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
      module.get(), &llvm_context, target_triple_, data_layout_,
      stream_exec->platform()->Name(), stream_exec->platform()->id(),
      gpu_device_info,
      stream_exec->GetDeviceDescription().cuda_compute_capability(),
      stream_exec->GetDeviceDescription().rocm_compute_capability(),
      GetCanShareBuffer(), pointer_size_, &compile_module_results,
      stream_exec));

  if (user_pre_optimization_hook_) {
    user_pre_optimization_hook_(*compile_module_results.llvm_module);
  }
  std::string ir_module_string_before_opt;
  const bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  if (embed_ir_in_executable) {
    ir_module_string_before_opt =
        llvm_ir::DumpToString(compile_module_results.llvm_module.get());
  }

  llvm_ir::DumpIrIfEnabled(*module, *compile_module_results.llvm_module,
                           /*optimized=*/false);

  using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;
  TF_ASSIGN_OR_RETURN(
      BackendCompileResult backend_result,
      CompileToTargetBinary(
          module->config(), std::move(compile_module_results.llvm_module),
          GetGpuVersion(stream_exec), stream_exec, options, module.get()));
  if (DumpingEnabledForHloModule(*module) &&
      std::holds_alternative<OwnedThunkSequence>(
          compile_module_results.executable)) {
    const ThunkSequence& thunk_sequence =
        *std::get<OwnedThunkSequence>(compile_module_results.executable);
    DumpToFileInDirOrStdout(*module, "", "thunk_sequence.txt",
                            thunk_sequence.ToString());
  }

  auto buffer_assignment_proto = std::make_unique<BufferAssignmentProto>(
      compile_module_results.buffer_assignment->ToProto());

  // Make it shared to be captured in the following lambda.
  std::shared_ptr<const BufferAssignment> buffer_assignment(
      std::move(compile_module_results.buffer_assignment));

  GpuVersion gpu_version = GetGpuVersion(stream_exec);
  TF_ASSIGN_OR_RETURN(
      auto gpu_executable,
      GpuExecutable::Create(
          {std::move(backend_result.first), std::move(backend_result.second),
           gpu_version, std::move(compile_module_results.executable),
           compile_module_results.entry_func_attrs,
           std::move(compile_module_results.constants),
           std::move(compile_module_results.output_info),
           compile_module_results.module_name,
           compile_module_results.output_shape,
           std::move(compile_module_results.allocations),
           std::move(buffer_assignment_proto),
           [buffer_assignment] { return buffer_assignment->ToVerboseString(); },
           std::move(module)}));
  if (embed_ir_in_executable) {
    DCHECK_NE("", ir_module_string_before_opt);
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }

  // Dump computation proto state and buffer assignment for
  // CompiledMemoryAnalysis.
  auto hlo_proto = std::make_unique<HloProto>();
  *hlo_proto->mutable_hlo_module() = gpu_executable->module().ToProto();
  *hlo_proto->mutable_buffer_assignment() = buffer_assignment->ToProto();
  gpu_executable->set_hlo_proto(std::move(hlo_proto));
  gpu_executable->set_debug_info(buffer_assignment->GetStats().ToString());
  return static_cast<std::unique_ptr<Executable>>(std::move(gpu_executable));
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
GpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& options) {
  CHECK(options.PlatformId() == se::cuda::kCudaPlatformId);

  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();
  std::vector<std::unique_ptr<AotCompilationResult>> results;

  std::any target_config = options.target_config();
  auto* gpu_target_config = std::any_cast<GpuTargetConfig>(&target_config);
  CHECK(gpu_target_config != nullptr || options.executor() != nullptr);

  for (const auto& module : modules) {
    llvm::LLVMContext llvm_context;

    // Compile the module
    CompileModuleResults compile_module_results;

    const std::any& target_config = options.target_config();
    auto* gpu_target_config = std::any_cast<GpuTargetConfig>(&target_config);

    if (gpu_target_config) {
      // CUDA "CC" major value, -1 if not available.
      se::CudaComputeCapability cuda_compute_capability{-1, -1};
      // ROCm gfx arch,  "gfx000" if not available.
      se::RocmComputeCapability rocm_compute_capability{"gfx000"};
      if (auto* cuda = std::get_if<se::CudaComputeCapability>(
              &gpu_target_config->gpu_version)) {
        cuda_compute_capability = *cuda;
      } else {
        rocm_compute_capability =
            std::get<se::RocmComputeCapability>(gpu_target_config->gpu_version);
      }

      TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
          module.get(), &llvm_context, target_triple_, data_layout_,
          gpu_target_config->platform_name, options.PlatformId(),
          gpu_target_config->gpu_device_info, cuda_compute_capability,
          rocm_compute_capability, GetCanShareBuffer(), pointer_size_,
          &compile_module_results));
    } else {
      CHECK(options.executor() != nullptr);
      auto stream_exec = options.executor();
      const stream_executor::DeviceDescription& device_description =
          stream_exec->GetDeviceDescription();
      TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
          module.get(), &llvm_context, target_triple_, data_layout_,
          stream_exec->platform()->Name(), options.PlatformId(),
          GetGpuDeviceInfo(stream_exec),
          device_description.cuda_compute_capability(),
          device_description.rocm_compute_capability(), GetCanShareBuffer(),
          pointer_size_, &compile_module_results));
    }

    if (user_pre_optimization_hook_) {
      user_pre_optimization_hook_(*compile_module_results.llvm_module);
    }

    using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;
    BackendCompileResult backend_result;
    if (gpu_target_config) {
      TF_ASSIGN_OR_RETURN(
          backend_result,
          CompileToTargetBinary(
              module->config(), std::move(compile_module_results.llvm_module),
              gpu_target_config->gpu_version, options.executor(),
              {options.device_allocator()}, module.get()));
    } else {
      TF_ASSIGN_OR_RETURN(
          backend_result,
          CompileToTargetBinary(
              module->config(), std::move(compile_module_results.llvm_module),
              GetGpuVersion(options.executor()), options.executor(),
              {options.device_allocator()}, module.get()));
    }

    auto& compiled_executable = compile_module_results.executable;

    if (!std::holds_alternative<OwnedGpuRuntimeProgram>(compiled_executable)) {
      return InternalError("Gpu runtime program was not provided");
    }

    // TODO(ezhulenev): Unify AOT compilation with GpuRuntimeExecutable::Create
    // (see `gpu/runtime/executable.h`).

    const auto& program = std::get<OwnedGpuRuntimeProgram>(compiled_executable);

    // Options for the default XLA runtime compilation pipeline.
    runtime::CompilationPipelineOptions copts;

    // Populate mapping from XLA (SE) enums/structs type id to symbol names.
    copts.populate_type_id_names = RegisterXlaGpuTypeIdNames;

    // For passing LMHLO attributes as XLA (SE) enums/structs to custom calls.
    copts.populate_attr_encodings = RegisterXlaGpuAttrEncoding;

    // Options for constructing XLA runtime JitExecutable.
    runtime::JitExecutable::Options opts;
    opts.specialization = runtime::JitExecutable::Specialization::kDisabled;
    opts.compiler.register_dialects =
        runtime::RegisterDefaultXlaGpuRuntimeDialects;

    // Register XLA Gpu runtime custom calls with the linker.
    opts.compiler.symbols_binding = runtime::ToSymbolsBinding(
        RegisterXlaGpuRuntimeCustomCalls, RegisterXlaGpuTypeIdNames);

    opts.compiler.create_compilation_pipeline =
        [copts](xla::runtime::PassManager& passes) {
          runtime::CreateDefaultXlaGpuRuntimeCompilationPipeline(passes, copts);
        };

    // Instantiate new JitExecutable from the MLIR source.
    auto jit_executable = runtime::JitExecutable::Instantiate(
        program->module, program->entry_point, opts);
    if (!jit_executable.ok())
      return InternalError("Failed to compile XLA program: %s",
                           jit_executable.status().message());

    // For static shapes we can always serialize only the default executable.
    runtime::Executable& executable = jit_executable->DefaultExecutable().get();

    // Check if XLA runtime executable saved the compilation result.
    std::unique_ptr<llvm::MemoryBuffer> obj_file = executable.obj_file();
    if (!obj_file)
      return InternalError("XLA runtime executable didn't save the obj file");

    std::string data(obj_file->getBuffer().data(),
                     obj_file->getBuffer().size());

    results.emplace_back(std::make_unique<GpuXlaRuntimeAotCompilationResult>(
        module->ToProto(), data, program->module,
        compile_module_results.entry_func_attrs, backend_result.first,
        backend_result.second, compile_module_results.constants));
  }
  return std::move(results);
}

HloCostAnalysis::ShapeSizeFunction GpuCompiler::ShapeSizeBytesFunction() const {
  // Capture just the pointer size, not the entire GpuCompiler object.
  return [pointer_size = pointer_size_](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
}

StatusOr<std::unique_ptr<AotCompilationResult>> GpuCompiler::Export(
    Executable* executable) const {
  auto* gpu_executable = tensorflow::down_cast<GpuExecutable*>(executable);
  if (!gpu_executable) return Internal("GpuExecutable is null");
  HloModuleProto module_proto = gpu_executable->module().ToProto();
  TF_ASSIGN_OR_RETURN(auto obj_file, gpu_executable->GetObjFile());
  TF_ASSIGN_OR_RETURN(auto mlir_module, gpu_executable->GetMlirModule());
  xla::EntryFunctionAttributes entry_func_attrs =
      gpu_executable->entry_func_attrs();
  auto text = gpu_executable->text();
  auto binary = gpu_executable->binary();

  std::unique_ptr<AotCompilationResult> result =
      std::make_unique<xla::gpu::GpuXlaRuntimeAotCompilationResult>(
          module_proto, obj_file, mlir_module, entry_func_attrs, text, binary,
          gpu_executable->constants());
  return result;
}

StatusOr<std::unique_ptr<llvm::Module>> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, const se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability, int pointer_size) {
  CompileModuleResults results;
  TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
      hlo_module, llvm_context, target_triple, data_layout, platform_name,
      platform_id, gpu_device_info, cuda_compute_capability,
      rocm_compute_capability, DummyCanShareBufferFunction, pointer_size,
      &results));
  return std::move(results.llvm_module);
}

// Analyze the function signature to reconstruct a vector of BufferAllocation
// objects, as well as other output information.
//
// This function also serves as a half-baked verifier for function arg
// attributes, since a full verifier doens't exist yet.
static Status GetMlirAllocationInfo(mlir::func::FuncOp func,
                                    std::vector<BufferAllocation>* allocations,
                                    OutputInfoMap* output_info,
                                    Shape* output_shape,
                                    EntryFunctionAttributes* entry_func_attrs) {
  CHECK(allocations->empty());
  allocations->reserve(func.getNumArguments());

  std::vector<int64_t> buffer_sizes;
  for (int i = 0; i < func.getNumArguments(); i++) {
    mlir::BlockArgument arg = func.getArgument(i);

    TF_RET_CHECK(arg.getType().isa<mlir::ShapedType>());
    mlir::ShapedType type = arg.getType().cast<mlir::ShapedType>();
    TF_ASSIGN_OR_RETURN(auto element_type_bytes,
                        GetElementTypeBytes(type.getElementType()));
    size_t size = type.getNumElements() * element_type_bytes;
    buffer_sizes.push_back(size);
  }

  for (int i = 0; i < func.getNumArguments(); i++) {
    llvm::ArrayRef<mlir::NamedAttribute> attrs =
        mlir::function_interface_impl::getArgAttrs(func, i);
    for (const mlir::NamedAttribute& attr : attrs) {
      TF_RET_CHECK(attr.getName() == "lmhlo.params" ||
                   attr.getName() == "lmhlo.param_shape_index" ||
                   attr.getName() == "lmhlo.constant_name" ||
                   attr.getName() == "lmhlo.must_alias" ||
                   attr.getName() == "lmhlo.output_index");
    }
  }

  // Encode buffer parameter metadata in a proto for persisting, because BEF
  // doesn't persist function attributes.
  for (int i = 0; i < func.getNumArguments(); i++) {
    auto buffer = entry_func_attrs->add_buffers();
    if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
      buffer->set_lmhlo_params_present(true);
      buffer->set_lmhlo_params(param_attr.cast<mlir::IntegerAttr>().getInt());
    }
    if (auto shape_index_attr = func.getArgAttr(i, "lmhlo.param_shape_index")) {
      auto param_shape_index = buffer->mutable_lmhlo_param_shape_index();
      for (const llvm::APInt& element :
           shape_index_attr.cast<mlir::DenseIntElementsAttr>()) {
        param_shape_index->add_indices(element.getSExtValue());
      }
    }
    if (auto constant_name_attr = func.getArgAttr(i, "lmhlo.constant_name")) {
      buffer->set_lmhlo_constant_name(
          constant_name_attr.cast<mlir::StringAttr>().str());
    }
    if (func.getArgAttr(i, "lmhlo.must_alias")) {
      buffer->set_lmhlo_must_alias(true);
    }
    if (auto output_index_attr = func.getArgAttr(i, "lmhlo.output_index")) {
      auto output_index = buffer->mutable_lmhlo_output_index();
      for (const llvm::APInt& element :
           output_index_attr.cast<mlir::DenseIntElementsAttr>()) {
        output_index->add_indices(element.getSExtValue());
      }
    }
  }
  entry_func_attrs->set_result_xla_shape(
      func->getAttrOfType<mlir::StringAttr>("result_xla_shape")
          .getValue()
          .str());

  return GpuExecutable::SetUpMlirAllocation(func, buffer_sizes, allocations,
                                            output_info, output_shape);
}

StatusOr<std::unique_ptr<Executable>> CompileLmhloToExecutable(
    GpuCompiler* compiler, mlir::ModuleOp module, std::string module_name,
    const HloModuleConfig& module_config,
    const Compiler::CompileOptions& options,
    absl::string_view entry_function_name, se::StreamExecutor* stream_exec,
    std::unique_ptr<llvm::Module> llvm_module,
    IrEmitterContext* ir_emitter_context) {
  mlir::func::FuncOp entry_function =
      mlir::cast<mlir::func::FuncOp>(module.lookupSymbol(llvm::StringRef(
          entry_function_name.data(), entry_function_name.size())));

  std::vector<BufferAllocation> allocations;
  OutputInfoMap output_info;
  Shape output_shape;
  EntryFunctionAttributes entry_func_attrs;
  TF_RETURN_IF_ERROR(GetMlirAllocationInfo(entry_function, &allocations,
                                           &output_info, &output_shape,
                                           &entry_func_attrs));

  TF_RET_CHECK(!allocations.empty());

  ir_emitter_context->set_allocations(allocations);

  TF_ASSIGN_OR_RETURN(auto ir_emitter, IrEmitterUnnested::Create(
                                           module_config, ir_emitter_context));
  TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(&entry_function.getBody()));

  bool supports_runtime_managed_constants =
      // TODO(b/218907125): Implement this feature for ROCm as well.
      compiler->PlatformId() != se::rocm::kROCmPlatformId;
  if (supports_runtime_managed_constants) {
    // Remove these globals from the generated code to indicate that XLA is
    // responsible for allocating and initializing them.
    RemoveUnusedAndUninitializedGlobals(ir_emitter_context->llvm_module(),
                                        ir_emitter_context->constants());
  }

  using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;
  TF_ASSIGN_OR_RETURN(BackendCompileResult backend_result,
                      compiler->CompileToTargetBinary(
                          module_config, std::move(llvm_module),
                          compiler->GetGpuVersion(stream_exec), stream_exec,
                          options, /*debug_module=*/nullptr));

  if (IsXlaRuntimeExecutableEnabled(module_config)) {
    std::vector<int64_t> buffer_sizes;
    llvm::transform(
        allocations, std::back_inserter(buffer_sizes),
        [](const BufferAllocation& allocation) { return allocation.size(); });
    TF_ASSIGN_OR_RETURN(
        auto executable,
        LowerToJitRt(module, entry_function.getName(), buffer_sizes,
                     module_config, ir_emitter->ConsumeThunkSequence()));

    GpuVersion gpu_version = compiler->GetGpuVersion(stream_exec);
    return GpuExecutable::Create(
        {std::move(backend_result.first), std::move(backend_result.second),
         gpu_version, std::move(executable), entry_func_attrs,
         std::move(ir_emitter_context->constants()), std::move(output_info),
         module_name, output_shape, std::move(allocations)});
  }

  auto thunk_sequence = ir_emitter->ConsumeThunkSequence();

  GpuVersion gpu_version = compiler->GetGpuVersion(stream_exec);
  return GpuExecutable::Create(
      {std::move(backend_result.first), std::move(backend_result.second),
       gpu_version, std::move(thunk_sequence), entry_func_attrs,
       std::move(ir_emitter_context->constants()), std::move(output_info),
       module_name, output_shape, std::move(allocations)});
}

}  // namespace gpu
}  // namespace xla

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

#include <atomic>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Transforms/Passes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/LocationSnapshot.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Transforms/gpu_passes.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/all_gather_broadcast_reorder.h"
#include "tensorflow/compiler/xla/service/all_gather_combiner.h"
#include "tensorflow/compiler/xla/service/all_gather_decomposer.h"
#include "tensorflow/compiler/xla/service/all_reduce_combiner.h"
#include "tensorflow/compiler/xla/service/all_reduce_contiguous.h"
#include "tensorflow/compiler/xla/service/all_reduce_folder.h"
#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"
#include "tensorflow/compiler/xla/service/all_to_all_decomposer.h"
#include "tensorflow/compiler/xla/service/async_collective_creator.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/bitcast_decomposer.h"
#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"
#include "tensorflow/compiler/xla/service/broadcast_canonicalizer.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/collectives_schedule_linearizer.h"
#include "tensorflow/compiler/xla/service/comparison_expander.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/convolution_4d_expander.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dot_merger.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_simplifier.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/dynamic_padder.h"
#include "tensorflow/compiler/xla/service/eigh_expander.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/gpu/alias_passthrough_params.h"
#include "tensorflow/compiler/xla/service/gpu/all_reduce_blueconnect.h"
#include "tensorflow/compiler/xla/service/gpu/bef_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_broadcast_folding_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_reduce_scatter_creator.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_sanitize_constant_names.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_scatter_expander.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_fusion_stats.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_input_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/metrics.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_dimension_grouper.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_splitter.h"
#include "tensorflow/compiler/xla/service/gpu/runtime_intrinsics.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/tree_reduction_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/variadic_op_splitter.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
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
#include "tensorflow/compiler/xla/service/reshape_decomposer.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/result_caster.h"
#include "tensorflow/compiler/xla/service/rng_bit_generator_expander.h"
#include "tensorflow/compiler/xla/service/rng_expander.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/sharding_remover.h"
#include "tensorflow/compiler/xla/service/simplify_fp_conversions.h"
#include "tensorflow/compiler/xla/service/slice_sinker.h"
#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/stable_sort_expander.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_trip_count_annotator.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/env_var.h"

#if XLA_ENABLE_XLIR
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pass_utils.h"
#include "tfrt/gpu/gpu_executor.h"  // from @tf_runtime
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef_translate.h"  // from @tf_runtime
#endif  // XLA_ENABLE_XLIR

namespace xla {
namespace gpu {
namespace {

class GpuBfloat16Support : public BFloat16Support {
 public:
  explicit GpuBfloat16Support(bool supports_matrix_multiplication,
                              se::StreamExecutor* stream_exec)
      : supports_matrix_multiplication_(supports_matrix_multiplication),
        stream_exec_(stream_exec) {}

  bool SupportsBF16Operand(const HloInstruction& hlo,
                           int64_t operand_index) const override {
    return BFloat16Support::SupportsBF16Operand(hlo, operand_index) ||
           IsSupported(hlo);
  }

  // Returns whether the backend supports BF16 output for the HLO instruction.
  bool SupportsBF16Output(const HloInstruction& hlo) const override {
    return BFloat16Support::SupportsBF16Output(hlo) || IsSupported(hlo);
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
      case HloOpcode::kConvolution:
        return IsConvBF16Supported();
      default:
        return supports_matrix_multiplication_ &&
               gpu::IsMatrixMultiplication(hlo);
    }
  }

  bool IsConvBF16Supported() const {
    if (se::dnn::DnnSupport* dnn = stream_exec_->AsDnn()) {
      se::port::StatusOr<se::dnn::VersionInfo> cudnn_version =
          dnn->GetVersion();
      return cudnn_version.ok() &&
             (cudnn_version->major_version() > 8 ||
              (cudnn_version->major_version() == 8 &&
               cudnn_version->minor_version() >= 2)) &&
             stream_exec_->GetDeviceDescription()
                 .cuda_compute_capability()
                 .IsAtLeast(se::CudaComputeCapability::AMPERE);
    }
    return false;
  }

  bool supports_matrix_multiplication_;
  se::StreamExecutor* stream_exec_;
};

int64_t GetSizeOfShape(const Shape& shape, int pointer_size) {
  if (shape.is_static() || shape.IsTuple()) {
    return ShapeUtil::ByteSizeOf(shape, pointer_size);
  }
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return ShapeUtil::ByteSizeOf(shape, pointer_size) + metadata_size;
}

bool ConvIsLowerable(HloInstruction* conv) {
  return conv_matchers::CanImplementAsGpuForwardConv(conv) ||
         std::get<0>(conv_matchers::MatchBackwardFilter(conv)) ||
         std::get<0>(conv_matchers::MatchBackwardInput(conv));
}

}  // end anonymous namespace

using OwnedThunkSchedule = GpuExecutable::OwnedThunkSchedule;
using OwnedBefBuffer = GpuExecutable::OwnedBefBuffer;
using OwnedJitRtProgram = GpuExecutable::OwnedJitRtProgram;

StatusOr<std::unique_ptr<Executable>> GpuAotCompilationResult::LoadExecutable(
    Compiler* compiler, se::StreamExecutor* executor) const {
  TF_ASSIGN_OR_RETURN(
      HloModuleConfig hlo_module_config,
      HloModule::CreateModuleConfigFromProto(bef_executable_.hlo_module_proto(),
                                             GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(bef_executable_.hlo_module_proto(),
                                 hlo_module_config));
  auto gpu_compiler = tensorflow::down_cast<GpuCompiler*>(compiler);
  return GpuExecutable::LoadFromBef(
      std::move(hlo_module), bef_executable_.bef(),
      bef_executable_.entry_func_attrs(), gpu_compiler->GetGpuVersion(executor),
      executor);
}

GpuCompiler::GpuCompiler(se::Platform::Id platform_id,
                         const char* target_triple, const char* data_layout)
    : platform_id_(platform_id),
      target_triple_(target_triple),
      data_layout_(data_layout),
      pointer_size_(llvm::DataLayout(data_layout)
                        .getPointerSize(0 /* default address space */)) {}

// Runs optimization passes on the given HLO module.
Status GpuCompiler::OptimizeHloModule(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  // Save proto state before optimizations if we want a snapshot.
  if (DumpingEnabledForHloModule(*hlo_module)) {
    hlo_proto_ = std::make_unique<HloProto>();
    *hlo_proto_->mutable_hlo_module() = hlo_module->ToProto();
  }

  const DebugOptions& debug_options = hlo_module->config().debug_options();

  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      // Run some IR cleanup passes before running the SPMD partitioning
      // passes.
      spmd_pipeline.AddInvariantChecker<HloVerifier>(HloVerifierOpts{});
      spmd_pipeline.AddPass<CallInliner>();
      spmd_pipeline.AddPass<ZeroSizedHloElimination>();
      spmd_pipeline.AddPass<ConditionalCanonicalizer>();

      HloPassPipeline& spmd_simplify =
          spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

      AlgebraicSimplifierOptions options;
      options.set_enable_conv_operand_swap(false);
      // "slow" minmax means we propagate nan.
      options.set_minmax_propagate_nan(
          !debug_options.xla_gpu_enable_fast_min_max());
      spmd_simplify.AddPass<AlgebraicSimplifier>(options);

      spmd_simplify.AddPass<SortSimplifier>();
      spmd_simplify.AddPass<TupleSimplifier>();
      spmd_simplify.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);
      spmd_simplify.AddPass<GatherExpander>(
          GatherExpander::kEliminateSimpleGathers);
      spmd_simplify.AddPass<WhileLoopConstantSinking>();
      spmd_simplify.AddPass<WhileLoopSimplifier>();

      spmd_simplify.AddPass<ReshapeMover>();
      spmd_simplify.AddPass<HloConstantFolding>();
      spmd_simplify.AddPass<ConditionalSimplifier>();
      spmd_simplify.AddPass<HloDCE>();

      spmd_pipeline.AddPass<ShardingPropagation>(
          /*is_spmd=*/true, /*propagate_metadata=*/false,
          hlo_module->config().allow_spmd_sharding_propagation_to_output());
      spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
          num_partitions, hlo_module->config().replica_count());
    } else {
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }

  {
    HloPassPipeline pipeline("optimization");
    pipeline.AddInvariantChecker<HloVerifier>(HloVerifierOpts{});
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
      // Scatter is nondeterministic, so eliminate all Scatters.
      pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);
    } else {
      // Only Scatters unsupported on XLA:GPU are eliminated.
      pipeline.AddPass<GpuScatterExpander>();
    }
    // TODO(phawkins): replace QR and Eigh decompositions with calls to
    // cuSOLVER.
    pipeline.AddPass<QrExpander>();
    pipeline.AddPass<EighExpander>();

    pipeline.AddPass<DynamicIndexSplitter>();

    // TODO(b/64094172): make Call work on GPU instead of inlining.
    pipeline.AddPass<CallInliner>();

    pipeline.AddPass<DotDecomposer>();

    pipeline.AddPass<Convolution4DExpander>();

    // Expand the sort op to support stable sorting if required.
    pipeline.AddPass<StableSortExpander>();

    GpuBfloat16Support bf16(/*supports_matrix_multiplication=*/true,
                            stream_exec);
    pipeline.AddPass<BFloat16Normalization>(&bf16);

    // Remove `f32 -> bf16 -> f32` casts inserted by bf16 normalization.
    if (debug_options.xla_gpu_simplify_all_fp_conversions())
      pipeline.AddPass<SimplifyFPConversions>();

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
      pipeline.AddInvariantCheckerDebug<HloVerifier>(HloVerifierOpts{});

      // BatchNormExpander can create zero-sized ops, so zero-sized HLO
      // elimination has to come after that pass.
      pipeline.AddPass<ZeroSizedHloElimination>();

      pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
      pipeline.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);

      AlgebraicSimplifierOptions options({}, ConvIsLowerable);
      // "slow" minmax means we propagate nan.
      options.set_minmax_propagate_nan(
          !debug_options.xla_gpu_enable_fast_min_max());

      const se::Platform* platform = stream_exec->platform();
      if (platform->Name() == "ROCM") {
        // SwapConvOperands does not yet work on ROCM
        options.set_enable_conv_operand_swap(false);
      }
      pipeline.AddPass<AlgebraicSimplifier>(options);
      pipeline.AddPass<BitcastDtypesExpander>();
      // AlgebraicSimplifier may add contracting dimensions to a dot.
      pipeline.AddPass<DotDecomposer>();
      // Only merge "smallish" dots.  This threshold was not set carefully, but
      // so far we know that 1mb is too small.
      pipeline.AddPass<DotMerger>(/*max_size_to_merge=*/int64_t{16} << 20);
      pipeline.AddPass<SortSimplifier>();
      pipeline.AddPass<TupleSimplifier>();
      pipeline.AddPass<WhileLoopConstantSinking>();
      pipeline.AddPass<WhileLoopSimplifier>();

      // TODO(b/134075051): Re-enable after b/134075051 is fixed.
      // pipeline.AddPass<SliceSinker>();

      pipeline.AddPass<ReshapeMover>();
      pipeline.AddPass<HloConstantFolding>();
      pipeline.AddPass<ConditionalSimplifier>();
      pipeline.AddPass<RealImagExpander>();
      pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot);
      pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
      pipeline.AddPass<HloDCE>();
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
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  // Optimize collectives generated by SPMD partitioning. Enable these passes
  // otherwise as well so that all collectives can get these optimizations.
  {
    HloPassPipeline collectives_pipeline("collective-optimizations");
    collectives_pipeline.AddPass<AllReduceFolder>();
    collectives_pipeline.AddPass<ReduceScatterCreator>();
    collectives_pipeline.AddPass<AllReduceReassociate>();

    // Run algebraic simplifier to reshape(broadcast) into a broadcast when
    // the reshape is just adding a unit dimension. This will help with the
    // AllGatherBroadcastReorder pass.
    AlgebraicSimplifierOptions options;
    options.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    options.set_minmax_propagate_nan(
        !debug_options.xla_gpu_enable_fast_min_max());

    collectives_pipeline.AddPass<AlgebraicSimplifier>(options);

    collectives_pipeline.AddPass<AllGatherBroadcastReorder>();
    TF_RETURN_IF_ERROR(collectives_pipeline.Run(hlo_module).status());
  }

  // Run target-specific HLO optimization passes for convolution
  // canonicalization.
  TF_RETURN_IF_ERROR(OptimizeHloConvolutionCanonicalization(
      hlo_module, stream_exec, device_allocator));

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
  TF_RETURN_IF_ERROR(OptimizeHloPostLayoutAssignment(hlo_module, stream_exec,
                                                     device_allocator));

  {
    HloPassFix<HloPassPipeline> fusion("fusion");
    // We try to split variadic ops with many parameters into several such ops
    // to avoid exceeding the parameter space.
    fusion.AddPass<VariadicOpSplitter>();
    fusion.AddInvariantCheckerDebug<HloVerifier>(
        HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
            LayoutAssignment::InstructionCanChangeLayout));
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/false);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true);
    fusion.AddPass<FusionMerger>();
    fusion.AddPass<GpuMultiOutputFusion>();
    fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                           /*only_fusion_computations=*/true);
    fusion.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(fusion.Run(hlo_module).status());
  }

  {
    HloPassFix<HloPassPipeline> horizontal_fusion("horizontal fusion");
    horizontal_fusion.AddPass<GpuHorizontalLoopFusion>();
    horizontal_fusion.AddPass<GpuHorizontalInputFusion>();
    // FusionBitcastLift must be after InstructionFusion, as it undoes
    // part of it.
    // TODO(b/209005695) Renable once the bug is fixed.
    // horizontal_fusion.AddPass<FusionBitcastLift>();
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

    if (debug_options.xla_gpu_enable_async_all_reduce()) {
      AsyncCollectiveCreator::CollectiveCreatorConfig config;
      config.convert_all_reduce = [](const HloInstruction*) { return true; };
      pipeline.AddPass<AsyncCollectiveCreator>(std::move(config));
    }

    pipeline.AddPass<CollectivesScheduleLinearizer>();

    AlgebraicSimplifierOptions options;
    options.set_is_layout_sensitive(true);
    options.set_enable_conv_operand_swap(false);
    // "slow" minmax means we propagate nan.
    options.set_minmax_propagate_nan(
        !debug_options.xla_gpu_enable_fast_min_max());
    pipeline.AddPass<AlgebraicSimplifier>(options);
    pipeline.AddPass<OptimizationBarrierExpander>();
    pipeline.AddPass<BitcastDecomposer>();
    pipeline.AddPass<TupleSimplifier>();

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
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout));

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
  pipeline.AddPass<GpuSanitizeConstantNames>();
  return pipeline.Run(hlo_module).status();
}

Status GpuCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();

  {
    HloPassPipeline pipeline("hlo normalization");
    pipeline.AddPass<ReshapeDecomposer>();
    pipeline.AddPass<ReduceDecomposer>([&](const HloInstruction* r) {
      return IsReductionFromOrToContiguousDimensions(*r);
    });
    if (hlo_module->config().debug_options().xla_gpu_normalize_layouts()) {
      pipeline.AddPass<LayoutNormalization>();
    }
    pipeline.AddPass<BroadcastCanonicalizer>();
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  HloPassPipeline pipeline("post-layout_assignment");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout));

  pipeline.AddPass<ReshapeDecomposer>();
  pipeline.AddPass<ReductionDegenerateDimRemover>();
  pipeline.AddPass<ReductionLayoutNormalizer>();
  pipeline.AddPass<ReductionDimensionGrouper>();
  pipeline.AddPass<HloPassFix<ReductionSplitter>>();
  pipeline.AddPass<HloPassFix<GpuTreeReductionRewriter>>(
      stream_exec->GetDeviceDescription().cuda_compute_capability());

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  options.set_enable_conv_operand_swap(false);
  // "slow" minmax means we propagate nan.
  options.set_minmax_propagate_nan(
      !hlo_module->config().debug_options().xla_gpu_enable_fast_min_max());
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

  // GemmRewriter assumes that all transposes are folded into gemms, but,
  // since commit 7d529df, this is not always true at this point.
  // Therefore, rerun transpose folding.
  pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot,
                                     TransposeFolding::NeverFoldTranspose);
  // Rewrite GEMMs into custom calls.
  pipeline.AddPass<GemmRewriter>();

  // Rewrite GEMMs with broadcasted inputs as strided GEMMs.
  pipeline.AddPass<GemmBroadcastFoldingRewriter>();

  // Run conversion again, to catch those matrix multiplications which were not
  // rewritten into cuBLAS calls.
  GpuBfloat16Support bf16(/*supports_matrix_multiplication=*/false,
                          stream_exec);
  pipeline.AddPass<BFloat16Normalization>(&bf16);

  // Remove `f32 -> bf16 -> f32` casts inserted by bf16 normalization.
  if (debug_options.xla_gpu_simplify_all_fp_conversions())
    pipeline.AddPass<SimplifyFPConversions>();

  // Choose the fastest algorithm for each conv.
  //
  // We pick the algorithm before fusion so we can generate better HLO. After
  // GpuConvRewriter, our convolutions are CustomCalls which return a
  // tuple (conv_result, scratch_memory), and the each conv uses 0 bytes of
  // scratch:
  //
  //   customcall = (f32[...], f32[0])
  //   return gte(customcall, 0)
  //
  // The algorithm picker then chooses the best algorithm, and potentially
  // increases the scratch space.  It replaces customcall with new_tuple,
  // giving us the following:
  //
  //   new_customcall = (f32[...], f32[N])
  //   new_tuple = tuple(gte(new_customcall, 0), constant f32[0])
  //   return gte(new_tuple, 0)
  //
  // The new tuple and gte instructions then be simplified away, because
  // nobody is expected to use the scratch value.
  //
  // However, if we were to run GpuConvAlgorithmPicker after fusion
  // the gte(customcall, 0) would probably already be into a fusion node.  We
  // can't simplify across HloComputation boundaries, so in this case we
  // wouldn't be able to simplify away the new_tuple bits.
  pipeline.AddPass<GpuConvAlgorithmPicker>(stream_exec, device_allocator);

  // Clean up new_tuple described above.
  pipeline.AddPass<TupleSimplifier>();

  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return OkStatus();
}

StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunHloPasses");
  uint64_t start_usecs = tensorflow::Env::Default()->NowMicros();
  tensorflow::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tensorflow::profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(
      OptimizeHloModule(module.get(), stream_exec, options.device_allocator));

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

  uint64_t end_usecs = tensorflow::Env::Default()->NowMicros();

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
    const HloModule* hlo_module) {
  std::unique_ptr<StreamAssignment> stream_assignment =
      AssignStreams(*hlo_module);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuHloSchedule> hlo_schedule,
      GpuHloSchedule::Build(hlo_module, *stream_assignment, pointer_size_));

  auto buffer_size_bytes_function =
      [this](const BufferValue& buffer_value) -> int64_t {
    return GetSizeOfShape(buffer_value.shape(), pointer_size_);
  };

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          hlo_module, hlo_schedule->ConsumeHloOrdering(),
          buffer_size_bytes_function,
          /*color_alignment=*/
          [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
          /*allocate_buffers_for_constants=*/true,
          /*colorer=*/BufferAssigner::DefaultColorer(),
          /*must_not_live_out=*/{}, GetCanShareBuffer()));

  return std::move(assignment);
}

#if XLA_ENABLE_XLIR
static StatusOr<OwnedBefBuffer> LowerToBef(mlir::ModuleOp mlir_module,
                                           llvm::StringRef entry_function_name,
                                           llvm::ArrayRef<int64_t> buffer_sizes,
                                           HloModule* hlo_module) {
  // Forward collective permute attributes for use by the lowering pipeline.
  mlir::OpBuilder builder(mlir_module.getContext());
  mlir::IntegerAttr replica_count_attr =
      builder.getI64IntegerAttr(hlo_module->config().replica_count());
  mlir::IntegerAttr num_partitions_attr =
      builder.getI64IntegerAttr(hlo_module->config().num_partitions());
  mlir::func::FuncOp func =
      mlir_module.lookupSymbol<mlir::func::FuncOp>(entry_function_name);
  func->setAttr("replica_count", replica_count_attr);
  func->setAttr("num_partitions", num_partitions_attr);

  // LMHLO -> TFRT Dialect
  TF_RETURN_IF_ERROR(tensorflow::ConvertLmhloToTfrtGpuWithBinary(
      mlir_module, {entry_function_name.data(), entry_function_name.size()},
      buffer_sizes));

  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "tfrt_gpu", mlir_module);
  }

  // TFRT Dialect -> BEF
  std::string bef;
  llvm::raw_string_ostream bef_ostream(bef);
  if (tfrt::MLIRToBEFTranslate(mlir_module, bef_ostream).failed()) {
    return InternalError("Failed to lower TFRT Dialect to BEF.");
  }

  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "", "bef", bef);
  }

  auto ptr = static_cast<uint8_t*>(
      tfrt::AlignedAlloc(tfrt::GetRequiredBefAlignment(), bef.size()));
  std::copy(bef.begin(), bef.end(), ptr);
  return OwnedBefBuffer(ptr, {bef.size()});
}

static StatusOr<OwnedJitRtProgram> LowerToJitRt(
    mlir::ModuleOp mlir_module, llvm::StringRef entry_function_name,
    llvm::ArrayRef<int64_t> buffer_sizes, HloModule* hlo_module) {
  // Forward collective (NCCL) attributes for use by the lowering pipeline.
  mlir::OpBuilder builder(mlir_module.getContext());
  mlir::IntegerAttr replica_count_attr =
      builder.getI64IntegerAttr(hlo_module->config().replica_count());
  mlir::IntegerAttr num_partitions_attr =
      builder.getI64IntegerAttr(hlo_module->config().num_partitions());
  mlir::func::FuncOp func =
      mlir_module.lookupSymbol<mlir::func::FuncOp>(entry_function_name);
  func->setAttr("replica_count", replica_count_attr);
  func->setAttr("num_partitions", num_partitions_attr);

  // Lower LMHLO operations to the JitRt compatible custom calls.
  TF_RETURN_IF_ERROR(tensorflow::ConvertLmhloToJitRt(
      mlir_module, {entry_function_name.data(), entry_function_name.size()},
      buffer_sizes));
  // Serialize module to pass it to GpuExecutable for compilation.
  std::string serialized_module;
  llvm::raw_string_ostream os(serialized_module);
  mlir_module.print(os);

  // TODO(b/232033540): Pass MLIR module directly to JitRt to instantiate an
  // executable, without forcing serialization.
  return std::make_unique<GpuExecutable::JitRtProgram>(
      entry_function_name.str(), os.str(), buffer_sizes.vec(),
      hlo_module->config().debug_options());
}
#endif  // XLA_ENABLE_XLIR

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
  std::variant<OwnedThunkSchedule, OwnedBefBuffer, OwnedJitRtProgram>
      executable;
  GpuExecutable::OwnedGpuContextCache gpu_ctx_cache;
  EntryFunctionAttributes entry_func_attrs;
  std::vector<GpuExecutable::ConstantInfo> constants;
  OutputInfoMap output_info;
  Shape output_shape;
  std::string module_name;
};

// The order of `thunk_sequence` corresponds to
// `hlo_schedule->ThunkLaunchOrder()`.
static Status CompileModuleToLlvmIrImpl(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, const se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    int pointer_size, CompileModuleResults* results,
    se::StreamExecutor* stream_exec = nullptr) {
  results->llvm_module = std::make_unique<llvm::Module>("", *llvm_context);
  results->llvm_module->setTargetTriple(target_triple);
  results->llvm_module->setDataLayout(data_layout);

  std::unique_ptr<StreamAssignment> stream_assignment =
      AssignStreams(*hlo_module);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuHloSchedule> hlo_schedule,
      GpuHloSchedule::Build(hlo_module, *stream_assignment, pointer_size));

  auto buffer_size_bytes_function =
      [pointer_size](const BufferValue& buffer_value) -> int64_t {
    return GetSizeOfShape(buffer_value.shape(), pointer_size);
  };

  TF_ASSIGN_OR_RETURN(
      results->buffer_assignment,
      BufferAssigner::Run(
          hlo_module, hlo_schedule->ConsumeHloOrdering(),
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
                                      "_gpu_after_optimizations"));

  uint64_t start_usecs = tensorflow::Env::Default()->NowMicros();
  mlir::MLIRContext mlir_context;
  mlir_context
      .loadDialect<mlir::lmhlo::LmhloDialect, mlir::mhlo::MhloDialect,
                   mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                   mlir::lmhlo_gpu::LmhloGpuDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::ModuleOp::create(mlir::Builder(&mlir_context).getUnknownLoc());

  TF_RETURN_IF_ERROR(
      HloToLhloModule(*results->buffer_assignment, *hlo_module, *mlir_module));

  results->module_name = mlir::GetNameFromLoc(mlir_module->getLoc());

  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "lmhlo", mlir_module.get());
  }

  auto entry_function = mlir::cast<mlir::func::FuncOp>(
      mlir_module->lookupSymbol(hlo_module->entry_computation()->name()));

  TF_RETURN_IF_ERROR(GetMlirAllocationInfo(
      entry_function, &results->allocations, &results->output_info,
      &results->output_shape, &results->entry_func_attrs));

  if (hlo_module->config().debug_options().xla_gpu_enable_mlir_lowering()) {
    mlir::PassManager pm(&mlir_context);
    pm.addPass(mlir::createGpuFusionRewritePass());
    if (failed(pm.run(mlir_module.get()))) {
      return InternalError("Failed to run gpu-fusion-rewrite pass");
    }
  }

  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr, platform_name,
      gpu_device_info, cuda_compute_capability, rocm_compute_capability,
      &mlir_context, results->llvm_module.get());

  ir_emitter_context.set_allocations(results->allocations);

  TF_ASSIGN_OR_RETURN(
      auto ir_emitter,
      IrEmitterUnnested::Create(hlo_module->config(), &ir_emitter_context));

  {
    XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunBackend - IR emission");

    TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(&entry_function.getBody()));

    bool supports_runtime_managed_constants =
        !IsBefThunkEnabled(hlo_module->config()) &&
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
    uint64_t end_usecs = tensorflow::Env::Default()->NowMicros();

    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    RecordHloToLlvmDuration(end_usecs - start_usecs);
  }

#if XLA_ENABLE_XLIR
  if (IsBefExecutableEnabled(hlo_module->config())) {
    std::vector<int64_t> buffer_sizes;
    llvm::transform(
        results->allocations, std::back_inserter(buffer_sizes),
        [](const BufferAllocation& allocation) { return allocation.size(); });
    TF_ASSIGN_OR_RETURN(results->executable,
                        LowerToBef(*mlir_module, entry_function.getName(),
                                   buffer_sizes, hlo_module));
    if (stream_exec) {
      auto& bef_buffer = std::get<OwnedBefBuffer>(results->executable);
      llvm::ArrayRef<uint8_t> bef_array(bef_buffer.get(),
                                        bef_buffer.get_deleter().size);
      TF_ASSIGN_OR_RETURN(results->gpu_ctx_cache,
                          GpuExecutable::CreatePreloadedGpuContextCache(
                              bef_array, stream_exec));
    }
    return OkStatus();
  }

  if (IsJitRtExecutableEnabled(hlo_module->config())) {
    std::vector<int64_t> buffer_sizes;
    llvm::transform(
        results->allocations, std::back_inserter(buffer_sizes),
        [](const BufferAllocation& allocation) { return allocation.size(); });
    TF_ASSIGN_OR_RETURN(results->executable,
                        LowerToJitRt(*mlir_module, entry_function.getName(),
                                     buffer_sizes, hlo_module));
    return OkStatus();
  }
#endif  // XLA_ENABLE_XLIR

  results->executable =
      std::make_unique<ThunkSchedule>(ir_emitter->ConsumeThunkSequence());
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
                                   se::StreamExecutor* stream_exec,
                                   const CompileOptions& options,
                                   const HloModule* debug_module) {
  using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;

  const auto compile_single_module =
      [this, stream_exec, &module_config, debug_module](
          llvm::Module* llvm_module, bool relocatable,
          std::optional<int> shard_number) -> StatusOr<BackendCompileResult> {
    {
      XLA_SCOPED_LOGGING_TIMER(
          "GpuCompiler::RunBackend - Running LLVM verifier");

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
    GpuVersion gpu_version = GetGpuVersion(stream_exec);
    StatusOr<std::pair<std::string, std::vector<uint8_t>>> result =
        CompileTargetBinary(module_config, llvm_module, gpu_version,
                            stream_exec, relocatable, debug_module);

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

  tensorflow::thread::ThreadPool* thread_pool;
  std::optional<tensorflow::thread::ThreadPool> overriding_thread_pool;
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
          tensorflow::Env::Default(), "",
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
  if (this->LinkModules(stream_exec, {}).status().code() ==
      tensorflow::error::Code::UNIMPLEMENTED) {
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

  llvm::SplitModule(
      *llvm_module,
      std::max<unsigned>(
          1, std::min<unsigned>(thread_pool->NumThreads(), num_functions)),
      [&](std::unique_ptr<llvm::Module> module) {
        llvm_modules.push_back(std::move(module));
      },
      /*PreserveLocals=*/true);

  std::vector<StatusOr<BackendCompileResult>> compile_results(
      llvm_modules.size());
  tensorflow::BlockingCounter counter(llvm_modules.size());
  for (int i = 0; i < llvm_modules.size(); i++) {
    thread_pool->Schedule(
        [&compile_results, compile_single_module, i, &llvm_modules, &counter] {
          llvm::Module* original_module = llvm_modules[i].get();
          llvm::LLVMContext context;
          std::string buffer;
          llvm::raw_string_ostream error(buffer);

          std::unique_ptr<llvm::Module> new_llvm_module;
          // Switch to a new context by dumping and re-parsing LLVM IR. Each
          // thread has its own context to avoid race conditions.
          {
            std::string ir;
            {
              llvm::raw_string_ostream os(ir);
              original_module->print(os, nullptr);
            }
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
      this->LinkModules(stream_exec, std::move(submodule_compile_results));
  if (!maybe_backend_result.ok()) {
    LOG(ERROR) << "The CUDA linking API did not work. Please use "
                  "XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 to "
                  "bypass it, but expect to get longer compilation time due to "
                  "the lack of multi-threading.";
    return maybe_backend_result.status();
  }

  return std::make_pair(ptx_snippets, std::move(*maybe_backend_result));
}

StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  VLOG(1) << "Starting to compile HLO module " << module->name();
  XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunBackend");
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

  TF_RET_CHECK(stream_exec != nullptr);

  llvm::LLVMContext llvm_context;

  GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);

  if (module->config().hlo_profiling_enabled() || VLOG_IS_ON(1)) {
    HloCostAnalysis::Options options{ShapeSizeBytesFunction()};
    options.set_bytes_per_second(
        stream_exec->GetDeviceDescription().memory_bandwidth());
    GpuHloCostAnalysis cost_analysis(options);
    TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&cost_analysis));
    VLOG(1) << "HLO memory read+written: "
            << tensorflow::strings::HumanReadableNumBytes(
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
        llvm_ir::DumpModuleToString(*compile_module_results.llvm_module);
  }

  llvm_ir::DumpIrIfEnabled(*module, *compile_module_results.llvm_module,
                           /*optimized=*/false);

  using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;
  TF_ASSIGN_OR_RETURN(
      BackendCompileResult backend_result,
      CompileToTargetBinary(module->config(),
                            std::move(compile_module_results.llvm_module),
                            stream_exec, options, module.get()));
  if (DumpingEnabledForHloModule(*module) &&
      std::holds_alternative<OwnedThunkSchedule>(
          compile_module_results.executable)) {
    const ThunkSchedule& thunk_schedule =
        *std::get<OwnedThunkSchedule>(compile_module_results.executable);
    DumpToFileInDirOrStdout(*module, "", "thunk_schedule",
                            thunk_schedule.ToString());
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
           std::move(module),
           std::move(compile_module_results.gpu_ctx_cache)}));
  if (embed_ir_in_executable) {
    DCHECK_NE("", ir_module_string_before_opt);
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }

  // Dump computation proto state and buffer assignment for debug and test, if
  // dump or embed_ir_in_executable is enabled.
  if (embed_ir_in_executable ||
      DumpingEnabledForHloModule(gpu_executable->module())) {
    auto hlo_proto = std::make_unique<HloProto>();
    if (hlo_proto_) {
      *hlo_proto = *hlo_proto_;
    } else {
      *hlo_proto->mutable_hlo_module() = gpu_executable->module().ToProto();
    }
    *hlo_proto->mutable_buffer_assignment() = buffer_assignment->ToProto();
    gpu_executable->set_hlo_proto(std::move(hlo_proto));
  }
  gpu_executable->set_debug_info(buffer_assignment->GetStats().ToString());
  return static_cast<std::unique_ptr<Executable>>(std::move(gpu_executable));
}

StatusOr<std::unique_ptr<AotCompilationResult>>
GpuCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<AotCompilationResult> aot_result,
      GpuAotCompilationResult::FromString(serialized_aot_result));
  return aot_result;
}

GpuDeviceInfo GetGpuDeviceInfo(se::StreamExecutor* stream_exec) {
  GpuDeviceInfo gpu_device_info;
  gpu_device_info.threads_per_block_limit =
      stream_exec->GetDeviceDescription().threads_per_block_limit();
  gpu_device_info.threads_per_warp =
      stream_exec->GetDeviceDescription().threads_per_warp();
  gpu_device_info.shared_memory_per_block =
      stream_exec->GetDeviceDescription().shared_memory_per_block();
  gpu_device_info.threads_per_core_limit =
      stream_exec->GetDeviceDescription().threads_per_core_limit();
  gpu_device_info.core_count = stream_exec->GetDeviceDescription().core_count();
  gpu_device_info.block_dim_limit_x =
      stream_exec->GetDeviceDescription().block_dim_limit().x;
  gpu_device_info.block_dim_limit_y =
      stream_exec->GetDeviceDescription().block_dim_limit().y;
  gpu_device_info.block_dim_limit_z =
      stream_exec->GetDeviceDescription().block_dim_limit().z;
  return gpu_device_info;
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
GpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& options) {
#if XLA_ENABLE_XLIR
  CHECK(options.PlatformId() == se::cuda::kCudaPlatformId);
  CHECK(options.executor() != nullptr);
  auto stream_exec = options.executor();

  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();
  std::vector<std::unique_ptr<AotCompilationResult>> results;

  for (const auto& module : modules) {
    XLA_SCOPED_LOGGING_TIMER(
        "GpuCompiler::CompileAheadOfTime - compiling one HloModule");

    if (!options.run_backend_only()) {
      uint64_t start_usecs = tensorflow::Env::Default()->NowMicros();
      tensorflow::profiler::TraceMe activity(
          [&] { return absl::StrCat("HLO Transforms:", module->name()); },
          tensorflow::profiler::TraceMeLevel::kInfo);
      TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(), stream_exec,
                                           options.device_allocator()));

      TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

      uint64_t end_usecs = tensorflow::Env::Default()->NowMicros();

      // This won't record values for calls that error out (because if they
      // error out we have no way of telling how far through the process we
      // got).
      RecordHloPassesDuration(end_usecs - start_usecs);
    }

    std::string slow_compilation_msg =
        absl::StrCat("Compiling module ", module->name());
    auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

    llvm::LLVMContext llvm_context;

    GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);

    if (module->config().hlo_profiling_enabled() || VLOG_IS_ON(1)) {
      HloCostAnalysis::Options options{ShapeSizeBytesFunction()};
      options.set_bytes_per_second(
          stream_exec->GetDeviceDescription().memory_bandwidth());
      GpuHloCostAnalysis cost_analysis(options);
      TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&cost_analysis));
      VLOG(1) << "HLO memory read+written for " << module->name() << " "
              << tensorflow::strings::HumanReadableNumBytes(
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
        GetCanShareBuffer(), pointer_size_, &compile_module_results));

    if (!std::holds_alternative<OwnedBefBuffer>(
            compile_module_results.executable)) {
      return FailedPrecondition("Expected BefBuffer is not supplied.");
    }
    const auto& bef_buffer =
        std::get<OwnedBefBuffer>(compile_module_results.executable);
    const std::string bef(reinterpret_cast<char*>(bef_buffer.get()),
                          bef_buffer.get_deleter().size);

    results.emplace_back(std::make_unique<GpuAotCompilationResult>(
        module->ToProto(), bef, compile_module_results.entry_func_attrs));
  }

  return std::move(results);
#else   // XLA_ENABLE_XLIR
  return FailedPrecondition("Not built with XLA_ENABLE_XLIR");
#endif  // XLA_ENABLE_XLIR
}

HloCostAnalysis::ShapeSizeFunction GpuCompiler::ShapeSizeBytesFunction() const {
  // Capture just the pointer size, not the entire GpuCompiler object.
  return [pointer_size = pointer_size_](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
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
    for (const mlir::NamedAttribute& attr : func.getArgAttrs(i)) {
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
      !IsBefThunkEnabled(module_config) &&
      // TODO(b/218907125): Implement this feature for ROCm as well.
      compiler->PlatformId() != se::rocm::kROCmPlatformId;
  if (supports_runtime_managed_constants) {
    // Remove these globals from the generated code to indicate that XLA is
    // responsible for allocating and initializing them.
    RemoveUnusedAndUninitializedGlobals(ir_emitter_context->llvm_module(),
                                        ir_emitter_context->constants());
  }

  auto thunk_schedule =
      std::make_unique<ThunkSchedule>(ir_emitter->ConsumeThunkSequence());

  using BackendCompileResult = std::pair<std::string, std::vector<uint8_t>>;
  TF_ASSIGN_OR_RETURN(BackendCompileResult backend_result,
                      compiler->CompileToTargetBinary(
                          module_config, std::move(llvm_module), stream_exec,
                          options, /*debug_module=*/nullptr));

  GpuVersion gpu_version = compiler->GetGpuVersion(stream_exec);
  return GpuExecutable::Create(
      {std::move(backend_result.first), std::move(backend_result.second),
       gpu_version, std::move(thunk_schedule), entry_func_attrs,
       std::move(ir_emitter_context->constants()), std::move(output_info),
       module_name, output_shape, std::move(allocations)});
}

}  // namespace gpu
}  // namespace xla

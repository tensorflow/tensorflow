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
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/all_gather_broadcast_reorder.h"
#include "tensorflow/compiler/xla/service/all_gather_combiner.h"
#include "tensorflow/compiler/xla/service/all_gather_decomposer.h"
#include "tensorflow/compiler/xla/service/all_reduce_combiner.h"
#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"
#include "tensorflow/compiler/xla/service/all_to_all_decomposer.h"
#include "tensorflow/compiler/xla/service/async_collective_creator.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/collectives_schedule_linearizer.h"
#include "tensorflow/compiler/xla/service/comparison_expander.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/convolution_4d_expander.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/dynamic_padder.h"
#include "tensorflow/compiler/xla/service/eigh_expander.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/gpu/alias_passthrough_params.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_reduce_scatter_creator.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_sanitize_constant_names.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_scatter_expander.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_input_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_dimension_grouper.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_splitter.h"
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
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/logistic_expander.h"
#include "tensorflow/compiler/xla/service/loop_schedule_linearizer.h"
#include "tensorflow/compiler/xla/service/operand_upcaster.h"
#include "tensorflow/compiler/xla/service/qr_expander.h"
#include "tensorflow/compiler/xla/service/real_imag_expander.h"
#include "tensorflow/compiler/xla/service/reduce_scatter_combiner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/result_caster.h"
#include "tensorflow/compiler/xla/service/rng_bit_generator_expander.h"
#include "tensorflow/compiler/xla/service/rng_expander.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/sharding_remover.h"
#include "tensorflow/compiler/xla/service/slice_sinker.h"
#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
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
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/env_var.h"

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
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllReduceStart:
      case HloOpcode::kAllReduceDone:
      case HloOpcode::kReduceScatter:
      case HloOpcode::kAllToAll:
      case HloOpcode::kBitcast:
      case HloOpcode::kCollectivePermute:
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

}  // end anonymous namespace

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
  {
    HloPassPipeline pipeline("optimization");
    pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                              /*allow_mixed_precision=*/false);
    pipeline.AddPass<AllToAllDecomposer>();

    OpExpanderPass::PatternExtraFilter upcaster_filter =
        [&](const HloInstruction* instr) {
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

    pipeline.AddPass<GpuScatterExpander>();
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

    // If cudnn batchnorms are enabled, rewrite batchnorm HLOs to cudnn calls
    // where possible.  Not every batchnorm op can be implemented as a call to
    // cudnn, so decompose any remaining batchnorm ops into a soup of HLOs.
    if (hlo_module->config().debug_options().xla_gpu_use_cudnn_batchnorm()) {
      // Since BatchNorm inference is essentially pointwise operations, it is
      // always advantageous to use kernel fusion rather than cudnn.
      pipeline.AddPass<BatchNormExpander>(
          /*rewrite_training_op=*/false,
          /*rewrite_inference_op=*/true,
          /*rewrite_grad_op=*/false);
      pipeline.AddPass<CudnnBatchNormRewriter>();
    }
    pipeline.AddPass<BatchNormExpander>(
        /*rewrite_training_op=*/true,
        /*rewrite_inference_op=*/true,
        /*rewrite_grad_op=*/true);

    pipeline.AddPass<LogisticExpander>(
        /*expansion_type=*/LogisticExpansionType::kExp);
    pipeline.AddPass<ConditionalCanonicalizer>();
    pipeline.AddPass<DynamicPadder>();

    // Build simplification pipeline.  The passes in here are run to a fixed
    // point.
    [&pipeline =
         pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
      pipeline.AddInvariantCheckerDebug<HloVerifier>(
          /*layout_sensitive=*/false,
          /*allow_mixed_precision=*/false);

      // BatchNormExpander can create zero-sized ops, so zero-sized HLO
      // elimination has to come after that pass.
      pipeline.AddPass<ZeroSizedHloElimination>();

      pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
      pipeline.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);

      AlgebraicSimplifierOptions options;
      // When transposes appear in a fusion node, we can easily adjust the
      // multi-dimensional index to create the one needed for the operand.
      // This is not as easy with bitcasts, because we don't have the
      // information readily available which dimensions are permuted. In
      // addition to that, if we have a transpose and a reshape next to each
      // other, they will both be replaced by a bitcast, and we replace
      // bitcast(bitcast) with one bitcast. This leads to having to
      // linearize and then delinearize the index.
      options.set_replace_transpose_with_bitcast(false);
      options.set_enable_conv_operand_swap(false);
      pipeline.AddPass<AlgebraicSimplifier>(options);
      // AlgebraicSimplifier may add contracting dimensions to a dot.
      pipeline.AddPass<DotDecomposer>();
      pipeline.AddPass<SortSimplifier>();
      pipeline.AddPass<TupleSimplifier>();
      pipeline.AddPass<WhileLoopConstantSinking>();
      pipeline.AddPass<WhileLoopSimplifier>();

      // TODO(b/134075051): Re-enable after b/134075051 is fixed.
      // pipeline.AddPass<SliceSinker>();

      pipeline.AddPass<HloDCE>();
      pipeline.AddPass<ReshapeMover>();
      pipeline.AddPass<HloConstantFolding>();
      pipeline.AddPass<ConditionalSimplifier>();
      pipeline.AddPass<RealImagExpander>();
    }();

    pipeline.AddPass<TransposeFolding>(
        [](const HloInstruction& dot,
           const TransposeFolding::OperandIndices& candidate_operands) {
          return IsMatrixMultiplication(dot)
                     ? candidate_operands
                     : TransposeFolding::OperandIndices{};
        });
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    pipeline.AddPass<HloDCE>();

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

  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      spmd_pipeline.AddPass<ShardingPropagation>(/*is_spmd=*/true);
      spmd_pipeline.AddPass<GpuSpmdPartitioner>(
          num_partitions, hlo_module->config().replica_count());
    } else {
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }

  // Optimize collectives generated by SPMD partitioning. Enable these passes
  // otherwise as well so that all collectives can get these optimizations.
  {
    HloPassPipeline collectives_pipeline("collective-optimizations");
    collectives_pipeline.AddPass<ReduceScatterCreator>();
    collectives_pipeline.AddPass<AllReduceReassociate>();

    // Run algebraic simplifier to reshape(broadcast) into a broadcast when
    // the reshape is just adding a unit dimension. This will help with the
    // AllGatherBroadcastReorder pass.
    AlgebraicSimplifierOptions options;
    options.set_replace_transpose_with_bitcast(false);
    options.set_enable_conv_operand_swap(false);
    collectives_pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

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
        hlo_module->mutable_entry_computation_layout(),
        LayoutAssignment::InstructionCanChangeLayout, stream_exec,
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
    /* TODO(b/117531509): Use LayoutAssignment::InstructionCanChangeLayout after
     * fixing the ticket. */
    fusion.AddInvariantCheckerDebug<HloVerifier>(
        /*layout_sensitive=*/true,
        /*allow_mixed_precision=*/false,
        LayoutAssignment::InstructionCanChangeLayout);
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
    horizontal_fusion.AddPass<FusionBitcastLift>();
    horizontal_fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                                      /*only_fusion_computations=*/true);
    horizontal_fusion.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(horizontal_fusion.Run(hlo_module).status());
  }

  {
    HloPassPipeline pipeline("post-fusion optimization");
    pipeline.AddPass<AllGatherCombiner>(
        /*combine_threshold_in_bytes=*/1024 * 1024 * 1024,
        /*combine_threshold_count=*/256);
    pipeline.AddPass<AllReduceCombiner>(
        /*combine_threshold_in_bytes=*/30 * 1024 * 1024,
        /*combine_threshold_count=*/256);
    pipeline.AddPass<ReduceScatterCombiner>(
        /*combine_threshold_in_bytes=*/30 * 1024 * 1024,
        /*combine_threshold_count=*/256);

    if (hlo_module->config()
            .debug_options()
            .xla_gpu_enable_async_all_reduce()) {
      pipeline.AddPass<AsyncCollectiveCreator>(/*convert_all_reduce=*/true,
                                               /*convert_all_gather=*/false);
    }

    pipeline.AddPass<CollectivesScheduleLinearizer>();

    // Now we allow replacing any transposes outside of fusions with bitcasts.
    AlgebraicSimplifierOptions options;
    options.set_is_layout_sensitive(true);
    options.set_enable_conv_operand_swap(false);
    pipeline.AddPass<AlgebraicSimplifier>(options);

    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  return Status::OK();
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
  /* TODO(b/117531509): Use LayoutAssignment::InstructionCanChangeLayout after
   * fixing the ticket. */
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/true,
      /*allow_mixed_precision=*/false,
      LayoutAssignment::InstructionCanChangeLayout);

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
  pipeline.AddPass<GpuCopyInsertion>(GetCanShareBuffer());
  pipeline.AddPass<GpuSanitizeConstantNames>();
  return pipeline.Run(hlo_module).status();
}

Status GpuCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  HloPassPipeline pipeline("post-layout_assignment");
  /* TODO(b/117531509): Use LayoutAssignment::InstructionCanChangeLayout after
   * fixing the ticket. */
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/true,
      /*allow_mixed_precision=*/false,
      LayoutAssignment::InstructionCanChangeLayout);

  pipeline.AddPass<ReductionDegenerateDimRemover>();
  pipeline.AddPass<ReductionLayoutNormalizer>();
  pipeline.AddPass<ReductionDimensionGrouper>();
  pipeline.AddPass<HloPassFix<ReductionSplitter>>();

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  // When transposes appear in a fusion node, we can easily adjust the
  // multi-dimensional index to create the one needed for the operand. This
  // is not as easy with bitcasts, because we don't have the information
  // readily available which dimensions are permuted. In addition to that,
  // if we have a transpose and a reshape next to each other, they will both
  // be replaced by a bitcast, and we replace bitcast(bitcast) with one
  // bitcast. This leads to having to linearize and then delinearize the
  // index.
  options.set_replace_transpose_with_bitcast(false);
  options.set_enable_conv_operand_swap(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

  if (RequireDeterminism(hlo_module->config()) ||
      hlo_module->config().debug_options().xla_gpu_deterministic_reductions()) {
    pipeline.AddPass<HloPassFix<GpuTreeReductionRewriter>>();
  }

  // GemmRewriter assumes that all transposes are folded into gemms, but,
  // since commit 7d529df, this is not always true at this point.
  // Therefore, rerun transpose folding.
  pipeline.AddPass<TransposeFolding>(
      [](const HloInstruction& dot,
         const TransposeFolding::OperandIndices& candidate_operands) {
        return IsMatrixMultiplication(dot) ? candidate_operands
                                           : TransposeFolding::OperandIndices{};
      },
      TransposeFolding::NeverFoldTranspose);
  // Rewrite GEMMs into custom calls.
  pipeline.AddPass<GemmRewriter>();

  // Run conversion again, to catch those matrix multiplications which were not
  // rewritten into cuBLAS calls.
  GpuBfloat16Support bf16(/*supports_matrix_multiplication=*/false,
                          stream_exec);
  pipeline.AddPass<BFloat16Normalization>(&bf16);

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

  return Status::OK();
}

StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunHloPasses");
  tensorflow::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tensorflow::profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(
      OptimizeHloModule(module.get(), stream_exec, options.device_allocator));

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));

  return std::move(module);
}

static absl::optional<bool> DummyCanShareBufferFunction(const HloInstruction*,
                                                        const HloInstruction*,
                                                        const ShapeIndex&) {
  return absl::nullopt;
}

StatusOr<
    std::tuple<std::unique_ptr<HloModule>, std::unique_ptr<BufferAssignment>>>
GpuCompiler::RunHloPassesAndBufferAssignement(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* executor,
    bool optimize, const CompileOptions& options) {
  if (optimize) {
    TF_ASSIGN_OR_RETURN(hlo_module,
                        RunHloPasses(std::move(hlo_module), executor, options));
  }

  std::unique_ptr<StreamAssignment> stream_assignment =
      AssignStreams(*hlo_module);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GpuHloSchedule> hlo_schedule,
                      GpuHloSchedule::Build(hlo_module.get(),
                                            *stream_assignment, pointer_size_));

  auto buffer_size_bytes_function =
      [this](const BufferValue& buffer_value) -> int64 {
    return GpuCompiler::GetSizeOfShape(buffer_value.shape(), pointer_size_);
  };

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          hlo_module.get(), hlo_schedule->ConsumeHloOrdering(),
          buffer_size_bytes_function,
          /*color_alignment=*/
          [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
          /*allocate_buffers_for_constants=*/true,
          /*colorer=*/BufferAssigner::DefaultColorer(),
          /*must_not_live_out=*/{}, GetCanShareBuffer()));

  return std::make_tuple(std::move(hlo_module), std::move(assignment));
}

using OutputInfoMap =
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;
static Status GetMlirAllocationInfo(mlir::FuncOp func,
                                    std::vector<BufferAllocation>* allocations,
                                    OutputInfoMap* output_info,
                                    Shape* output_shape);

struct CompileModuleResults {
  std::unique_ptr<llvm::Module> llvm_module;
  std::unique_ptr<BufferAssignment> buffer_assignment;
  std::vector<BufferAllocation> allocations;
  std::unique_ptr<ThunkSchedule> thunk_schedule;
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
    const std::string& platform_name, GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    int pointer_size, const HloProfileIndexMap* profile_index_map,
    CompileModuleResults* results) {
  results->llvm_module = absl::make_unique<llvm::Module>("", *llvm_context);
  results->llvm_module->setTargetTriple(target_triple);
  results->llvm_module->setDataLayout(data_layout);

  std::unique_ptr<StreamAssignment> stream_assignment =
      AssignStreams(*hlo_module);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuHloSchedule> hlo_schedule,
      GpuHloSchedule::Build(hlo_module, *stream_assignment, pointer_size));

  auto buffer_size_bytes_function =
      [pointer_size](const BufferValue& buffer_value) -> int64 {
    return GpuCompiler::GetSizeOfShape(buffer_value.shape(), pointer_size);
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

  VLOG(1) << "Buffer Assignment Stats "
          << results->buffer_assignment->GetStats().ToString();
  DumpHloModuleIfEnabled(*hlo_module, *results->buffer_assignment,
                         absl::StrCat("sm_", cuda_compute_capability.ToString(),
                                      "_gpu_after_optimizations"));

  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::lmhlo::LmhloDialect, mlir::mhlo::MhloDialect,
                           mlir::StandardOpsDialect,
                           mlir::lmhlo_gpu::LmhloGpuDialect>();
  mlir::OwningModuleRef mlir_module =
      mlir::ModuleOp::create(mlir::Builder(&mlir_context).getUnknownLoc());

  TF_RETURN_IF_ERROR(
      HloToLhloModule(*results->buffer_assignment, *hlo_module, *mlir_module));

  results->module_name = mlir::GetNameFromLoc(mlir_module->getLoc());

  llvm_ir::DumpIrIfEnabled(mlir_module.get(), hlo_module->unique_id(),
                           hlo_module->config().debug_options());

  auto entry_function = mlir::cast<mlir::FuncOp>(
      mlir_module->lookupSymbol(hlo_module->entry_computation()->name()));

  TF_RETURN_IF_ERROR(
      GetMlirAllocationInfo(entry_function, &results->allocations,
                            &results->output_info, &results->output_shape));

  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr, platform_name,
      gpu_device_info, cuda_compute_capability, profile_index_map,
      &mlir_context, results->llvm_module.get());

  ir_emitter_context.set_allocations(results->allocations);

  TF_ASSIGN_OR_RETURN(
      auto ir_emitter,
      IrEmitterUnnested::Create(hlo_module->config(), &ir_emitter_context));

  {
    XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunBackend - IR emission");

    TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(&entry_function.body()));

    results->thunk_schedule =
        absl::make_unique<ThunkSchedule>(ir_emitter->ConsumeThunkSequence());

    results->constants = std::move(ir_emitter_context.constants());
  }

  return Status::OK();
}

static void NullDiagnosticHandler(const llvm::DiagnosticInfo& diag_info,
                                  void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info.print(diagnostic_printer);

  VLOG(1) << error_string;
}

StatusOr<std::pair<std::string, std::vector<uint8>>>
GpuCompiler::CompileToTargetBinary(const HloModuleConfig& module_config,
                                   std::unique_ptr<llvm::Module> llvm_module,
                                   se::StreamExecutor* stream_exec,
                                   const CompileOptions& options,
                                   const HloModule* debug_module) {
  using BackendCompileResult = std::pair<std::string, std::vector<uint8>>;

  const auto compile_single_module =
      [this, stream_exec, &module_config, debug_module](
          llvm::Module* llvm_module, bool relocatable,
          absl::optional<int> shard_number) -> StatusOr<BackendCompileResult> {
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
    StatusOr<std::pair<std::string, std::vector<uint8>>> result =
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
  absl::optional<tensorflow::thread::ThreadPool> overriding_thread_pool;
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
                                 /*shard_number=*/absl::nullopt);
  }

  // Test whether LinkModules is supported.
  if (this->LinkModules(stream_exec, {}).status().code() ==
      tensorflow::error::Code::UNIMPLEMENTED) {
    return compile_single_module(llvm_module.get(), /*relocatable=*/false,
                                 /*shard_number=*/absl::nullopt);
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
      *llvm_module.get(),
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
  std::vector<std::vector<uint8>> submodule_compile_results;
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
  XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunBackend");
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

  TF_RET_CHECK(stream_exec != nullptr);

  llvm::LLVMContext llvm_context;

  GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);

  std::unique_ptr<HloProfileIndexMap> profile_index_map;
  std::unique_ptr<HloProfilePrinterData> profile_printer;

  if (module->config().hlo_profiling_enabled() || VLOG_IS_ON(1)) {
    HloCostAnalysis cost_analysis(ShapeSizeBytesFunction());
    cost_analysis.set_bytes_per_second(
        stream_exec->GetDeviceDescription().memory_bandwidth());
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
      stream_exec->platform()->Name(), gpu_device_info,
      stream_exec->GetDeviceDescription().cuda_compute_capability(),
      GetCanShareBuffer(), pointer_size_, profile_index_map.get(),
      &compile_module_results));

  if (user_pre_optimization_hook_) {
    user_pre_optimization_hook_(*compile_module_results.llvm_module);
  }
  string ir_module_string_before_opt;
  const bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  if (embed_ir_in_executable) {
    ir_module_string_before_opt =
        llvm_ir::DumpModuleToString(*compile_module_results.llvm_module);
  }

  llvm_ir::DumpIrIfEnabled(*module, *compile_module_results.llvm_module,
                           /*optimized=*/false);

  using BackendCompileResult = std::pair<std::string, std::vector<uint8>>;
  TF_ASSIGN_OR_RETURN(
      BackendCompileResult backend_result,
      CompileToTargetBinary(module->config(),
                            std::move(compile_module_results.llvm_module),
                            stream_exec, options, module.get()));
  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(*module, "", "thunk_schedule",
                            compile_module_results.thunk_schedule->ToString());
  }

  auto buffer_assignment_proto = std::make_unique<BufferAssignmentProto>(
      compile_module_results.buffer_assignment->ToProto());

  size_t profile_index = 0;
  if (profile_index_map) {
    profile_index =
        profile_index_map->GetProfileIndexFor(*module->entry_computation());
  }

  GpuVersion gpu_version = GetGpuVersion(stream_exec);
  auto* gpu_executable = new GpuExecutable(
      {std::move(backend_result.first), std::move(backend_result.second),
       gpu_version, std::move(compile_module_results.thunk_schedule),
       std::move(compile_module_results.constants),
       std::move(compile_module_results.output_info),
       compile_module_results.module_name, compile_module_results.output_shape,
       std::move(compile_module_results.allocations),
       std::move(buffer_assignment_proto), std::move(module), profile_index,
       std::move(profile_printer), std::move(profile_index_map)});
  if (embed_ir_in_executable) {
    DCHECK_NE("", ir_module_string_before_opt);
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }

  // Dump computation proto state and buffer assignment for debug and test, if
  // dump is enabled.
  if (DumpingEnabledForHloModule(gpu_executable->module())) {
    auto hlo_proto = absl::make_unique<HloProto>();
    *hlo_proto->mutable_hlo_module() = gpu_executable->module().ToProto();
    *hlo_proto->mutable_buffer_assignment() =
        compile_module_results.buffer_assignment->ToProto();
    gpu_executable->set_hlo_proto(std::move(hlo_proto));
  }
  gpu_executable->set_debug_info(
      compile_module_results.buffer_assignment->GetStats().ToString());
  return std::unique_ptr<Executable>(gpu_executable);
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
  return Unimplemented("not yet implemented: GpuCompiler::CompileAheadOfTime");
}

StatusOr<std::unique_ptr<llvm::Module>> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability, int pointer_size) {
  CompileModuleResults results;
  TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
      hlo_module, llvm_context, target_triple, data_layout, platform_name,
      gpu_device_info, cuda_compute_capability, DummyCanShareBufferFunction,
      pointer_size, /*profile_index_map=*/nullptr, &results));
  return std::move(results.llvm_module);
}

// Analyze the function signature to reconstruct a vector of BufferAllocation
// objects, as well as other output information.
//
// This function also serves as a half-baked verifier for function arg
// attributes, since a full verifier doens't exist yet.
static Status GetMlirAllocationInfo(mlir::FuncOp func,
                                    std::vector<BufferAllocation>* allocations,
                                    OutputInfoMap* output_info,
                                    Shape* output_shape) {
  CHECK(allocations->empty());
  allocations->reserve(func.getNumArguments());

  for (int i = 0; i < func.getNumArguments(); i++) {
    mlir::BlockArgument arg = func.getArgument(i);

    TF_RET_CHECK(arg.getType().isa<mlir::ShapedType>());
    mlir::ShapedType type = arg.getType().cast<mlir::ShapedType>();
    TF_ASSIGN_OR_RETURN(auto element_type_bytes,
                        GetElementTypeBytes(type.getElementType()));
    size_t size = type.getNumElements() * element_type_bytes;
    allocations->emplace_back(i, size, 0);
  }

  for (int i = 0; i < func.getNumArguments(); i++) {
    for (const mlir::NamedAttribute& attr : func.getArgAttrs(i)) {
      TF_RET_CHECK(attr.first == "lmhlo.params" ||
                   attr.first == "lmhlo.param_shape_index" ||
                   attr.first == "lmhlo.constant_name" ||
                   attr.first == "lmhlo.must_alias" ||
                   attr.first == "lmhlo.output_index");
    }
  }

  std::vector<std::pair<ShapeIndex, Shape>> sub_shapes;
  for (int i = 0; i < func.getNumArguments(); i++) {
    if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
      xla::ShapeIndex shape_index;
      if (auto shape_index_attr =
              func.getArgAttrOfType<mlir::DenseIntElementsAttr>(
                  i, "lmhlo.param_shape_index")) {
        for (const llvm::APInt& element : shape_index_attr) {
          shape_index.push_back(element.getSExtValue());
        }
      }
      allocations->at(i).set_entry_computation_parameter(
          param_attr.cast<mlir::IntegerAttr>().getInt(), shape_index,
          static_cast<bool>(func.getArgAttr(i, "lmhlo.output_index")));
    }
    // TODO(timshen): this information is redundant. This is here only for
    // smooth migration to LMHLO. Remove it.
    if (func.getArgAttr(i, "lmhlo.constant_name")) {
      allocations->at(i).set_constant(true);
    }
    if (auto output_index_attr = func.getArgAttr(i, "lmhlo.output_index")) {
      allocations->at(i).set_maybe_live_out(true);

      // Reconstruct a shape index from output_index.
      ShapeIndex shape_index;
      for (const llvm::APInt& element :
           output_index_attr.cast<mlir::DenseIntElementsAttr>()) {
        shape_index.push_back(element.getSExtValue());
      }
      auto& o = (*output_info)[shape_index];
      o.allocation_index = i;
      if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
        HloInputOutputAliasConfig::AliasKind kind =
            HloInputOutputAliasConfig::kMayAlias;
        if (func.getArgAttr(i, "lmhlo.must_alias")) {
          kind = HloInputOutputAliasConfig::kMustAlias;
        }
        o.alias_config.emplace(param_attr.cast<mlir::IntegerAttr>().getInt(),
                               ShapeIndex{}, kind);
      }
      if (func.getArgument(i).use_empty()) {
        o.passthrough = true;
      }

      mlir::BlockArgument arg = func.getArgument(i);
      sub_shapes.push_back(std::make_pair(shape_index, GetShape(arg)));
    }
  }
  // Expects result_xla_shape as a XLA shape in string form.
  //
  // The attribute is necessary, because GpuExecutable/ExecutionOutput supports
  // tuples / tree-like shapes, while the LMHLO argument list loses the tree
  // form.
  //
  // The string format is necessary since MLIR doesn't support XLA shape with
  // dynamic_dimension.
  //
  // TODO(timshen): now this field is mandatory. Make it optional for
  // non-GpuExecutable outputs.
  TF_ASSIGN_OR_RETURN(
      *output_shape,
      ParseShape(func->getAttrOfType<mlir::StringAttr>("result_xla_shape")
                     .getValue()
                     .str()));

  return Status::OK();
}

StatusOr<std::unique_ptr<Executable>> CompileLmhloToExecutable(
    GpuCompiler* compiler, mlir::ModuleOp module, std::string module_name,
    const HloModuleConfig& module_config,
    const Compiler::CompileOptions& options,
    absl::string_view entry_function_name, se::StreamExecutor* stream_exec,
    std::unique_ptr<llvm::Module> llvm_module,
    IrEmitterContext* ir_emitter_context) {
  mlir::FuncOp entry_function = mlir::cast<mlir::FuncOp>(module.lookupSymbol(
      llvm::StringRef(entry_function_name.data(), entry_function_name.size())));

  std::vector<BufferAllocation> allocations;
  OutputInfoMap output_info;
  Shape output_shape;
  TF_RETURN_IF_ERROR(GetMlirAllocationInfo(entry_function, &allocations,
                                           &output_info, &output_shape));

  TF_RET_CHECK(!allocations.empty());

  ir_emitter_context->set_allocations(allocations);

  TF_ASSIGN_OR_RETURN(auto ir_emitter, IrEmitterUnnested::Create(
                                           module_config, ir_emitter_context));
  TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(&entry_function.body()));

  auto thunk_schedule =
      absl::make_unique<ThunkSchedule>(ir_emitter->ConsumeThunkSequence());

  using BackendCompileResult = std::pair<std::string, std::vector<uint8>>;
  TF_ASSIGN_OR_RETURN(BackendCompileResult backend_result,
                      compiler->CompileToTargetBinary(
                          module_config, std::move(llvm_module), stream_exec,
                          options, /*debug_module=*/nullptr));

  GpuVersion gpu_version = compiler->GetGpuVersion(stream_exec);
  auto* gpu_executable = new GpuExecutable(
      {std::move(backend_result.first), std::move(backend_result.second),
       gpu_version, std::move(thunk_schedule),
       std::move(ir_emitter_context->constants()), std::move(output_info),
       module_name, output_shape, std::move(allocations)});
  return std::unique_ptr<Executable>(gpu_executable);
}

}  // namespace gpu
}  // namespace xla

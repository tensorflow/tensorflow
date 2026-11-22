/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_pipeline.h"

#include <cstdint>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/pass/hlo_pipeline_editor.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

constexpr int32_t kDefaultMaxFixedPointIterations = 25;

void AppendPreSPMDPartitionerPasses(
    HloPipelineEditor pipeline,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts) {
  pipeline.AppendPass("erase-xla-late-frontend-attributes");
  pipeline.AppendPass("cudnn-custom-call-converter");
  pipeline.AppendPass("composite-rewriter");
  pipeline.AppendPass("convert-memory-placement-to-internal-annotations");
  pipeline.AppendPass("flatten-call-graph");
  pipeline.AppendPass("call-inliner");
  // ZeroSizedHloElimination replaces zero-sized ops with constants and
  // GpuAlgebraicSimplifier folds those constants into users.
  pipeline.AppendPass("zero_sized_hlo_elimination");
  pipeline.AppendPass("gpu-algebraic-simplifier",
                      HloPassOptions(layout_insensitive_algsimp_opts));
  pipeline.AppendPass("conditional-canonicalizer");
  // The TopkDecomposer generates a compare op with order=TOTAL and must
  // run before the ComparisonExpander which rewrites such comparisons.
  pipeline.AppendPass("topk-decomposer", {{"only_topk_op", true}});
  // The SPMD partitioner would mess up the sort+slice structure, so we need to
  // rewrite Topk before that happens.
  pipeline.AppendPass("topk-rewriter");
}

void AppendSPMDSimplifyPasses(
    HloPipelineEditor spmd_simplify, const HloModuleConfig& config,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts) {
  spmd_simplify.AppendPass("gpu-algebraic-simplifier",
                           HloPassOptions(layout_insensitive_algsimp_opts));
  spmd_simplify.AppendPass("simplify-sorts");
  spmd_simplify.AppendPass("tuple-simplifier");
  spmd_simplify.AppendPass("scatter_expander",
                           {{"mode", "eliminate_simple_scatters"}});
  spmd_simplify.AppendPass("gather_expander",
                           {{"mode", "eliminate_simple_gathers"}});
  spmd_simplify.AppendPass("while-loop-constant-sinking");
  spmd_simplify.AppendPass("simplify-while-loops");
  spmd_simplify.AppendPass("reshape-mover",
                           {{"reshape_of_1d_broadcast_is_cheap", true}});
  // Run AlgebraicSimplifier directly before HloConstantFolding, because we
  // need to simplify DynamicSlice(Broadcast) away. Constant folding of
  // DynamicSlice can be quite costly, as the whole operand will be evaluated.
  // We run AlgebraicSimplifier as HloPassFix to make sure all simplifications
  // have been done before running HloConstantFolding. This is necessary
  // because simplifications create new instructions which may not be visited
  // in the same iteration of AlgebraicSimplifier.
  spmd_simplify.AppendPass("gpu-algebraic-simplifier",
                           HloPassOptions(layout_insensitive_algsimp_opts),
                           kDefaultMaxFixedPointIterations);
  spmd_simplify.AppendPass("constant_folding");
  spmd_simplify.AppendPass("simplify-conditional");
  if (!config.use_shardy_partitioner()) {
    spmd_simplify.AppendPass("dce");
  }
}

void AppendSPMDPartitionerPasses(
    HloPipelineEditor spmd_pipeline, const HloModuleConfig& config,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    const GpuTopology& gpu_topology) {
  AppendSPMDSimplifyPasses(
      spmd_pipeline.AppendSequence("spmd-simplify",
                                   kDefaultMaxFixedPointIterations),
      config, layout_insensitive_algsimp_opts);

  if (config.use_shardy_partitioner()) {
    spmd_pipeline.AppendPass("shardy-xla");
  } else {
    spmd_pipeline.AppendPass("hlo-constant-splitter");
    spmd_pipeline.AppendPass("sharding-propagation", {{"is_spmd", true}});
  }

  const DebugOptions& debug_options = config.debug_options();
  // Set max_windowed_einsum_iteration to slice_size, as there will be
  // significant overhead when scaled beyond the maximum size of the
  // fast-interconnect domain.
  spmd_pipeline.AppendPass(
      "spmd-partitioning",
      {
          {"num_partitions", config.num_partitions()},
          {"num_replicas", config.replica_count()},
          {"threshold_for_windowed_einsum_mib",
           debug_options.xla_gpu_threshold_for_windowed_einsum_mib()},
          {"unroll_windowed_einsum",
           debug_options.xla_gpu_multi_streamed_windowed_einsum()},
          {"skip_checking_windowed_einsum_users", true},
          {"disable_ag_rewrite_for_multiple_consumers", true},
          {"partial_windowed_einsum", true},
          {"total_bytes_windowed_einsum_threshold",
           debug_options.xla_gpu_operand_bytes_threshold_for_windowed_einsum()},
          {"max_windowed_einsum_iteration", gpu_topology.slice_size()},
          {"enable_dynamic_slice_collective_broadcast",
           debug_options.xla_spmd_enable_dynamic_slice_collective_broadcast()},
      });
  spmd_pipeline.AppendPass("nullary-function-wrap-inliner");
  if (debug_options.xla_enable_enzyme_comms_opt()) {
    spmd_pipeline.AppendPass("recognize-reduce-window");
    spmd_pipeline.AppendPass("collective-permute-cse");
  }
  // NOTE: even though the inliner is called in `pre-spmd-partitioner`,
  // it doesn't inline functions needed for ShardyXLA. ShardyXLA will also leave
  // functions called `kInlineableManualComputationFuncName` not inlined, so
  // we need to call the inliner again here.
  spmd_pipeline.AppendPass("shardy-manual-computation-inliner");
  spmd_pipeline.AppendPass("collective-permute-motion");
  // Custom partitioning callbacks can emit xla.scaled_dot composites during
  // partitioning, after the pre-SPMD CompositeRewriter has already run. From
  // here on they take the same path as the composites rewritten before SPMD.
  spmd_pipeline.AppendPass("composite-rewriter");
}

void AppendShardingRemovalPasses(HloPipelineEditor pipeline,
                                 const HloModuleConfig& config) {
  // Remove redundant sharding ops when partition_count == 1.
  pipeline.AppendPass("sharding-remover");
  // Run ShardyXLA without propagation, which enforces use-tuple-args.
  if (config.use_shardy_partitioner()) {
    pipeline.AppendPass("shardy-xla", {{"run_sharding_propagation", false}});
  }
  pipeline.AppendPass("nullary-function-wrap-inliner");
  pipeline.AppendPass("dce");
}

}  // namespace

AlgebraicSimplifierOptions GetAlgebraicSimplifierOptions(
    AlgebraicSimplifierMode mode, const DebugOptions& debug_options,
    bool is_cuda_platform) {
  AlgebraicSimplifierOptions opts;

  opts.set_enable_dot_strength_reduction(false);
  // On GPU it helps to reorder them so that the fused cuDNN kernel can be
  // used.
  opts.set_enable_conv_add_multiply_reorder(true);
  // GPU only supports canonical convolutions.
  opts.set_supports_non_canonical_dots(false);
  opts.set_enable_unconditional_reduce_of_concat_replacement(false);
  opts.set_rewrite_no_op_bitcast_convert_to_bitcast(true);
  opts.set_enable_conditional_simplification(true);
  opts.set_enable_fold_transpose_into_scatter(true);

  if (is_cuda_platform &&
      debug_options.xla_gpu_experimental_enable_conv_fusion()) {
    opts.set_enable_folding_pad_into_convolution(false);
    opts.set_enable_conv_operand_swap(false);
  }

  switch (mode) {
    case AlgebraicSimplifierMode::kPostFusionSimplification:
    case AlgebraicSimplifierMode::kLayoutInsensitive:
      // Always simplify reduce(transpose(x)) and reduce(reshape(x)), even when
      // the transpose/reshape has multiple users.  This helps int8 models,
      // which tend to have lots of transpose+reshape's (converting between NCHW
      // and NCHW_VECT_C).  Without this, those reshape+transposes can get
      // materialized out, which is really bad for perf.
      opts.set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);
      // GPU pipeline handles transposes better than slice+concatenate, so keep
      // the transpose.
      opts.set_rewrite_reshape_transpose_as_slice_concatenate(false);
      opts.set_enable_hoist_transpose_of_reshape(true);
      if (!is_cuda_platform) {
        opts.set_enable_conv_operand_swap(false);
      }
      break;
    default:
      opts.set_enable_conv_operand_swap(false);
  }

  switch (mode) {
    case AlgebraicSimplifierMode::kAfterSimplifyFPConversions:
    case AlgebraicSimplifierMode::kPostFusionSimplification:
    case AlgebraicSimplifierMode::kLayoutNormalization:
    case AlgebraicSimplifierMode::kPostLayoutAssignment:
      opts.set_is_layout_sensitive(true);
      break;
    default:
      break;
  }

  // "slow" minmax means we propagate nan.
  opts.set_minmax_propagate_nan(!debug_options.xla_gpu_enable_fast_min_max());

  if (mode == AlgebraicSimplifierMode::kAfterSimplifyFPConversions) {
    opts.set_enable_remove_no_op_reduce_precision(true);
  }

  return opts;
}

absl::StatusOr<HloPipelineProto> GetDefaultCompilationPipeline(
    const HloModuleConfig& config, const HloPassContext& context) {
  ABSL_ASSIGN_OR_RETURN(const se::DeviceDescription& device_description,
                   context.Get<const se::DeviceDescription>());
  ABSL_ASSIGN_OR_RETURN(const GpuTopology& gpu_topology,
                   context.Get<const GpuTopology>());
  const AlgebraicSimplifierOptions layout_insensitive_algsimp_opts =
      GetAlgebraicSimplifierOptions(
          AlgebraicSimplifierMode::kLayoutInsensitive, config.debug_options(),
          device_description.gpu_compute_capability().IsCuda());
  HloPipelineProto proto;
  HloPipelineEditor root(proto);
  root.AppendSequence("annotate-host-compute")
      .AppendPass("hlo_host_device_type_call_wrapper");
  AppendPreSPMDPartitionerPasses(root.AppendSequence("pre-spmd-partitioner"),
                                 layout_insensitive_algsimp_opts);
  if (config.num_partitions() > 1 && config.use_spmd_partitioning()) {
    AppendSPMDPartitionerPasses(root.AppendSequence("spmd-partitioner"), config,
                                layout_insensitive_algsimp_opts, gpu_topology);
  } else {
    AppendShardingRemovalPasses(root.AppendSequence("sharding-removal"),
                                config);
  }
  return proto;
}

}  // namespace xla::gpu

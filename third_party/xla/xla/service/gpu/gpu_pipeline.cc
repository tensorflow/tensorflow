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

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/pass/hlo_pipeline_editor.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

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
  return proto;
}

}  // namespace xla::gpu

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

#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/pass/hlo_pipeline_editor.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"

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

HloPipelineProto GetDefaultCompilationPipeline(
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts) {
  HloPipelineProto proto;
  HloPipelineEditor root(proto);
  root.AppendSequence("annotate-host-compute")
      .AppendPass("hlo_host_device_type_call_wrapper");
  AppendPreSPMDPartitionerPasses(root.AppendSequence("pre-spmd-partitioner"),
                                 layout_insensitive_algsimp_opts);
  return proto;
}

}  // namespace xla::gpu

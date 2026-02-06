/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/fusion_pipeline.h"

#include <memory>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/transforms/conv_fusion_rewriter.h"
#include "xla/service/gpu/transforms/multi_output_fusion.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/gpu/transforms/sort_iota_fusion.h"
#include "xla/service/gpu/transforms/variadic_op_splitter.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

HloPassPipeline FusionPipeline(
    const DebugOptions& debug_options,
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
    const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& gpu_device_info,
    mlir::MLIRContext* mlir_context) {
  HloPassFix<HloPassPipeline> fusion("fusion");
  // We try to split variadic ops with many parameters into several such ops
  // to avoid exceeding the parameter space.
  fusion.AddPass<VariadicOpSplitter>();
  HloVerifierOpts opts =
      HloVerifierOpts().MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout);
  fusion.AddInvariantCheckerDebug<HloVerifier>(
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts)),
      "hlo verifier (debug)");

  fusion.AddPass<SortIotaFusion>();

  // Rewrite convs into conv fusions.
  if (!debug_options.xla_gpu_experimental_disable_binary_libraries() &&
      debug_options.xla_gpu_experimental_enable_conv_fusion()) {
    fusion.AddPass<ConvFusionRewriter>();
  }

  GpuHloCostAnalysis::Options cost_analysis_options{
      shape_size_bytes_function,
      /*per_second_rates=*/{},
      /*min_latencies_seconds=*/{},
      /*count_multiple_input_accesses=*/true};
  fusion.AddPass<PriorityFusion>(thread_pool, gpu_device_info, alias_info,
                                 std::move(cost_analysis_options),
                                 mlir_context);

  // Running CSE affects how many users an op has. This plays a role in what
  // we detect as a tiled transpose fusion.
  fusion.AddPass<HloCSE>(
      /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
      /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
  fusion.AddPass<HloDCE>();
  fusion.AddPass<MultiOutputFusion>(gpu_device_info, alias_info,
                                    shape_size_bytes_function, mlir_context);
  fusion.AddPass<HloCSE>(
      /*is_layout_sensitive=*/true, /*ignore_control_dependencies=*/false,
      /*should_eliminate_computation=*/&HloComputation::IsFusionComputation);
  fusion.AddPass<HloDCE>();

  return std::move(fusion);
}

}  // namespace gpu
}  // namespace xla

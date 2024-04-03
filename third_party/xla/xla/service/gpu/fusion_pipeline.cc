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

#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/gpu/fusion_merger.h"
#include "xla/service/gpu/horizontal_input_fusion.h"
#include "xla/service/gpu/horizontal_loop_fusion.h"
#include "xla/service/gpu/instruction_fusion.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/multi_output_fusion.h"
#include "xla/service/gpu/priority_fusion.h"
#include "xla/service/gpu/variadic_op_splitter.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

HloPassPipeline FusionPipeline(
    const DebugOptions& debug_options,
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
    tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& gpu_device_info) {
  HloPassFix<HloPassPipeline> fusion("fusion");
  // We try to split variadic ops with many parameters into several such ops
  // to avoid exceeding the parameter space.
  fusion.AddPass<VariadicOpSplitter>();
  fusion.AddInvariantCheckerDebug<HloVerifier>(
      std::make_unique<CpuGpuVerifierMetadata>(
          HloVerifierOpts()
              .MakeLayoutSensitive()
              .WithInstructionCanChangeLayout(
                  LayoutAssignment::InstructionCanChangeLayout)),
      "hlo verifier (debug)");

  if (debug_options.xla_gpu_enable_priority_fusion()) {
    GpuHloCostAnalysis::Options cost_analysis_options{
        shape_size_bytes_function,
        /*per_second_rates=*/{},
        /*count_multiple_input_accesses=*/true};
    fusion.AddPass<GpuPriorityFusion>(thread_pool, gpu_device_info,
                                      std::move(cost_analysis_options));
  } else {
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/false,
                                         gpu_device_info);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true,
                                         gpu_device_info);
    fusion.AddPass<FusionMerger>(gpu_device_info, shape_size_bytes_function);
  }
  // Running CSE affects how many users an op has. This plays a role in what
  // we detect as a tiled transpose fusion.
  fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                         /*only_fusion_computations=*/true);
  fusion.AddPass<GpuMultiOutputFusion>(gpu_device_info,
                                       shape_size_bytes_function);
  fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                         /*only_fusion_computations=*/true);
  fusion.AddPass<HloDCE>();
  return std::move(fusion);
}

HloPassPipeline HorizontalFusionPipeline(
    const se::DeviceDescription& gpu_device_info) {
  HloPassFix<HloPassPipeline> horizontal_fusion("horizontal fusion");
  horizontal_fusion.AddPass<GpuHorizontalLoopFusion>();
  horizontal_fusion.AddPass<GpuHorizontalInputFusion>(gpu_device_info);
  horizontal_fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                                    /*only_fusion_computations=*/true);
  horizontal_fusion.AddPass<HloDCE>();

  return std::move(horizontal_fusion);
}

}  // namespace gpu
}  // namespace xla

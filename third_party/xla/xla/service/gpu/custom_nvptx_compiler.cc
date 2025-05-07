/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/custom_nvptx_compiler.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/hlo_fusion_stats.h"
#include "xla/service/gpu/transforms/add_tracking_suffix_to_instruction_names.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

absl::Status CustomNVPTXCompiler::RunFusionPasses(
    HloModule* hlo_module, const Compiler::TargetConfig& gpu_target_config,
    tsl::thread::ThreadPool* thread_pool,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  const se::DeviceDescription& gpu_device_info =
      gpu_target_config.device_description;

  HloPassPipeline pre_fusion("pre-fusion");
  pre_fusion.AddPass<AddTrackingSuffixToInstructionNames>();
  TF_RETURN_IF_ERROR(pre_fusion.Run(hlo_module).status());

  TF_RETURN_IF_ERROR(FusionPipeline(hlo_module->config().debug_options(),
                                    shape_size_fn, thread_pool, gpu_device_info)
                         .Run(hlo_module)
                         .status());

  TF_RETURN_IF_ERROR(
      HorizontalFusionPipeline(gpu_device_info).Run(hlo_module).status());

  // add the custom pass here

  if (VLOG_IS_ON(2)) {
    HloFusionStatsVisitor stats;
    TF_RETURN_IF_ERROR(hlo_module->entry_computation()->Accept(&stats));
    VLOG(2) << stats.ToString();
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

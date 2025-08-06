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

#include "xla/service/gpu/pre_scheduling_copy_insertion_pipeline.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/transforms/alias_passthrough_params.h"
#include "xla/service/gpu/transforms/copy_fusion.h"
#include "xla/service/gpu/transforms/horizontal_loop_fusion.h"
#include "xla/service/gpu/transforms/sanitize_constant_names.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/loop_schedule_linearizer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

HloPassPipeline PreSchedulingCopyInsertionPipeline(
    const HloModuleConfig& config, const GpuAliasInfo* alias_info,
    const se::DeviceDescription& device_description) {
  const DebugOptions& debug_options = config.debug_options();

  // In some cases, we have to place the result of an instruction in a temporary
  // buffer. For instance, the buffer that holds an external parameter is
  // assumed immutable at this point, and should not be reused for output
  // (b/27180329). Therefore, in that case, we set the output to be a copy of
  // the parameter.
  HloPassPipeline pipeline("pre-scheduling-copy-insertion");
  HloVerifierOpts opts =
      HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout);
  opts.verify_unique_channel_ids = !debug_options.xla_ignore_channel_id();
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
  pipeline.AddInvariantCheckerDebug<HloVerifier>(std::move(verifier_metadata),
                                                 "hlo verifier (debug)");

  // Copy insertion should be performed immediately before IR emission to avoid
  // inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes an
  // instruction which materializes a value). DCE must be run immediately before
  // (and sometime after) copy insertion, to avoid dead code from interfering
  // with the rewrites.
  pipeline.AddPass<HloDCE>();
  if (config.alias_passthrough_params()) {
    pipeline.AddPass<AliasPassthroughParams>();
  }
  pipeline.AddPass<LoopScheduleLinearizer>(alias_info);

  if (debug_options.xla_gpu_copy_insertion_use_region_analysis()) {
    constexpr int64_t kNoRegionBasedLiveRangeAnalysisLimit = -1;
    pipeline.AddPass<CopyInsertion>(alias_info,
                                    kNoRegionBasedLiveRangeAnalysisLimit);
  } else {
    pipeline.AddPass<CopyInsertion>(alias_info);
  }

  // We are using a sub-pipeline here, so that the verifier only runs after both
  // HorizontalLoopFusion and HloDCE.
  auto& sub_pipeline =
      pipeline.AddPass<HloPassPipeline>("horizontal-loop-fusion-for-copy");
  // To fuse the copy.
  sub_pipeline.AddPass<CopyFusion>(device_description);
  // Make sure to run HorizontalLoopFusion only inside the entry computation.
  // Fusing copies outside of the entry computation can break buffer assignment!
  sub_pipeline.AddPass<HorizontalLoopFusion>(device_description, "copy_",
                                             /*only_entry_computation=*/true);
  sub_pipeline.AddPass<HloDCE>();
  pipeline.AddPass<SanitizeConstantNames>();
  return pipeline;
}

}  // namespace gpu
}  // namespace xla

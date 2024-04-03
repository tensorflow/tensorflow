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

#include "xla/service/gpu/prepare_hlo_for_ir_emitting_pipeline.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/gpu/alias_passthrough_params.h"
#include "xla/service/gpu/copy_fusion.h"
#include "xla/service/gpu/gpu_sanitize_constant_names.h"
#include "xla/service/gpu/horizontal_loop_fusion.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/loop_schedule_linearizer.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

HloPassPipeline PrepareHloModuleForIrEmittingPipeline(
    HloModule& hlo_module,
    HloDataflowAnalysis::CanShareBuffer can_share_buffer) {
  const DebugOptions& debug_options = hlo_module.config().debug_options();

  // In some cases, we have to place the result of an instruction in a temporary
  // buffer. For instance, the buffer that holds an external parameter is
  // assumed immutable at this point, and should not be reused for output
  // (b/27180329). Therefore, in that case, we set the output to be a copy of
  // the parameter.
  HloPassPipeline pipeline("GPU-ir-emit-prepare");
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(
          HloVerifierOpts{}
              .MakeLayoutSensitive()
              .WithInstructionCanChangeLayout(
                  LayoutAssignment::InstructionCanChangeLayout));
  pipeline.AddInvariantCheckerDebug<HloVerifier>(std::move(verifier_metadata),
                                                 "hlo verifier (debug)");

  // Copy insertion should be performed immediately before IR emission to avoid
  // inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes an
  // instruction which materializes a value). DCE must be run immediately before
  // (and sometime after) copy insertion, to avoid dead code from interfering
  // with the rewrites.
  pipeline.AddPass<HloDCE>();
  if (hlo_module.config().alias_passthrough_params()) {
    pipeline.AddPass<AliasPassthroughParams>();
  }
  pipeline.AddPass<LoopScheduleLinearizer>(can_share_buffer);

  if (debug_options.xla_gpu_copy_insertion_use_region_analysis()) {
    constexpr int64_t kNoRegionBasedLiveRangeAnalysisLimit = -1;
    pipeline.AddPass<CopyInsertion>(can_share_buffer,
                                    kNoRegionBasedLiveRangeAnalysisLimit);
  } else {
    pipeline.AddPass<CopyInsertion>(can_share_buffer);
  }

  // We are using a sub-pipeline here, so that the verifier only runs after both
  // GpuHorizontalLoopFusion and HloDCE.
  auto& sub_pipeline =
      pipeline.AddPass<HloPassPipeline>("horizontal-loop-fusion-for-copy");
  // To fuse the copy.
  sub_pipeline.AddPass<CopyFusion>();
  sub_pipeline.AddPass<GpuHorizontalLoopFusion>("copy_");
  sub_pipeline.AddPass<HloDCE>();
  pipeline.AddPass<GpuSanitizeConstantNames>();
  return pipeline;
}

}  // namespace gpu
}  // namespace xla

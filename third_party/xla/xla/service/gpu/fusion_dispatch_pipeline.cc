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

#include "xla/service/gpu/fusion_dispatch_pipeline.h"

#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/transforms/fusion_block_level_rewriter.h"
#include "xla/backends/gpu/transforms/fusion_dynamic_memcpy_rewriter.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

HloPassPipeline FusionDispatchPipeline(
    const se::DeviceDescription& device_description,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    mlir::MLIRContext* mlir_context) {
  HloPassPipeline pipeline("fusion-dispatch-pipeline");
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<FusionBlockLevelRewriter>(device_description, shape_size_fn,
                                             mlir_context);
  pipeline.AddPass<FusionDynamicMemcpyRewriter>();
  return pipeline;
}

}  // namespace gpu
}  // namespace xla

/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_FUSION_DISPATCH_PIPELINE_H_
#define XLA_SERVICE_GPU_FUSION_DISPATCH_PIPELINE_H_

#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Returns a pipeline that attempts to redirect fusions to the most efficient
// emitter possible.
HloPassPipeline FusionDispatchPipeline(
    const se::DeviceDescription& device_description,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSION_DISPATCH_PIPELINE_H_

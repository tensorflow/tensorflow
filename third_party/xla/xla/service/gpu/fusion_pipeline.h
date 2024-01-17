/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_FUSION_PIPELINE_H_
#define XLA_SERVICE_GPU_FUSION_PIPELINE_H_

#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

// Function wrapper around the (non-horizontal) XLA GPU fusion pipeline.
// Thread pool may be nullptr.
HloPassPipeline FusionPipeline(
    const DebugOptions& debug_options,
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
    tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& gpu_device_info);

// Function wrapper around the horizontal XLA GPU fusion pipeline.
HloPassPipeline HorizontalFusionPipeline(
    const se::DeviceDescription& gpu_device_info);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSION_PIPELINE_H_

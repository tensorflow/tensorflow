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

#ifndef XLA_SERVICE_GPU_PREPARE_HLO_FOR_IR_EMITTING_PIPELINE_H_
#define XLA_SERVICE_GPU_PREPARE_HLO_FOR_IR_EMITTING_PIPELINE_H_

#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Function wrapper around the XLA GPU base pre-IR emission module preparation
// pipeline. This pipeline must be run right before IR emission to ensure
// correctness of the input module.
HloPassPipeline PrepareHloModuleForIrEmittingPipeline(
    HloModule& hlo_module, HloDataflowAnalysis::CanShareBuffer can_share_buffer,
    const se::DeviceDescription& device_description);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_PREPARE_HLO_FOR_IR_EMITTING_PIPELINE_H_

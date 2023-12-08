/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
#define XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

int64_t GetSizeOfShape(const Shape& shape, int pointer_size);

// Determines the schedule of HLO instructions for a module run on the GPU.
Status ScheduleGpuModule(HloModule* module, int64_t pointer_size,
                         int64_t memory_limit,
                         const se::DeviceDescription& gpu_device_info);
HloInstructionSequence PostProcessSchedule(const HloInstructionSequence& input);

int64_t GetSchedulerMemoryLimit(const HloModule* module,
                                const se::DeviceDescription& gpu_device_info,
                                int pointer_size);

constexpr absl::string_view kFingerprintBeforeLHS = "fingerprint_before_lhs";

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_

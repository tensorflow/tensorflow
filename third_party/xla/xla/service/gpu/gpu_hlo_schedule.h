/* Copyright 2017 The OpenXLA Authors.

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

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {
namespace gpu {

// Converts sync collective instructions to a pair of async start and done
// instructions.
absl::Status RunAsyncCollectivesConversionPasses(HloModule* module);

struct ScheduleMetadata {
  uint64_t scheduler_mem_limit;
};

// Defines the scheduler config to be used by LHS.
SchedulerConfig MakeGPUSchedulerConfig(uint64_t memory_limit,
                                       int64_t overlap_limit);

// Compute the device memory limit to be used by passes like scheduler and
// HLO rematerialization.
uint64_t GetSchedulerMemoryLimit(const HloModule& module,
                                 const se::DeviceDescription& gpu_device_info,
                                 int pointer_size);

// Determines the schedule of HLO instructions for a module run on the GPU.
absl::StatusOr<ScheduleMetadata> ScheduleGpuModule(
    HloModule* module, int64_t pointer_size,
    const se::DeviceDescription& gpu_device_info);

// Schedules a GPU module with `DefaultMemoryScheduler` and
// `PostProcessSchedule` postprocessing. If `peak_memory_bytes` is not nullptr,
// then the it will be set to peak memory usage in bytes.
absl::StatusOr<HloSchedule> ScheduleGpuModuleWithMemoryScheduler(
    const HloModule* module, int64_t pointer_size,
    int64_t* peak_memory_bytes = nullptr);

HloInstructionSequence PostProcessSchedule(const HloInstructionSequence& input);

constexpr absl::string_view kFingerprintBeforeLHS = "fingerprint_before_lhs";

namespace detail {

bool IsUnifiedAnalyticalModelEnabled(
    const HloModule& module, const se::DeviceDescription& gpu_device_info);

}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_

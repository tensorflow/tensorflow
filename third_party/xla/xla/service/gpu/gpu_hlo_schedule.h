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
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {
namespace gpu {

// Returns `absl::OkStatus` if every instruction in the profile is present in
// the module. `absl::InvalidArgumentError` with missing culprit costs/latencies
// otherwise.
absl::Status IsProfileApplicable(
    const HloModule* module,
    const tensorflow::profiler::ProfiledInstructionsProto& profile);

int64_t GetSizeOfShape(const Shape& shape, int pointer_size);

struct ScheduleMetadata {
  int64_t scheduler_mem_limit;
};

// Determines the schedule of HLO instructions for a module run on the GPU.
absl::StatusOr<ScheduleMetadata> ScheduleGpuModule(
    HloModule* module, int64_t pointer_size,
    const se::DeviceDescription& gpu_device_info);

HloInstructionSequence PostProcessSchedule(const HloInstructionSequence& input);

constexpr absl::string_view kFingerprintBeforeLHS = "fingerprint_before_lhs";

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_

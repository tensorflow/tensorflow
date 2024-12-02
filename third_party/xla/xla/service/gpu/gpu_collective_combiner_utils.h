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

#ifndef XLA_SERVICE_GPU_GPU_COLLECTIVE_COMBINER_UTILS_H_
#define XLA_SERVICE_GPU_GPU_COLLECTIVE_COMBINER_UTILS_H_

#include <cstdint>
#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Suggests a combiner threshold to the caller (combiner). At the moment it only
// suggests a lower value than a default combiner threshold if it exceeds
// available memory on a device. If the scheduling of a `module` failed for any
// reason the method return a default value of a combiner threshold for
// `collective_opcode`.
int64_t ComputeSuggestedCombinerThreshold(
    const HloModule& module, const se::DeviceDescription& device_info,
    std::function<absl::StatusOr<HloSchedule>(const HloModule*, int64_t,
                                              int64_t*)>
        scheduler,
    HloOpcode collective_opcode, int64_t pointer_size);

// Adds information that `instr` has been pipelined to the
// `CollectiveBackendInfo`. It is up to the caller to decide when to invoke
// this.
absl::Status AppendPipelinedInstruction(HloInstruction* instr);

// Returns true if module contains any pipelined instruction. False otherwise.
bool ContainsPipelinedInstruction(const HloModule& module);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_COLLECTIVE_COMBINER_UTILS_H_

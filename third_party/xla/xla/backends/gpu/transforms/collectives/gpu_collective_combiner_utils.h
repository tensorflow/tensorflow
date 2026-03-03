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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_GPU_COLLECTIVE_COMBINER_UTILS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_GPU_COLLECTIVE_COMBINER_UTILS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Adds information that `instr` has been pipelined to the
// `CollectiveBackendInfo`. It is up to the caller to decide when to invoke
// this.
absl::Status AppendPipelinedInstruction(HloInstruction* instr,
                                        HloInstruction* new_while_instr);

// Returns true if `instr` is a pipelined collective instruction.
bool IsPipelinedCollective(const HloInstruction& instr);

// Returns true if module contains any pipelined instruction. False otherwise.
bool ContainsPipelinedInstruction(const HloModule& module);

// Returns true if heuristic collective combining is enabled.
// Heuristic collective combining enables more aggressive optimizations based
// on the platform and HLO's topology.
bool EnableHeuristicCollectiveCombining(
    const HloModuleConfig& config,
    const se::DeviceDescription& device_description, int64_t nvlink_slice_size);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_GPU_COLLECTIVE_COMBINER_UTILS_H_

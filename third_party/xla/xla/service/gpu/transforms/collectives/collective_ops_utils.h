/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

enum class GPUCommunicationType {
  // The communication type could not be determined.
  UNDEFINED = 0,
  // Communication involves devices from multiple hosts, and every host
  // involved in the communication pattern has all of its devices participating.
  RAIL_ALIGNED = 1,
  // Communication involves devices from multiple hosts, but at least one of
  // the involved hosts has only a subset of its devices participating.
  NON_RAIL_ALIGNED = 2,
  // All devices participating in the collective operation reside on the same
  // host machine.
  SINGLE_HOST = 3
};

// Returns the type of communication pattern for a channel instruction.
absl::StatusOr<GPUCommunicationType> CommunicationType(
    int num_devices_per_host, const HloChannelInstruction& instr,
    const se::GpuComputeCapability& gpu_version);

// Returns true if instruction is a synchronous collective op.
bool IsGPUSyncCollective(const HloInstruction& instr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_

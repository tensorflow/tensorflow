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

#include <optional>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

enum class GPUCommunicationType {
  UNDEFINED = 0,
  RAIL_ALIGNED = 1,
  NON_RAIL_ALIGNED = 2,
  SINGLE_HOST = 3
};

// Returns the type of communication pattern for a channel instruction.
absl::StatusOr<GPUCommunicationType> CommunicationType(
    int num_devices_per_host, const HloChannelInstruction& instr,
    const se::GpuComputeCapability& gpu_version);

// Returns true if instruction is a synchronous collective op.
bool IsGPUSyncCollective(const HloInstruction& instr);

// Returns true if the topology is multi-host. Currently this function is
// heuristic based: it can be the case it will not detect a multi host case when
// a user decides to use < 8 GPUs per host. Moreover it tells nothing about how
// fast the interconnect between hosts is (Infiniband, NVLINK, DCN, etc.).
//
// Will return `std::nullopt` on any platform other than Hopper and Ampere.
std::optional<bool> IsMultiHostTopology(
    const HloModuleConfig& config,
    const se::DeviceDescription& device_description);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_

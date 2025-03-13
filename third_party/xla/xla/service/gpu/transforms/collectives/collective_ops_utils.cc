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

#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

bool IsGPUSyncCollective(const HloInstruction& instr) {
  auto backend_config = instr.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return false;
  }
  return backend_config->collective_backend_config().is_sync();
}

absl::StatusOr<GPUCommunicationType> CommunicationType(
    const HloCollectiveInstruction& instr,
    const se::GpuComputeCapability& gpu_version) {
  auto iota = instr.device_list().iota_replica_group_list();

  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(gpu_version);
  if (!(cuda_compute_capability.IsAtLeastAmpere() &&
        !cuda_compute_capability.IsAtLeastBlackwell())) {
    return absl::FailedPreconditionError(
        "Only Hopper is supported to get communication type");
  }

  // We assume no topology was provided to the compiler and no
  // `CUDA_VISIBLE_DEVICES` env var has been set.
  int num_devices_per_host = 8;

  if (!iota.has_value()) {
    return absl::FailedPreconditionError(
        "Only iota device assignment is supported.");
  }
  if (iota->num_replica_groups() == 1) {
    return GPUCommunicationType::RAIL_ALIGNED;
  }
  if (iota->num_replica_groups() == num_devices_per_host &&
      iota->transpose_perm().size() == 2 && iota->transpose_perm()[0] == 1) {
    return GPUCommunicationType::NON_RAIL_ALIGNED;
  }
  if (iota->num_devices_per_group() == num_devices_per_host) {
    return GPUCommunicationType::SINGLE_HOST;
  }
  return GPUCommunicationType::UNDEFINED;
}

}  // namespace gpu
}  // namespace xla

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

#include "xla/backends/gpu/transforms/collectives/gpu_collective_combiner_utils.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

absl::Status AppendPipelinedInstruction(HloInstruction* instr,
                                        HloInstruction* new_while_instr) {
  if (!IsCollective(instr)) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(auto config,
                      instr->backend_config<gpu::GpuBackendConfig>());
  config.mutable_collective_backend_config()->set_is_pipelined(true);
  return instr->set_backend_config(config);
}

bool IsPipelinedCollective(const HloInstruction& instr) {
  auto backend_config = instr.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    VLOG(2) << "Cannot read backend config for: " << instr.ToString();
    return false;
  }
  return backend_config->collective_backend_config().is_pipelined();
}

bool ContainsPipelinedInstruction(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (IsPipelinedCollective(*instr)) {
        return true;
      }
    }
  }
  return false;
}

bool EnableHeuristicCollectiveCombining(
    const HloModuleConfig& config,
    const se::DeviceDescription& device_description,
    int64_t nvlink_slice_size) {
  if (!config.debug_options()
           .xla_gpu_experimental_enable_heuristic_collective_combining()) {
    return false;
  }
  se::CudaComputeCapability cc = device_description.cuda_compute_capability();
  // Heuristic collective combining is not turned on before Ampere GPUs.
  if (!cc.IsAtLeastAmpere()) {
    return false;
  }
  if (IsIntraNVLinkDomain(config, nvlink_slice_size)) {
    VLOG(1) << "Disabled heuristic collective combining for intra-NVLink "
               "domain communication: HLO device count "
            << (config.num_partitions() * config.replica_count())
            << " <= NVLink slice size " << nvlink_slice_size;
    return false;
  }
  VLOG(1) << "Enabled heuristic collective combining for inter-NVLink domain "
             "communication: HLO device count "
          << (config.num_partitions() * config.replica_count())
          << " > NVLink slice size " << nvlink_slice_size;
  return true;
}

}  // namespace xla::gpu

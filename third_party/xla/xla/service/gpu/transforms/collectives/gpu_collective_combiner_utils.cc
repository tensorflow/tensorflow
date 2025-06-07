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

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla::gpu {

static int64_t GetDefaultValue(HloOpcode opcode) {
  if (opcode == HloOpcode::kAllGather) {
    return kDefaultAllGatherCombineThreshold;
  } else if (opcode == HloOpcode::kAllReduce) {
    return kDefaultAllReduceCombineThreshold;
  } else if (opcode == HloOpcode::kReduceScatter) {
    return kDefaultReduceScatterCombineThreshold;
  } else {
    LOG(FATAL) << "Expected collective op. Got: " << opcode;
  }
  return -1;
}

int64_t MaxAvailableMemory(const HloModule& module,
                           const se::DeviceDescription& device_info) {
  int64_t base_limit = module.config().device_memory_size() != 0
                           ? module.config().device_memory_size()
                           : device_info.device_memory_size();
  int32_t slop_factor =
      module.config().debug_options().xla_gpu_memory_limit_slop_factor();
  return base_limit * slop_factor / 100;
}

int64_t ComputeSuggestedCombinerThreshold(
    const HloModule& module, const se::DeviceDescription& device_info,
    HloOpcode collective_opcode, int64_t pointer_size) {
  int64_t peak_memory_bytes = -1;
  auto mem_schedule = ScheduleGpuModuleWithMemoryScheduler(
      &module, pointer_size, &peak_memory_bytes);

  if (!mem_schedule.ok() || peak_memory_bytes == -1) {
    VLOG(1) << "Cannot schedule module: " << mem_schedule.status().message();
    return GetDefaultValue(collective_opcode);
  }

  return MaxAvailableMemory(module, device_info) - peak_memory_bytes;
}

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

}  // namespace xla::gpu

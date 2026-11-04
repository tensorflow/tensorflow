/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_info_utils.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/all_gather.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/all_to_all.h"
#include "xla/backends/gpu/runtime/reduce_scatter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu_topology.h"

namespace xla::gpu {

absl::StatusOr<CollectiveKernelInfo> GetCollectiveInfo(
    const HloInstruction* instr, const GpuTopology& gpu_topology,
    const DeviceAssignment* device_assignment, bool is_multimem_enabled) {
  if (const auto* all_reduce = DynCast<HloAllReduceInstruction>(instr)) {
    return BuildAllReduceInfo(is_multimem_enabled, gpu_topology, all_reduce,
                              device_assignment);
  }

  if (const auto* all_gather = DynCast<HloAllGatherInstruction>(instr)) {
    return BuildAllGatherInfo(gpu_topology, all_gather, device_assignment);
  }

  if (const auto* all_to_all = DynCast<HloAllToAllInstruction>(instr)) {
    return BuildAllToAllInfo(gpu_topology, all_to_all, device_assignment);
  }

  if (const auto* reduce_scatter =
          DynCast<HloReduceScatterInstruction>(instr)) {
    return BuildReduceScatterInfo(gpu_topology, reduce_scatter,
                                  device_assignment);
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported collective opcode: ", instr->opcode()));
}

}  // namespace xla::gpu

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

#include "xla/backends/gpu/transforms/collectives/collective_backend_assigner.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

bool IsCollectivePermuteOp(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kCollectivePermute,
                          HloOpcode::kCollectivePermuteStart>(instr);
}

bool IsAllGatherOp(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kAllGather, HloOpcode::kAllGatherStart>(
      instr);
}

absl::StatusOr<bool> AssignCollectivesMode(
    HloModule* module, DebugOptions::CollectivesMode mode,
    bool (*predicate)(const HloInstruction*)) {
  if (mode == DebugOptions::COLLECTIVES_PRIVATE_MEMORY) {
    return false;
  }

  bool changed = false;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!predicate(instr)) {
        continue;
      }

      ASSIGN_OR_RETURN(auto config, instr->backend_config<GpuBackendConfig>());
      config.mutable_collective_backend_config()->set_collectives_mode(mode);
      RETURN_IF_ERROR(instr->set_backend_config(config));
      changed = true;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> CollectiveBackendAssigner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  ASSIGN_OR_RETURN(
      bool permute_mode_changed,
      AssignCollectivesMode(
          module,
          module->config().debug_options().xla_gpu_collective_permute_mode(),
          IsCollectivePermuteOp));
  changed |= permute_mode_changed;

  ASSIGN_OR_RETURN(
      bool all_gather_mode_changed,
      AssignCollectivesMode(
          module, module->config().debug_options().xla_gpu_all_gather_mode(),
          IsAllGatherOp));
  changed |= all_gather_mode_changed;

  return changed;
}

}  // namespace xla::gpu

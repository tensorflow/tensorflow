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

#include "xla/backends/gpu/transforms/erase_xla_late_frontend_attributes.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/spmd/shardy/constants.h"

namespace xla {
namespace gpu {
namespace {

bool IsShardyManualComputationCall(const HloModule* module,
                                   const HloInstruction* instruction) {
  if (!module->config().use_shardy_partitioner() ||
      instruction->opcode() != HloOpcode::kCall) {
    return false;
  }
  absl::string_view callee_name = instruction->to_apply()->name();
  return absl::StrContains(callee_name, "shmap_body") ||
         absl::StrContains(callee_name, sdy::kManualComputationFuncName);
}

}  // namespace

absl::StatusOr<bool> EraseXlaLateFrontendAttributes::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsShardyManualComputationCall(module, instruction)) {
        continue;
      }
      auto it = instruction->frontend_attributes().map().find("inlineable");
      if (it != instruction->frontend_attributes().map().end() &&
          it->second == "xla_late") {
        instruction->erase_frontend_attribute("inlineable");
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla

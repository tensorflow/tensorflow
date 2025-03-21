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

#include "xla/service/gpu/transforms/add_tracking_suffix_to_instruction_names.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> AddTrackingSuffixToInstructionNames::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Only rename instructions in non-fusion computations.
  for (xla::HloComputation* computation :
       module->MakeNonfusionComputationsSorted(execution_threads)) {
    for (auto* instruction : computation->instructions()) {
      // Skip non-fusible instruction to avoid breaking tests that are not
      // related to fusion.
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->opcode() == HloOpcode::kCustomCall ||
          instruction->opcode() == HloOpcode::kFusion ||
          !instruction->IsFusible())
        continue;

      auto new_name = absl::StrCat(instruction->name(), ".0");
      module->SetAndUniquifyInstrName(instruction, new_name);

      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla

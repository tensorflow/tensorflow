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

#include "xla/service/gpu/transforms/collectives/collective_annotator.h"

#include <stdbool.h>

#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"

namespace xla::gpu {

constexpr char kCollectiveIdAttr[] = "collective_id";

absl::StatusOr<bool> CollectiveAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (hlo_query::IsCollectiveCommunicationOp(instr->opcode())) {
        instr->set_frontend_attribute(kCollectiveIdAttr,
                                      absl::StrCat(instr->unique_id()));
        changed = true;
      }
    }
  }

  return changed;
}

std::optional<std::string> CollectiveId(const HloInstruction* instr) {
  return instr->get_frontend_attribute(kCollectiveIdAttr);
}

}  // namespace xla::gpu

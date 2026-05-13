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

#include "xla/hlo/transforms/metadata_interner.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

absl::StatusOr<bool> MetadataInterner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      auto it = instruction->frontend_attributes().map().find(
          "xla_interned_metadata");
      if (it != instruction->frontend_attributes().map().end()) {
        instruction->mutable_metadata()
            .mutable_interned_metadata_payload()
            ->set_value(it->second);
        FrontendAttributes attributes = instruction->frontend_attributes();
        attributes.mutable_map()->erase("xla_interned_metadata");
        instruction->set_frontend_attributes(attributes);
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla

/* Copyright 2017 The OpenXLA Authors.

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
#include "xla/frontend_attributes.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/xla_data.pb.h"

namespace xla {

void SetDisjointReadWriteRegionsAttr(HloInstruction* instruction) {
  instruction->set_frontend_attribute(xla::kXlaDisjointReadWriteRegions,
                                      "true");
}

bool HasDisjointReadWriteRegionsAttr(HloInstruction* instruction) {
  return instruction->frontend_attributes().map().contains(
      xla::kXlaDisjointReadWriteRegions);
}

absl::flat_hash_set<int> NonInvariantOperands(
    const HloInstruction& instruction) {
  absl::flat_hash_set<int> no_invariant_operands;
  if (instruction.has_frontend_attributes()) {
    auto it =
        instruction.frontend_attributes().map().find(kXlaNoInvariantOperands);
    if (it != instruction.frontend_attributes().map().end()) {
      for (absl::string_view s : absl::StrSplit(it->second, ',')) {
        if (int idx; absl::SimpleAtoi(s, &idx)) {
          no_invariant_operands.insert(idx);
        }
      }
    }
  }
  return no_invariant_operands;
}

}  // namespace xla

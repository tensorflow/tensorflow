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

#include "xla/service/scheduling_annotations_util.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/side_effect_util.h"

namespace xla {

std::optional<int64_t> GetSchedulingAnnotation(
    const HloInstruction* instruction) {
  const auto& attrs = instruction->frontend_attributes().map();
  if (!attrs.contains(kXlaSchedulingGroupIdAttr)) {
    return std::nullopt;
  }
  int64_t annotation_id;
  if (!absl::SimpleAtoi(attrs.at(kXlaSchedulingGroupIdAttr), &annotation_id)) {
    return std::nullopt;
  }
  return annotation_id;
}

void SetSchedulingAnnotation(HloInstruction* instruction, int64_t id) {
  FrontendAttributes fas = instruction->frontend_attributes();
  if (fas.map().contains(kXlaSchedulingGroupIdAttr)) {
    fas.mutable_map()->find(kXlaSchedulingGroupIdAttr)->second =
        absl::StrCat(id);
  } else {
    fas.mutable_map()->insert({kXlaSchedulingGroupIdAttr, absl::StrCat(id)});
  }
  instruction->set_frontend_attributes(fas);
}

int64_t NextSchedulingId(const HloModule& module) {
  int64_t next_scheduling_id = 1;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* hlo : comp->instructions()) {
      std::optional<int64_t> scheduling_id = GetSchedulingAnnotation(hlo);
      if (scheduling_id.has_value()) {
        next_scheduling_id =
            std::max(next_scheduling_id, scheduling_id.value() + 1);
      }
    }
  }
  return next_scheduling_id;
}

}  // namespace xla

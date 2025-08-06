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

#include "xla/hlo/ir/hlo_instruction_utils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace hlo_instruction_utils {
bool IsUnstridedSlice(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kSlice) {
    return false;
  }
  return absl::c_all_of(hlo->slice_strides(),
                        [](int64_t stride) { return stride == 1; });
}

using Interval = std::pair<int64_t, int64_t>;
void AddOrUpdateVectorOfPairsAsAttribute(HloInstruction* instr,
                                         std::string attr_name,
                                         std::vector<Interval> intervals) {
  std::string intervals_str =
      "{" +
      absl::StrJoin(intervals, ",",
                    [](std::string* out, Interval item) {
                      absl::StrAppend(out, "{", item.first, ",", item.second,
                                      "}");
                    }) +
      "}";
  FrontendAttributes attributes;
  attributes = instr->frontend_attributes();
  (*attributes.mutable_map())[attr_name] = intervals_str;
  instr->set_frontend_attributes(attributes);
}

}  // namespace hlo_instruction_utils
}  // namespace xla

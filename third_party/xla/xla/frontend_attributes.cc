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

#include <cstdlib>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool IsTrueVal(absl::string_view val) {
  return absl::EqualsIgnoreCase(val, "true") || val == "1" ||
         absl::EqualsIgnoreCase(val, "yes");
}

}  // namespace

void SetDisjointReadWriteRegionsAttr(HloInstruction* instruction) {
  if (instruction != nullptr) {
    instruction->set_frontend_attribute(xla::kXlaDisjointReadWriteRegions,
                                        "true");
  }
}

bool HasDisjointReadWriteRegionsAttr(const HloInstruction* instruction) {
  if (instruction == nullptr || !instruction->has_frontend_attributes()) {
    return false;
  }
  const auto& map = instruction->frontend_attributes().map();
  for (absl::string_view key :
       {kXlaDisjointReadWriteRegions, kXlaDisjointReadWriteRegionsNoUnderscore,
        "xla.disjoint_read_write_regions"}) {
    auto it = map.find(key);
    if (it != map.end() && IsTrueVal(it->second)) {
      return true;
    }
  }
  return false;
}

bool HasDisableWhileLoopCopiesAttr(const HloInstruction* instruction) {
  if (instruction == nullptr) {
    return false;
  }
  if (instruction->has_frontend_attributes()) {
    const auto& map = instruction->frontend_attributes().map();
    for (absl::string_view key :
         {kXlaDisableWhileLoopCopies, kXlaDisableWhileLoopCopiesNoUnderscore,
          "xla.disable_while_loop_copies"}) {
      auto it = map.find(key);
      if (it != map.end() && IsTrueVal(it->second)) {
        return true;
      }
    }
  }
  if (instruction->GetModule() != nullptr) {
    const auto& extra_options = instruction->GetModule()
                                    ->config()
                                    .debug_options()
                                    .xla_backend_extra_options();
    for (const char* key :
         {kXlaDisableWhileLoopCopies, kXlaDisableWhileLoopCopiesNoUnderscore,
          "xla.disable_while_loop_copies"}) {
      auto it = extra_options.find(key);
      if (it != extra_options.end() && IsTrueVal(it->second)) {
        return true;
      }
    }
  }
  if (const char* env = std::getenv("XLA_DISABLE_WHILE_LOOP_COPIES")) {
    if (IsTrueVal(env)) {
      return true;
    }
  }
  return false;
}

bool DoesPdlLaunch(const HloInstruction& instruction) {
  return instruction.has_frontend_attributes() &&
         instruction.frontend_attributes().map().contains(kXlaPdlLaunch);
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

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

#include "xla/hlo/ir/hlo_computation_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla::hlo_computation_utils {

HloComputation* CloneComputation(HloComputation* computation,
                                 absl::string_view suffix) {
  CHECK_NE(computation, nullptr);
  HloModule* module = computation->parent();
  CHECK_NE(module, nullptr);
  std::string suffix_str(suffix);
  if (!module->has_schedule() ||
      !module->schedule().is_computation_scheduled(computation)) {
    return module->AddEmbeddedComputation(suffix_str.empty()
                                              ? computation->Clone()
                                              : computation->Clone(suffix_str));
  }
  auto [clone, clone_sequence] =
      suffix_str.empty() ? computation->CloneWithSchedule()
                         : computation->CloneWithSchedule(suffix_str);
  HloComputation* clone_ptr = module->AddEmbeddedComputation(std::move(clone));
  module->schedule().set_sequence(clone_ptr, clone_sequence);
  return clone_ptr;
}

bool SpecializeComputations(
    HloModule* module,
    absl::FunctionRef<bool(const HloInstruction*)> is_special_call,
    absl::string_view suffix) {
  CHECK_NE(module, nullptr);
  if (!module->has_entry_computation()) {
    return false;
  }

  // Phase 0: Order computations in reverse post-order.
  std::vector<HloComputation*> computations = module->MakeComputationPostOrder(
      {module->entry_computation()->execution_thread()});
  absl::c_reverse(computations);

  // Phase 1: Find ordinary and special reachabilities from entry computation.
  absl::flat_hash_set<const HloComputation*> specials;
  absl::flat_hash_set<const HloComputation*> ordinaries;
  ordinaries.insert(module->entry_computation());
  for (const HloComputation* computation : computations) {
    const bool on_ordinaries = ordinaries.contains(computation);
    const bool on_specials = specials.contains(computation);
    if (!on_ordinaries && !on_specials) {
      continue;
    }
    for (const HloInstruction* instruction : computation->instructions()) {
      const bool on_special_call = is_special_call(instruction);
      for (const HloComputation* callee : instruction->called_computations()) {
        if (on_ordinaries && !on_special_call) {
          ordinaries.insert(callee);
        }
        if (on_specials || on_special_call) {
          specials.insert(callee);
        }
      }
    }
  }

  // Phase 2: Clone computations so `ordinaries` and `specials` become disjoint.
  std::vector<std::pair<HloComputation*, HloComputation*>> special_clones;
  for (HloComputation* computation : computations) {
    if (ordinaries.contains(computation) && specials.contains(computation)) {
      HloComputation* clone = CloneComputation(computation, suffix);
      special_clones.emplace_back(computation, clone);
      specials.erase(computation);
      specials.insert(clone);
    }
  }

  // Phase 3: Rewire callers on special paths to target the cloned computations.
  // Note that `special_clones` are in reverse post-order.
  for (const auto& entry : special_clones) {
    HloComputation* computation = entry.first;
    HloComputation* clone = entry.second;
    absl::InlinedVector<HloInstruction*, 1> callers =
        computation->caller_instructions();
    // Sort callers by unique_id for deterministic edge insertion.
    absl::c_sort(callers, [](const HloInstruction* a, const HloInstruction* b) {
      return a->unique_id() < b->unique_id();
    });
    for (HloInstruction* caller : callers) {
      if (specials.contains(caller->parent()) ||
          (ordinaries.contains(caller->parent()) && is_special_call(caller))) {
        caller->ReplaceCalledComputations(
            [computation, clone](HloComputation* callee) {
              return callee == computation ? clone : callee;
            });
      }
    }
  }

  return !special_clones.empty();
}

}  // namespace xla::hlo_computation_utils

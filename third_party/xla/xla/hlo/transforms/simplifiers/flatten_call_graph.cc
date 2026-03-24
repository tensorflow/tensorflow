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

#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/util.h"

namespace xla {
absl::StatusOr<bool> FlattenCallGraph::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3, "Before flatten call graph:\n" + module->ToString());

  bool changed = false;

  std::vector<HloComputation*> computations =
      module->MakeComputationPostOrder(execution_threads);

  for (auto* computation : computations) {
    if (skip_cloning_handler_(*computation)) {
      continue;
    }
    absl::InlinedVector<HloInstruction*, 1> callers;
    for (HloInstruction* caller : computation->caller_instructions()) {
      if (execution_threads.empty() ||
          execution_threads.contains(caller->parent()->execution_thread())) {
        callers.push_back(caller);
      }
    }

    if (callers.empty()) {
      continue;
    }

    // The order of insertion into `callers` depends on the iteration order in
    // `caller_instructions()`, which is a pointer-keyed map, so it's not
    // stable. Sort `callers` by unique id to make the iteration order below
    // deterministic.
    absl::c_sort(callers, [](const HloInstruction* a, const HloInstruction* b) {
      return a->unique_id() < b->unique_id();
    });
    for (int i = 0; i < callers.size(); ++i) {
      HloInstruction* caller = callers[i];

      // If this is the first (or only) caller, and it only refers to the
      // computation once (consider an `if` instruction that leads to the same
      // computation on multiple branches, or a pathological `while` where
      // the condition and body are the same computation), no need to clone.
      if (i == 0) {
        int computation_count = 0;
        for (const HloComputation* callee : caller->called_computations()) {
          if (callee == computation) {
            ++computation_count;
          }
        }
        if (computation_count <= 1) {
          continue;
        }
      }

      changed = true;
      std::vector<HloComputation*> worklist;
      caller->ReplaceCalledComputations([&](HloComputation* callee) {
        if (callee == computation && !skip_cloning_handler_(*callee)) {
          HloComputation* clone =
              module->AddEmbeddedComputation(callee->Clone());
          worklist.push_back(clone);
          return clone;
        }
        return callee;
      });

      // Clone the sub-tree of all computations called from this node.
      while (!worklist.empty()) {
        HloComputation* current = worklist.back();
        worklist.pop_back();
        for (HloInstruction* instruction : current->instructions()) {
          instruction->ReplaceCalledComputations([&](HloComputation* callee) {
            if (skip_cloning_handler_(*callee)) {
              return callee;
            }
            HloComputation* clone =
                module->AddEmbeddedComputation(callee->Clone());
            worklist.push_back(clone);
            return clone;
          });
        }
      }
    }
  }

  XLA_VLOG_LINES(3, "After flatten call graph:\n" + module->ToString());
  return changed;
}

}  // namespace xla

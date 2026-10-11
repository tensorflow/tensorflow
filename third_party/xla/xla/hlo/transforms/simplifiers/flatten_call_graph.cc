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

#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla {

bool FlattenCallGraph::SkipCloningForCalls(const HloComputation& computation) {
  auto callers = computation.caller_instructions();
  return absl::c_all_of(callers, [](const HloInstruction* caller) {
    return caller->opcode() == HloOpcode::kCall;
  });
}

namespace {

// Pure read-only predicate returning true if `computation` is itself an async
// computation or has at least one caller path from an enclosing async
// computation.
// TODO: b/534428440 - Make this a class method and cache results per
// computation.
bool IsInAsyncComputationSubtree(const HloComputation& computation) {
  std::vector<const HloComputation*> worklist = {&computation};
  absl::flat_hash_set<const HloComputation*> visited = {&computation};
  while (!worklist.empty()) {
    const HloComputation* current = worklist.back();
    worklist.pop_back();
    if (current->IsAsyncComputation()) {
      return true;
    }
    for (const HloInstruction* caller : current->caller_instructions()) {
      const HloComputation* parent = caller->parent();
      if (parent != nullptr && visited.insert(parent).second) {
        worklist.push_back(parent);
      }
    }
  }
  return false;
}

// Returns true if `instruction` is an async consumer (`kAsyncUpdate` or
// `kAsyncDone`) whose root `kAsyncStart` can be resolved within
// `execution_threads`. Such consumers share their `kAsyncStart`'s computation
// rather than cloning independently.
//
// How async chains are inferred across pipelined while loops:
// `async_chain_start()` (via `FindAsyncChainDataflow` in hlo_instructions.cc)
// walks backwards from `kAsyncDone` / `kAsyncUpdate` while tracking the active
// tuple ShapeIndex:
// - At `kGetTupleElement(index=i)`, it pushes `i` onto the active ShapeIndex
//   and continues into operand(0).
// - At `kParameter` of a `while_body`, it hops out to the enclosing `kWhile`'s
//   init operand (`while_op->operand(0)`, resolving prologue `kAsyncStart`s) or
//   `while_body->root_instruction()` at the same tuple ShapeIndex.
// - At `kWhile` (e.g., from an epilogue `kAsyncDone(GTE(while_op, index=i))`),
//   it hops into `while_body->root_instruction()` at the same tuple ShapeIndex
//   (resolving the in-body `kAsyncStart`).
// - At `kTuple`, it pops `i` from the ShapeIndex and continues into
//   `tuple->operand(i)` until reaching the originating `kAsyncStart`.
// Thus, the prologue `kAsyncStart` connects to the `kAsyncDone` inside the
// while loop body, and the epilogue `kAsyncDone` connects to the
// `kAsyncStart` inside the while loop body.
// TODO: b/534428440 - Document how pipelined while-loop async chains are
// inferred in MSA documentation (go/xla-msa).
bool HasIncludedAsyncChainStart(
    const HloInstruction* instruction,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (instruction->opcode() != HloOpcode::kAsyncUpdate &&
      instruction->opcode() != HloOpcode::kAsyncDone) {
    return false;
  }
  const HloInstruction* start = instruction->async_chain_start();
  return start != nullptr &&
         HloInstruction::IsThreadIncluded(start->parent()->execution_thread(),
                                          execution_threads);
}

// Updates all `kAsyncUpdate` and `kAsyncDone` instructions in
// `execution_threads` to reference the same `async_wrapped_computation` as
// their root `kAsyncStart`.
// TODO: b/534428440 - VLOG the async chains for debugging.
bool SetIdenticalCalledComputationForAsyncChain(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeComputationPostOrder(execution_threads)) {
    for (HloInstruction* inst : comp->instructions()) {
      if (!HasIncludedAsyncChainStart(inst, execution_threads)) {
        continue;
      }
      HloComputation* target_comp =
          inst->async_chain_start()->async_wrapped_computation();
      if (target_comp != nullptr &&
          inst->async_wrapped_computation() != target_comp) {
        inst->ReplaceCalledComputations(
            [&](HloComputation*) { return target_comp; });
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace

bool FlattenCallGraph::SkipCloningForNonAsync(
    const HloComputation& computation) {
  // Skip cloning only when `computation` is neither an async computation nor
  // transitively called from inside an async computation.
  return !IsInAsyncComputationSubtree(computation);
}

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
      if ((execution_threads.empty() ||
           execution_threads.contains(caller->parent()->execution_thread())) &&
          !HasIncludedAsyncChainStart(caller, execution_threads)) {
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

      auto clone_callee = [&](HloComputation* callee) {
        if (!module->has_schedule() ||
            !module->schedule().is_computation_scheduled(callee)) {
          return module->AddEmbeddedComputation(callee->Clone());
        }

        auto [clone, clone_sequence] = callee->CloneWithSchedule();
        HloComputation* clone_ptr =
            module->AddEmbeddedComputation(std::move(clone));
        module->schedule().set_sequence(clone_ptr, clone_sequence);
        return clone_ptr;
      };

      changed = true;
      std::vector<HloComputation*> worklist;
      caller->ReplaceCalledComputations([&](HloComputation* callee) {
        if (callee == computation && !skip_cloning_handler_(*callee)) {
          HloComputation* clone = clone_callee(callee);
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
          if (HasIncludedAsyncChainStart(instruction, execution_threads)) {
            continue;
          }
          instruction->ReplaceCalledComputations([&](HloComputation* callee) {
            if (skip_cloning_handler_(*callee)) {
              return callee;
            }
            HloComputation* clone = clone_callee(callee);
            worklist.push_back(clone);
            return clone;
          });
        }
      }
    }
  }

  if (SetIdenticalCalledComputationForAsyncChain(module, execution_threads)) {
    changed = true;
  }

  XLA_VLOG_LINES(3, "After flatten call graph:\n" + module->ToString());
  return changed;
}

}  // namespace xla

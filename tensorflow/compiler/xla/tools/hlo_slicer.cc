/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/hlo_slicer.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/call_graph.h"

namespace xla {
namespace {

// Intra-Computation forward/backward slicing: Conduct slicing inside the given
// computation, starting from the instructions passed in `sliced_instructions`.
//
// `sliced_instructions` passes in the starting points of the intra-computation
// slicing, and all the propagated instructions are also recorded in this
// structure. In case of forward slicing, the passed-in starting points are some
// caller instructions in this computation. In case of backward slicing, the
// passed-in starting points are the root instruction of this computation.
//
// If a frontier instruction is encountered (determined by `frontier_selector`),
// it will be added to `frontier_instructions`.
void IntraComputationSlicing(
    const HloComputation* computation,
    absl::flat_hash_set<const HloInstruction*>& sliced_instructions,
    absl::flat_hash_set<const HloInstruction*>& frontier_instructions,
    bool forward_slice, FrontierSelector frontier_selector,
    bool ignore_control_dependency) {
  std::deque<const HloInstruction*> worklist(sliced_instructions.begin(),
                                             sliced_instructions.end());

  while (!worklist.empty()) {
    const HloInstruction* inst = worklist.back();
    worklist.pop_back();

    // If `inst` is at the frontier, bookkeep it, and continue.
    if (!frontier_selector(inst)) {
      frontier_instructions.insert(inst);
      continue;
    }

    // Initialize data-dependent instructions
    std::vector<HloInstruction*> instructions_to_propagate =
        forward_slice ? std::vector<HloInstruction*>(inst->users().begin(),
                                                     inst->users().end())
                      : std::vector<HloInstruction*>(inst->operands().begin(),
                                                     inst->operands().end());

    // Append control-dependent instructions if necessary
    if (!ignore_control_dependency) {
      if (forward_slice) {
        instructions_to_propagate.insert(instructions_to_propagate.end(),
                                         inst->control_successors().begin(),
                                         inst->control_successors().end());
      } else {
        instructions_to_propagate.insert(instructions_to_propagate.end(),
                                         inst->control_predecessors().begin(),
                                         inst->control_predecessors().end());
      }
    }

    for (auto next_inst : instructions_to_propagate) {
      if (!sliced_instructions.contains(next_inst)) {
        worklist.push_front(next_inst);
        sliced_instructions.insert(next_inst);
      }
    }
  }
}

}  // namespace

SliceOutput SliceModule(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    FrontierSelector frontier_selector, bool ignore_control_dependency,
    bool forward_slice) {
  // Initialize `sliced_computation_instructions_map`, which keeps track of all
  // the sliced instructions.
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      sliced_computation_instructions_map;
  for (auto inst : slice_starting_instructions) {
    sliced_computation_instructions_map[inst->parent()].insert(inst);
  }

  // Initialize `frontier_computation_instructions_map`.
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      frontier_computation_instructions_map;

  // Build call graph.
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module);

  // Traverse computations in the post-order(forward slicing) or
  // reverse post-order(backward slicing) manner, and conduct intra-computation
  // slicing in that order.
  //
  // Post-order guarantees that when computation `a` is visited, all of its
  // callee computations have been visited, thus all the necessary propagation
  // to `a` has been conducted (i.e., the sliced caller instruction in `a` has
  // been marked, which serve as the starting point in
  // `IntraComputationSlicing`).
  //
  // Similarly, reverse post-order guarantees that when computation `a` is
  // visited, all of its caller computations have been visited, thus its root
  // instruction has been marked, which serve as the starting point in
  // `IntraComputationSlicing`.
  std::vector<HloComputation*> post_order_computations =
      hlo_module->MakeComputationPostOrder();
  std::vector<HloComputation*> computations_to_traverse =
      forward_slice
          ? post_order_computations
          : std::vector<HloComputation*>(post_order_computations.rbegin(),
                                         post_order_computations.rend());

  for (auto computation : computations_to_traverse) {
    if (sliced_computation_instructions_map.contains(computation)) {
      // Do intra-computation slicing, starting from the instructions that has
      // been inserted in `sliced_computation_instructions_map[computation]`.
      IntraComputationSlicing(
          computation, sliced_computation_instructions_map[computation],
          frontier_computation_instructions_map[computation], forward_slice,
          frontier_selector, ignore_control_dependency);

      if (forward_slice) {
        // Skip propagating if the ROOT instruction of the current computation
        // is NOT sliced. It is either because (1) the sliced instructions are
        // actually dead code or (2) `frontier_selector` finds frontier and stop
        // propagation. The found frontier could be at the root instruction, and
        // in this case, we stop propagation.
        if (!sliced_computation_instructions_map[computation].contains(
                computation->root_instruction()) ||
            frontier_computation_instructions_map[computation].contains(
                computation->root_instruction())) {
          continue;
        }

        // Continue propagating to successors of the current computation, by
        // inserting its caller computation into
        // `sliced_computation_instructions_map`, and inserting the caller
        // instructions as the starting points for intra-computation slicing.
        for (auto caller_inst :
             call_graph->GetComputationCallers(computation)) {
          sliced_computation_instructions_map[caller_inst->parent()].insert(
              caller_inst);
        }
      }
      if (!forward_slice) {
        // Propagate to the callee computation of the current computation
        // that the sliced instructions invoke, by inserting its callee
        // computation into `sliced_computation_instructions_map`, and inserting
        // the root instruction of the callee computation as the starting points
        // for later intra-computation slicing.
        for (const auto& callsite :
             call_graph->GetNode(computation).callsites()) {
          if (sliced_computation_instructions_map[computation].contains(
                  callsite.instruction())) {
            for (auto callee : callsite.called_computations()) {
              sliced_computation_instructions_map[callee].insert(
                  callee->root_instruction());
            }
          }
        }
      }
    }
  }

  return SliceOutput{sliced_computation_instructions_map,
                     frontier_computation_instructions_map};
}

}  // namespace xla

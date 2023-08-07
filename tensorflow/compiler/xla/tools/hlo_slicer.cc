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
#include "absl/types/span.h"
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
    if (frontier_selector && !frontier_selector(inst)) {
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

SliceOutput SliceModuleHelper(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    FrontierSelector frontier_selector, bool ignore_control_dependency,
    bool forward_slice, bool nearest_common_ancestor_as_root) {
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

  // If `nearest_common_ancestor_as_root` is enabled, we compute the
  // HloComputations that hold the `nearest_common_ancestor` instruction, which
  // are the stopping points when iterating through `computations_to_traverse`.
  absl::flat_hash_set<const HloComputation*>
      nearest_common_ancestor_computations;
  if (nearest_common_ancestor_as_root) {
    std::vector<const HloComputation*> starting_computations;
    for (const auto& [computation, instructions] :
         sliced_computation_instructions_map) {
      starting_computations.push_back(computation);
    }
    nearest_common_ancestor_computations =
        call_graph->NearestCommonAncestorComputations(starting_computations);
    CHECK(!nearest_common_ancestor_computations.empty());
  }

  for (auto computation : computations_to_traverse) {
    if (sliced_computation_instructions_map.contains(computation)) {
      auto slicing_starting_instructions = std::vector<const HloInstruction*>(
          sliced_computation_instructions_map[computation].begin(),
          sliced_computation_instructions_map[computation].end());

      // Do intra-computation slicing, starting from the instructions that has
      // been inserted in `sliced_computation_instructions_map[computation]`.
      IntraComputationSlicing(
          computation, sliced_computation_instructions_map[computation],
          frontier_computation_instructions_map[computation], forward_slice,
          frontier_selector, ignore_control_dependency);

      // The block below propagate the slicing results from the current visiting
      // computation to the next ones.
      if (forward_slice) {
        // Check if the current computation is one of the
        // `nearest_common_ancestor_computations`. If yes, we find the
        // `nearest_common_ancestor` as an instruction, and stop here.
        if (nearest_common_ancestor_as_root &&
            nearest_common_ancestor_computations.contains(computation)) {
          // We use one of the nearest common ancestor instructions.
          const HloInstruction* nearest_common_ancestor_instruction =
              *(call_graph->NearestCommonAncestorInstructions(
                    slicing_starting_instructions))
                   .begin();
          CHECK_NE(nearest_common_ancestor_instruction, nullptr);
          return SliceOutput{sliced_computation_instructions_map,
                             frontier_computation_instructions_map,
                             nearest_common_ancestor_instruction};
        }

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

}  // namespace

SliceOutput SliceModule(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    FrontierSelector frontier_selector, bool ignore_control_dependency,
    bool forward_slice, bool nearest_common_ancestor_as_root) {
  if (forward_slice) {
    if (!nearest_common_ancestor_as_root) {
      // Forward slicing with the original root as the root.
      return SliceModuleHelper(hlo_module, slice_starting_instructions,
                               frontier_selector, ignore_control_dependency,
                               /*forward_slice=*/true,
                               /*nearest_common_ancestor_as_root=*/false);
    } else {
      // Forward slicing with the nearest common ancestor (NCA) as the root.
      //
      // Internally, this feature is implemented by the following two steps:
      //  1. Conducting a pass of forward slicing and looking for the NCA
      //     instruction.  We first compute the "NCA computation", which is the
      //     NCA, of the computations that hold the
      //     `slice_starting_instructions`. This computation is achieved by
      //     invoking "NearestCommonAncestorComputations" in the call graph.
      //     Then, when we reach the "NCA computation", we compute the NCA of
      //     the instructions that calls the computations which are on the path
      //     from the `slice_starting_instructions` to this NCA computation.
      //  2. The slice from step 1 contains some redundant instructions,
      //     because, when we do forward slicing, we do not know the exact path
      //     to the NCA, and there could some nodes that cannot be reached from
      //     the NCA. Therefore, in this step, we conduct a pass of backward
      //     slicing from the NCA and filter out the redundant instructions, by
      //     taking the intersection between the backward slicing results and
      //     the forward slicing results from step 1.

      // Sanity check.
      CHECK(forward_slice) << "Option `nearest_common_ancestor_as_root` can "
                              "only be enabled when "
                              "forward slicing";
      CHECK((frontier_selector == nullptr))
          << "Option `nearest_common_ancestor_as_root` can not be specified "
             "with `frontier_selector`";

      // Forward slicing to identify nearest common ancestor
      SliceOutput forward_slice_output =
          SliceModuleHelper(hlo_module, slice_starting_instructions,
                            /*frontier_selector=*/nullptr,
                            ignore_control_dependency, /*forward_slice=*/true,
                            /*nearest_common_ancestor_as_root=*/true);
      std::vector<const HloInstruction*> nearest_common_ancestor(
          {forward_slice_output.nearest_common_ancestor_root()});
      CHECK_EQ(nearest_common_ancestor.size(), 1);

      // Backward slicing from the nearest common ancestor to filter out
      // the redundant computations/instructions in the sliced result in step 1.
      SliceOutput backward_slice_output =
          SliceModuleHelper(hlo_module, /*slice_starting_instructions=*/
                            absl::MakeSpan(nearest_common_ancestor),
                            /*frontier_selector=*/nullptr,
                            ignore_control_dependency, /*forward_slice=*/false,
                            /*nearest_common_ancestor_as_root=*/false);

      // Intersect the sliced instructions between forward slicing pass and
      // backward slicing pass as the the new sliced instructions, and return
      // the new SliceOutput.
      return SliceOutput{SliceOutput::IntersectSlicedInstructions(
                             forward_slice_output, backward_slice_output),
                         backward_slice_output.frontier_instructions(),
                         forward_slice_output.nearest_common_ancestor_root()};
    }
  } else {
    // Backward slicing.
    return SliceModuleHelper(hlo_module, slice_starting_instructions,
                             frontier_selector, ignore_control_dependency,
                             /*forward_slice=*/false,
                             /*nearest_common_ancestor_as_root=*/false);
  }
}

}  // namespace xla

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

#include <deque>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/call_graph.h"

namespace xla {
namespace {

// Intra-Computation forward/backward slicing: Conduct slicing inside the given
// computation. It begins with the relevant instructions in
// `sliced_instructions_map`, and it adds all the instructions propagated
// in-place.
//
// If a frontier instruction is encountered, it will be added to
// `frontier_instructions`.
void IntraComputationSlicing(
    absl::flat_hash_set<const HloInstruction*>& sliced_instructions,
    absl::flat_hash_set<const HloInstruction*>& frontier_instructions,
    bool forward_slice, HloSelector hlo_selector,
    bool ignore_control_dependency) {
  std::deque<const HloInstruction*> worklist(sliced_instructions.begin(),
                                             sliced_instructions.end());

  while (!worklist.empty()) {
    const HloInstruction* inst = worklist.back();

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

    for (auto inst : instructions_to_propagate) {
      if (!hlo_selector(inst)) {
        frontier_instructions.insert(inst);
        sliced_instructions.insert(inst);
        continue;
      }

      if (!sliced_instructions.contains(inst)) {
        worklist.push_front(inst);
        sliced_instructions.insert(inst);
      }
    }
    worklist.pop_back();
  }
}

}  // namespace

SliceOutput SliceModule(
    const HloModule* hlo_module,
    std::vector<const HloInstruction*>& relevant_instructions,
    HloSelector hlo_selector, bool ignore_control_dependency) {
  // TODO(b/288160117): backward slicing not implemented yet
  bool forward_slice = true;

  // Initialize `sliced_comp_instructions_map`, which keeps track of all the
  // sliced instructions
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      sliced_comp_instructions_map;
  for (auto inst : relevant_instructions) {
    sliced_comp_instructions_map[inst->parent()].insert(inst);
  }

  // Initialize `frontier_comp_instructions_map`
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      frontier_comp_instructions_map;

  // Build call graph
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module);

  // Traverse computations in the post-order(forward slicing) or
  // pre-order(backward slicing) manner, and conduct intra-computation
  // slicing in that order.
  std::vector<HloComputation*> computations_to_traverse =
      forward_slice ? hlo_module->MakeComputationPostOrder()
                    // TODO(b/288160117): backward slicing not implemented yet
                    : std::vector<HloComputation*>();
  for (auto computation : computations_to_traverse) {
    if (sliced_comp_instructions_map.contains(computation)) {
      // Do intra-computation slicing
      IntraComputationSlicing(sliced_comp_instructions_map[computation],
                              frontier_comp_instructions_map[computation],
                              forward_slice, hlo_selector,
                              ignore_control_dependency);

      // Forward slicing: Continue propagating to successors of the current
      // computation if the ROOT instruction of the current computation is
      // sliced
      if (forward_slice && sliced_comp_instructions_map[computation].contains(
                               computation->root_instruction())) {
        for (auto caller_inst :
             call_graph->GetComputationCallers(computation)) {
          sliced_comp_instructions_map[caller_inst->parent()].insert(
              caller_inst);
        }
      }
      // TODO(b/288160117): Backward slice not implemented yet
      // Backward slicing: propagate to the predecessors of the current
      // computation that the sliced instructions invoke
      if (!forward_slice) {
        QCHECK(false);
      }
    }
  }

  return SliceOutput{sliced_comp_instructions_map,
                     frontier_comp_instructions_map};
}

}  // namespace xla

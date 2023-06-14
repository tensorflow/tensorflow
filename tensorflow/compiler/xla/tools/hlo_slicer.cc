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

// Intra-Computation forward slicing: Conduct forward slicing inside the given
// computation. It begins with the relevant instructions in
// `sliced_comp_insts_map[computation]`, and it adds all the instructions
// propagated in-place.
//
// We assume that the root inst will be propagated (otherwise the given relevant
// insts are dead code).
void IntraCompForwardSlicing(
    HloComputation* computation,
    absl::flat_hash_map<const HloComputation*,
                        absl::flat_hash_set<const HloInstruction*>>&
        sliced_comp_insts_map,
    bool ignore_control_predecessors) {
  std::deque<const HloInstruction*> worklist(
      sliced_comp_insts_map[computation].begin(),
      sliced_comp_insts_map[computation].end());

  while (!worklist.empty()) {
    const HloInstruction* inst = worklist.back();

    std::vector<HloInstruction*> successors(inst->users().begin(),
                                            inst->users().end());
    if (ignore_control_predecessors) {
      successors.insert(successors.end(), inst->control_successors().begin(),
                        inst->control_successors().end());
    }

    for (auto user_inst : successors) {
      if (!sliced_comp_insts_map[computation].contains(user_inst)) {
        worklist.push_front(user_inst);
        sliced_comp_insts_map[computation].insert(user_inst);
      }
    }
    worklist.pop_back();
  }

  // The root instruction should be included
  QCHECK(sliced_comp_insts_map[computation].contains(
      computation->root_instruction()))
      << "The root instruction should be included in the sliced computation";
}

}  // namespace

absl::flat_hash_map<const HloComputation*,
                    absl::flat_hash_set<const HloInstruction*>>
SliceModule(const HloModule* hlo_module,
            std::vector<const HloInstruction*>& relevant_instructions,
            bool ignore_control_predecessors) {
  // Initialize `sliced_comp_insts_map`
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      sliced_comp_insts_map;
  for (auto inst : relevant_instructions) {
    sliced_comp_insts_map[inst->parent()].insert(inst);
  }

  // Build call graph
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module);

  // Traverse computations in the post-order manner, and conduct
  // inter-computation forward slicing in that order.
  // `sliced_comp_insts_map` keeps track of all the relevant insts.
  for (auto computation : hlo_module->MakeComputationPostOrder()) {
    if (sliced_comp_insts_map.contains(computation)) {
      // Do intra-computation forward slicing
      IntraCompForwardSlicing(computation, sliced_comp_insts_map,
                              ignore_control_predecessors);

      // Track successor of the current computation computations containing the
      // caller instruction of the current computation.
      // TODO: Note that, theorectically, we can add only one of
      // these caller instructions (computations)
      for (auto caller_inst : call_graph->GetComputationCallers(computation)) {
        sliced_comp_insts_map[caller_inst->parent()].insert(caller_inst);
      }
    }
  }

  return sliced_comp_insts_map;
}

}  // namespace xla

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_UTIL_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_group_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {

// Collection of utilities for handling HloModuleGroups.
class HloModuleGroupUtil {
 public:
  explicit HloModuleGroupUtil(const HloModuleGroupMetadata& metadata)
      : metadata_(metadata) {}

  // Returns all unique predecessors of the instruction. This includes:
  // * predecessors in the same computation: operands and control predecessors
  // * Recv is a predecessor of Send
  // * Send is a predecessor of RecvDone
  // * predecessors of companions (if the instruction is a companion while)
  // * predecessors' companions (for any predecessor that is a companion while)
  std::vector<HloInstruction*> GlobalPredecessors(HloInstruction* instruction);

  // Returns all unique successors of the instruction. This includes:
  // * successors in the same computation: users and control successors
  // * Send is a successor of Recv
  // * RecvDone is a successor of Send
  // * successors of companions (if the instruction is a companion while)
  // * successors' companions (for any successor that is a companion while)
  std::vector<HloInstruction*> GlobalSuccessors(HloInstruction* instruction);

  // Returns the root instructions of the computations.
  std::vector<HloInstruction*> RootInstructions(
      absl::Span<HloComputation* const> computations);

  // Visit state of each instruction during DFS traversal.
  enum VisitState {
    kNotVisited = 0,
    kVisiting,
    kVisited,
  };

  // Function called on each instruction group during the DFS traversal. See the
  // comment for VisitTopologicalOrder()).
  using VisitFunction = absl::FunctionRef<Status(
      HloInstruction* hlo,
      const std::vector<HloInstruction*>& instruction_group)>;

  // Given the hlo instruction as the root, recursively visits all its
  // predecessor instructions in DFS order to visit nodes in topological order.
  //
  // Note that the DFS traversal does not only visit nodes in the same
  // computation (parent of the root instruction), but also visits nodes in
  // different computations connected via communication instructions. During the
  // traversal, companion While instructions (see the class comment in
  // HloModuleGroupMetadata) are treated as a single instruction (called
  // instruction group, which contains only a single instruction if the visiting
  // node is not a companion while) -- visiting one of the instructions in the
  // group effectively visits all other instructions in the group, and then all
  // predecessor instructions of the group are visited.
  //
  // * visit_state: map from each instruction to its visit state.
  // * visit_function: function called when each instruction group.
  // * root: the root instruction of the traversal.
  using VisitStates = absl::flat_hash_map<HloInstruction*, VisitState>;
  Status VisitTopologicalOrder(VisitStates* visit_state,
                               VisitFunction visit_function,
                               HloInstruction* root);

  // Verifies that the computations are well-formed (e.g., no cycles).
  Status VerifyComputations(absl::Span<HloComputation* const> computations);

  // Below Reachability utils resemble those in HloComputation, except that
  // they can handle instructions across multiple computations.
  //
  // Creates the reachability map for the instructions in the computations.
  StatusOr<std::unique_ptr<HloReachabilityMap>> ComputeReachability(
      absl::Span<HloComputation* const> computations);

  // Updates the reachability of the given instruction, taking the global
  // predecessors and successors into account.
  void UpdateReachabilityThroughInstruction(
      HloInstruction* instruction, HloReachabilityMap* reachability_map);

 private:
  std::string CycleToString(HloInstruction* instruction);

  const HloModuleGroupMetadata& metadata_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_UTIL_H_

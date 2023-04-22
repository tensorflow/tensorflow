/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_dce.h"

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<bool> HloDCE::RunOnComputation(
    HloComputation* computation, bool remove_cross_partition_collective_ops) {
  bool changed = false;
  VLOG(3) << "Before dce:";
  XLA_VLOG_LINES(3, computation->ToString());
  // Remove any dead roots and their dead transitive operands. Collect them
  // into a separate list first to avoid problems with iterating through the
  // computation's instruction while simultaneously removing instructions.
  std::vector<HloInstruction*> dead_roots;
  for (auto* instruction : computation->instructions()) {
    auto maybe_collective_op = DynCast<HloCollectiveInstruction>(instruction);
    if (instruction != computation->root_instruction() &&
        instruction->user_count() == 0 &&
        computation->IsSafelyRemovable(instruction) &&
        (!instruction->HasSideEffect() ||
         (remove_cross_partition_collective_ops && maybe_collective_op &&
          !maybe_collective_op->constrain_layout()))) {
      dead_roots.push_back(instruction);
    }
  }

  for (HloInstruction* dead_root : dead_roots) {
    VLOG(1) << "Removing dead root " << dead_root->ToString()
            << " and it's unused operands";
    TF_RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(dead_root));
    changed = true;
  }
  if (changed) {
    VLOG(3) << "After dce:";
    XLA_VLOG_LINES(3, computation->ToString());
  }
  return changed;
}

StatusOr<bool> HloDCE::Run(HloModule* module) {
  bool changed = false;

  VLOG(2) << "Before dce:";
  XLA_VLOG_LINES(2, module->ToString());

  // Run DCE on each computation.
  for (auto* computation : module->MakeComputationPostOrder()) {
    TF_ASSIGN_OR_RETURN(
        bool changed_for_computation,
        RunOnComputation(computation, remove_cross_partition_collective_ops_));
    changed |= changed_for_computation;
  }

  // Now DCE HloComputations.  First, collect the computations that are
  // referenced by some remaining instruction.
  absl::flat_hash_set<HloComputation*> live_computations;
  if (HloComputation* entry_computation = module->entry_computation()) {
    live_computations.insert(entry_computation);
  }
  for (auto* computation : module->MakeComputationPostOrder()) {
    for (auto* instruction : computation->instructions()) {
      for (auto* subcomp : instruction->called_computations()) {
        live_computations.insert(subcomp);
      }
    }
  }

  // Remove dead computations.
  for (auto* computation : module->MakeComputationPostOrder()) {
    if (!live_computations.contains(computation)) {
      TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(computation));
      changed = true;
    }
  }

  VLOG(2) << "After dce:";
  XLA_VLOG_LINES(2, module->ToString());

  return changed;
}

}  // namespace xla

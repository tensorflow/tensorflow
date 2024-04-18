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

#include "xla/service/hlo_dce.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

// Checks if the instruction is a removable while given
// remove_cross_partition_collective_ops
bool IsRemovableWhile(HloInstruction* instruction,
                      bool remove_cross_partition_collective_ops) {
  if (instruction->opcode() != HloOpcode::kWhile) {
    return false;
  }
  for (HloComputation* computation : instruction->called_computations()) {
    for (HloInstruction* called_instr : computation->instructions()) {
      auto maybe_collective_op =
          DynCast<HloCollectiveInstruction>(called_instr);
      if (called_instr->HasSideEffect() &&
          (!remove_cross_partition_collective_ops || !maybe_collective_op ||
           maybe_collective_op->constrain_layout())) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

/*static*/ absl::StatusOr<bool> HloDCE::RunOnComputation(
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
    if (instruction->IsDead() && computation->IsSafelyRemovable(instruction) &&
        (!instruction->IsCustomCall("Sharding") ||
         (!instruction->operand(0)->IsRoot() &&
          instruction->operand(0)->opcode() != HloOpcode::kParameter &&
          instruction->operand(0)->user_count() == 1)) &&
        (!instruction->HasSideEffect() ||
         (remove_cross_partition_collective_ops && maybe_collective_op &&
          !maybe_collective_op->constrain_layout()) ||
         IsRemovableWhile(instruction,
                          remove_cross_partition_collective_ops))) {
      dead_roots.push_back(instruction);
    }
  }

  for (HloInstruction* dead_root : dead_roots) {
    VLOG(1) << "Removing dead root " << dead_root->ToString()
            << " and its unused operands";
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

Status HloDCE::RecursivelyRemoveDeadComputation(
    HloModule* module, HloComputation* computation,
    absl::flat_hash_map<HloComputation*, int>& live_call_counts) {
  std::vector<HloComputation*> to_be_deleted;
  // First loops all the sub-instructions/sub-computations.
  for (HloInstruction* instruction : computation->instructions()) {
    for (HloComputation* subcomp : instruction->called_computations()) {
      auto iter = live_call_counts.find(subcomp);
      if (iter == live_call_counts.end()) {
        return tsl::errors::Internal(
            "called computation %s not found in live_call_counts table during "
            "HloDCE",
            subcomp->name());
      }

      // Decrements the live call count and sees if there are no more live
      // calls to this computation.
      int live_call_count = --iter->second;
      CHECK_GE(live_call_count, 0);
      if (live_call_count == 0) {
        to_be_deleted.push_back(subcomp);
        live_call_counts.erase(iter);
      }
    }
  }
  VLOG(1) << "Removing dead computation " << computation->name();
  // After looping called subcomputations, now safe to delete the computation.
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(computation));

  // Only remove the to be deleted subcomputations now after 'computation' has
  // been removed. Otherwise we might still have pointers to subcomputations
  // that we want to delete.
  for (HloComputation* subcomp : to_be_deleted) {
    TF_RETURN_IF_ERROR(
        RecursivelyRemoveDeadComputation(module, subcomp, live_call_counts));
  }
  return OkStatus();
}

absl::StatusOr<bool> HloDCE::RecursivelyRemoveDeadComputations(
    HloModule* module) {
  // Tracks whether any dead code is eliminated by this pass.
  bool module_contains_dead_code = false;

  // First, collect the computations that are
  // referenced by some remaining instruction. We need to record this as a
  // refcount map rather than a set since we cannot guarantee that control
  // flow flattening has been done and there may be multiple call sites.
  absl::flat_hash_map<HloComputation*, int> live_computation_call_count;
  if (HloComputation* entry_computation = module->entry_computation()) {
    ++live_computation_call_count[entry_computation];
  }
  // Account for all threads' caller when counting a sub computation's live call
  // count.
  for (auto* computation : module->MakeComputationPostOrder()) {
    for (auto* instruction : computation->instructions()) {
      for (auto* subcomp : instruction->called_computations()) {
        ++live_computation_call_count[subcomp];
      }
    }
  }

  // Find dead computations.
  for (auto* computation : module->MakeComputationPostOrder()) {
    // Finds all "top-level" dead computations not called by any instructions.
    // contains(comp) = true and live_computation_call_count[comp] = 0 also
    // implies that the computation is dead, but is nested in other dead
    // computations. These inner computations are ignored here since they will
    // be removed recursing through other computations.
    if (!live_computation_call_count.contains(computation)) {
      TF_RETURN_IF_ERROR(RecursivelyRemoveDeadComputation(
          module, computation, live_computation_call_count));
      module_contains_dead_code = true;
    }
  }
  return module_contains_dead_code;
}

absl::StatusOr<bool> HloDCE::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  VLOG(2) << "Before dce:";
  XLA_VLOG_LINES(2, module->ToString());

  // Run DCE on each computation.
  for (auto* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_ASSIGN_OR_RETURN(
        bool changed_for_computation,
        RunOnComputation(computation, remove_cross_partition_collective_ops_));
    changed |= changed_for_computation;
  }

  // Now DCE HloComputations.  Keep doing passes through the module until no
  // more computations can be eliminated. The function removes all
  // subcomputations that can be proved to have no remaining live callers.
  TF_ASSIGN_OR_RETURN(bool module_contains_dead_code,
                      RecursivelyRemoveDeadComputations(module));
  changed |= module_contains_dead_code;

  VLOG(2) << "After dce:";
  XLA_VLOG_LINES(2, module->ToString());

  return changed;
}

}  // namespace xla

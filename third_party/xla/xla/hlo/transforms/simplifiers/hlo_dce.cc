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

#include "xla/hlo/transforms/simplifiers/hlo_dce.h"

#include <cstdint>
#include <iterator>
#include <set>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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

// Returns true if it found and removed unused outputs.
absl::StatusOr<bool> RemoveMultiOutputFusionsUnusedOutputs(
    HloComputation* computation) {
  HloInstruction* fusion_instruction = computation->FusionInstruction();
  if (!fusion_instruction) {
    return false;
  }

  if (computation->root_instruction()->opcode() != HloOpcode::kTuple ||
      computation->root_instruction()->has_sharding() ||
      !fusion_instruction->output_operand_aliasing().empty() ||
      fusion_instruction->HasControlDependencies() ||
      fusion_instruction->IsCustomFusion()) {
    return false;
  }

  // The order of the used outputs is relevant for the algorithm below.
  std::set<int64_t> used_tuple_elements;

  // We only support this cleanup if all users of the fusion instruction are
  // GetTupleElement ops, and there is at least one user of
  // 'fusion_instruction'.
  if (fusion_instruction->users().empty()) {
    return false;
  }

  for (HloInstruction* gte : fusion_instruction->users()) {
    if (gte->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    used_tuple_elements.insert(gte->tuple_index());
  }

  // If all outputs are used, nothing to clean up.
  if (used_tuple_elements.size() ==
      computation->root_instruction()->operand_count()) {
    return false;
  }

  std::vector<Shape> tuple_shapes;
  tuple_shapes.reserve(used_tuple_elements.size());
  for (int64_t tuple_index : used_tuple_elements) {
    tuple_shapes.push_back(
        fusion_instruction->shape().tuple_shapes(tuple_index));
  }
  Shape new_shape = tuple_shapes.size() == 1
                        ? tuple_shapes[0]
                        : ShapeUtil::MakeTupleShape(tuple_shapes);
  *fusion_instruction->mutable_shape() = std::move(new_shape);

  // Update the users of the old fusion instruction.
  if (tuple_shapes.size() > 1) {
    for (HloInstruction* gte : fusion_instruction->users()) {
      auto it = used_tuple_elements.lower_bound(gte->tuple_index());
      int64_t new_tuple_index = std::distance(used_tuple_elements.begin(), it);
      gte->set_tuple_index(new_tuple_index);
    }
  } else {
    // Since we iterate over users while removing them .. make a local copy
    // first.
    std::vector<HloInstruction*> users(fusion_instruction->users());
    for (HloInstruction* gte : users) {
      // Replace and change control successors to be dependent on the fusion
      // instruction itself.
      TF_ASSIGN_OR_RETURN(std::ignore, gte->parent()->ReplaceInstruction(
                                           gte, fusion_instruction,
                                           /*preserve_sharding=*/true,
                                           /*relay_control_dependency=*/true));
    }
  }

  // Update the root of the fusion computation.
  if (tuple_shapes.size() > 1) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(used_tuple_elements.size());
    for (int64_t tuple_index : used_tuple_elements) {
      new_operands.push_back(
          computation->root_instruction()->mutable_operand(tuple_index));
    }
    auto new_tuple =
        computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
    TF_RETURN_IF_ERROR(computation->ReplaceInstructionWithDifferentShape(
        computation->root_instruction(), new_tuple));
  } else {
    TF_RETURN_IF_ERROR(
        computation->root_instruction()->ReplaceAllUsesWithDifferentShape(
            computation->root_instruction()->mutable_operand(
                *used_tuple_elements.begin())));
  }

  // We always updated the fusion if we got here.
  return true;
}

}  // namespace

/*static*/ absl::StatusOr<bool> HloDCE::RunOnComputation(
    HloComputation* computation, bool remove_cross_partition_collective_ops) {
  // We do this first, because it may create dead roots which we can clean up
  // next.
  TF_ASSIGN_OR_RETURN(bool changed,
                      RemoveMultiOutputFusionsUnusedOutputs(computation));

  // Remove any dead roots and their dead transitive operands. Collect
  // them into a separate list first to avoid problems with iterating through
  // the computation's instruction while simultaneously removing instructions.
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
  return changed;
}

absl::StatusOr<bool> HloDCE::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  VLOG(2) << "Before dce; threads: " << absl::StrJoin(execution_threads, ",");
  XLA_VLOG_LINES(2, module->ToString());

  // Run DCE on each computation. Visit callers before callees so that we
  // cleanup dead get-tuple-element users of MultiOutput fusions before cleaning
  // up the fusion computation. If the same callee is referred to by multiple
  // callers we'll only visit the first caller before visiting the callee, but
  // that's ok for the use case of fusion computations that should have a unique
  // calling instruction anyway.
  absl::flat_hash_set<HloComputation*> to_remove;
  // Use computations from all execution threads when determining reachability.
  for (HloComputation* computation : module->computations()) {
    to_remove.insert(computation);
  }

  std::stack<HloComputation*> agenda;
  agenda.push(module->entry_computation());
  to_remove.erase(module->entry_computation());
  while (!agenda.empty()) {
    HloComputation* computation = agenda.top();
    agenda.pop();

    if (execution_threads.empty() ||
        execution_threads.contains(computation->execution_thread())) {
      TF_ASSIGN_OR_RETURN(
          bool computation_changed,
          RunOnComputation(computation,
                           remove_cross_partition_collective_ops_));
      changed |= computation_changed;
    }

    for (auto* instruction : computation->instructions()) {
      for (HloComputation* called_computation :
           instruction->called_computations()) {
        if (to_remove.erase(called_computation) > 0) {
          agenda.push(called_computation);
        }
      }
    }
  }
  for (auto computation : to_remove) {
    // Only remove computations from the specified execution threads.
    if (execution_threads.empty() ||
        execution_threads.contains(computation->execution_thread())) {
      TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(computation));
    }
  }
  changed |= !to_remove.empty();

  if (changed) {
    VLOG(2) << "After dce:";
    XLA_VLOG_LINES(2, module->ToString());
  }

  return changed;
}

}  // namespace xla

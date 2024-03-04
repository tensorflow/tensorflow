/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/host_offload_legalize.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_value.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

// Find an annotation moving up. Meant to find an annotation from a DUS operand.
HloInstruction* FindAnnotationToUpdate(HloInstruction* instr) {
  while (!instr->IsCustomCall(
      host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
    if (instr->user_count() != 1 || (instr->opcode() != HloOpcode::kBitcast &&
                                     instr->opcode() != HloOpcode::kReshape)) {
      return nullptr;
    }
    instr = instr->mutable_operand(0);
  }
  return instr;
}

// Find a DUS starting from an annotation.
HloInstruction* FindDUSFromAnnotation(HloInstruction* instr) {
  while (instr->opcode() != HloOpcode::kDynamicUpdateSlice) {
    if (instr->user_count() != 1 || (instr->opcode() != HloOpcode::kBitcast &&
                                     instr->opcode() != HloOpcode::kReshape)) {
      break;
    }
    instr = instr->users()[0];
  }
  return instr;
}

// Make sure that broadcasts are duplicated for each use.
StatusOr<bool> DuplicateBroadcastForEachUse(HloModule* module) {
  bool split_at_least_one = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kBroadcast ||
          !instruction->HasConstantOperand()) {
        continue;
      }
      absl::InlinedVector<HloUse, 8> uses;
      for (HloInstruction* user : instruction->users()) {
        for (int64_t i = 0; i < user->operand_count(); ++i) {
          if (user->operand(i) != instruction) {
            continue;
          }
          uses.push_back(HloUse{user, i, /*operand_index=*/{}});
        }
      }

      if (uses.size() <= 1) {
        VLOG(5) << "Skipping broadcast " << instruction->ToString()
                << " which has " << uses.size() << " uses";
        continue;
      }

      VLOG(5) << "Splitting broadcast " << instruction->ToString()
              << " which has " << uses.size() << " uses";
      split_at_least_one = true;
      // Don't create a new broadcast for the first use; we can still use the
      // original.
      for (int i = 1; i < uses.size(); ++i) {
        const HloUse& use = uses[i];
        HloInstruction* new_broadcast =
            instruction->parent()->AddInstruction(instruction->Clone());
        VLOG(5) << "New broadcast " << new_broadcast->ToString();
        TF_RETURN_IF_ERROR(use.instruction->ReplaceOperandWith(
            use.operand_number, new_broadcast));
      }
    }
  }
  return split_at_least_one;
}

// Walk up in the chain of memory offloaded instructions. Status not-ok when
// an instructions not supported or end of chain reached.
StatusOr<std::pair<HloInstruction*, int>> WalkUpMemoryOffload(
    std::pair<HloInstruction*, int> current_value,
    const CallGraph& call_graph) {
  // TODO(maggioni): Verify that set of instructions supported in chain by
  // legalization is in sync with host_offloader.
  auto& [instruction, index] = current_value;
  // Walk up to find definition
  switch (instruction->opcode()) {
    case HloOpcode::kGetTupleElement: {
      CHECK_EQ(index, -1);
      return std::make_pair(instruction->mutable_operand(0),
                            instruction->tuple_index());
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape: {
      return std::make_pair(instruction->mutable_operand(0), index);
    }
    case HloOpcode::kTuple: {
      return std::make_pair(instruction->mutable_operand(index), -1);
    }
    case HloOpcode::kOptimizationBarrier: {
      return std::make_pair(instruction->mutable_operand(0), index);
    }
    case HloOpcode::kWhile: {
      HloComputation* while_body = instruction->while_body();
      HloInstruction* root = while_body->root_instruction();
      CHECK_EQ(root->opcode(), HloOpcode::kTuple);
      return std::make_pair(root, index);
    }
    case HloOpcode::kParameter: {
      CHECK_NE(instruction->parent(),
               instruction->GetModule()->entry_computation());
      auto callers = call_graph.GetComputationCallers(instruction->parent());
      if (callers.size() != 1) {
        return absl::InvalidArgumentError(
            "Expected to be called only by one caller");
      }
      auto* caller = callers[0];
      if (caller->opcode() != HloOpcode::kWhile) {
        return absl::InvalidArgumentError(
            "Expected to be called by a while loop");
      }
      return std::make_pair(caller->mutable_operand(0), index);
    }
    case HloOpcode::kDynamicUpdateSlice: {
      return std::make_pair(instruction->mutable_operand(0), index);
    }
    case HloOpcode::kCustomCall: {
      if (!instruction->IsCustomCall("AllocateBuffer") &&
          !instruction->IsCustomCall(
              host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
        return absl::InvalidArgumentError(
            "Expected AllocateBuffer or MoveToHost custom-call");
      }
      return std::make_pair(instruction, index);
    }
    case HloOpcode::kBroadcast: {
      auto* broadcast_operand = instruction->mutable_operand(0);
      if (broadcast_operand->opcode() != HloOpcode::kConstant) {
        return absl::InvalidArgumentError("Expected a constant as operand");
      }
      if (!ShapeUtil::IsEffectiveScalar(broadcast_operand->shape())) {
        return absl::InvalidArgumentError("Expected a scalar broadcast");
      }
      return std::make_pair(instruction, index);
    }
    default: {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid opcode %s", instruction->ToString()));
    }
  }
}

// Walk down in the chain of memory offloaded instructions. Status not-ok when
// an instructions not supported or end of chain reached.
StatusOr<std::pair<HloInstruction*, int>> WalkDownMemoryOffload(
    const std::pair<HloInstruction*, int64_t>& current_value,
    const CallGraph& call_graph) {
  // TODO(maggioni): Verify that set of instructions supported in chain by
  // legalization is in sync with host_offloader.
  VLOG(5) << "Current value in progress: " << current_value.first->ToString()
          << " idx: " << current_value.second;
  auto find_gte_for_idx = [](HloInstruction* instr,
                             int idx) -> StatusOr<HloInstruction*> {
    HloInstruction* gte = nullptr;
    for (HloInstruction* user : instr->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        return absl::InvalidArgumentError(
            "Expected users to be only get-tuple-elements");
      }
      if (user->tuple_index() != idx) {
        continue;
      }
      if (gte != nullptr) {
        return absl::InvalidArgumentError(
            "Expected to find only one gte per index.");
      }
      gte = user;
    }
    return gte;
  };
  if (current_value.first->user_count() == 0) {
    if (current_value.first->parent()->root_instruction() ==
        current_value.first) {
      auto callers =
          call_graph.GetComputationCallers(current_value.first->parent());
      if (callers.size() != 1 || callers[0]->opcode() != HloOpcode::kWhile) {
        return absl::InvalidArgumentError(
            "Expected to be called only by one caller and caller be a While");
      }
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          find_gte_for_idx(callers[0], current_value.second));
      return std::make_pair(gte, -1);
    }
  }
  if (current_value.first->user_count() != 1) {
    if (current_value.first->opcode() == HloOpcode::kParameter &&
        current_value.first->shape().IsTuple()) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * gte,
          find_gte_for_idx(current_value.first, current_value.second));
      return std::make_pair(gte, -1);
    }
    return absl::InvalidArgumentError("Number of users > 1");
  }
  HloInstruction* user = current_value.first->users()[0];
  switch (user->opcode()) {
    case HloOpcode::kGetTupleElement: {
      CHECK_NE(user->tuple_index(), -1);
      if (user->tuple_index() != current_value.second) {
        return absl::InvalidArgumentError("Invalid index for gte");
      }
      return std::make_pair(user, static_cast<int>(-1));
    }
    case HloOpcode::kTuple: {
      auto output_indices = user->OperandIndices(current_value.first);
      if (output_indices.size() != 1) {
        return absl::InvalidArgumentError(
            "Expected operand to be used only once in the tuple.");
      }
      return std::make_pair(user, output_indices[0]);
    }
    case HloOpcode::kOptimizationBarrier: {
      return std::make_pair(user, current_value.second);
    }
    case HloOpcode::kWhile: {
      HloComputation* while_body = user->while_body();
      HloInstruction* parameter = while_body->parameter_instruction(0);
      return std::make_pair(parameter, current_value.second);
    }
    case HloOpcode::kDynamicUpdateSlice: {
      if (user->OperandIndices(current_value.first)[0] != 0) {
        return absl::InvalidArgumentError(
            "Expected to be used by first operand of dynamic-update-slice");
      }
      return std::make_pair(user, current_value.second);
    }
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice: {
      return std::make_pair(user, current_value.second);
    }
    default: {
      return absl::InvalidArgumentError("Unrecognized user opcode");
    }
  }
}

StatusOr<bool> ProcessAnnotationForCopyMovement(
    HloInstruction* instruction, const CallGraph* call_graph,
    absl::flat_hash_set<HloInstruction*>& processed_annotations) {
  HloInstruction* starting_instr =
      FindDUSFromAnnotation(instruction->users()[0]);
  VLOG(3) << "Dus or Annotation: " << starting_instr->ToString();
  // If it's the pure copy case reset instruction.
  if (starting_instr->opcode() != HloOpcode::kDynamicUpdateSlice) {
    starting_instr = instruction;
  }
  std::pair<HloInstruction*, int> current_value =
      std::make_pair(starting_instr, -1);
  // Walk down the chain and find a layout changing copy to move.
  auto current_value_down = WalkDownMemoryOffload(current_value, *call_graph);
  while (current_value_down.ok()) {
    current_value = current_value_down.value();
    current_value_down = WalkDownMemoryOffload(current_value, *call_graph);
  }
  HloInstruction* last_value = nullptr;
  if (current_value.first->user_count() == 1) {
    if (current_value.first->users()[0]->opcode() != HloOpcode::kCopy) {
      VLOG(5) << "No copy";
      return false;
    }
    last_value = current_value.first;
  } else {
    VLOG(5) << "Users > 1";
    return false;
  }
  // Found a copy that would block offloading. Walk up to find all annotations
  // to update (required in case there are multiple insertions in the buffer).
  std::vector<std::pair<HloInstruction*, HloInstruction*>>
      annotations_to_update;
  if (current_value.first->IsCustomCall(
          host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
    annotations_to_update.push_back(
        std::make_pair(current_value.first, current_value.first));
  }
  while (true) {
    VLOG(10) << "Current value before: " << current_value.first->ToString();
    auto current_value_up = WalkUpMemoryOffload(current_value, *call_graph);
    if (!current_value_up.ok() || current_value_up.value() == current_value) {
      break;
    }
    current_value = current_value_up.value();
    VLOG(10) << "Current value after: "
             << current_value_up.value().first->ToString();
    HloInstruction* annotation = current_value_up.value().first;
    if (annotation->opcode() == HloOpcode::kDynamicUpdateSlice) {
      HloInstruction* real_annotation =
          FindAnnotationToUpdate(annotation->mutable_operand(1));
      if (real_annotation->IsCustomCall(
              host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
        annotations_to_update.push_back(
            std::make_pair(annotation, real_annotation));
      } else {
        annotations_to_update.push_back(std::make_pair(annotation, annotation));
      }
    }
    if (annotation->IsCustomCall(
            host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
      annotations_to_update.push_back(std::make_pair(
          current_value_up.value().first, current_value_up.value().first));
    }
  }
  if (current_value.first->parent() != last_value->parent()) {
    VLOG(5) << "Parent mismatch";
    return false;
  }
  std::vector<std::pair<HloInstruction*, int>> instructions_to_change(
      1, current_value);
  // Do a final walkdown from the top to collect all the instructions that need
  // their shape updated.
  while (true) {
    current_value_down = WalkDownMemoryOffload(current_value, *call_graph);
    if (!current_value_down.ok()) {
      break;
    }
    VLOG(5) << "Current value last down: "
            << current_value_down.value().first->ToString();
    current_value = current_value_down.value();
    instructions_to_change.push_back(current_value_down.value());
  }
  if (current_value.first != last_value) {
    return false;
  }
  if (instructions_to_change.empty()) {
    return false;
  }
  HloInstruction* copy_to_move = last_value->users()[0];
  VLOG(3) << "Copy to move: " << copy_to_move->ToString();
  VLOG(3) << "Instructions to change num: " << instructions_to_change.size();
  for (auto& instruction : instructions_to_change) {
    const Shape& shape =
        instruction.first->shape().IsTuple()
            ? instruction.first->shape().tuple_shapes()[instruction.second]
            : instruction.first->shape();
    if (shape != last_value->shape()) {
      return false;
    }
    VLOG(3) << "\tInstruction to change: " << instruction.first->ToString();
  }
  // Helper to update the shape.
  auto update_shape_layout =
      [&](const std::pair<HloInstruction*, int>& instruction) {
        // Update shape. Tuple shape vs array shape.
        if (instruction.second != -1) {
          *instruction.first->mutable_shape()
               ->mutable_tuple_shapes(instruction.second)
               ->mutable_layout() = copy_to_move->shape().layout();
        } else {
          *instruction.first->mutable_shape()->mutable_layout() =
              copy_to_move->shape().layout();
        }
      };
  for (auto& instruction : instructions_to_change) {
    update_shape_layout(instruction);
    if (instruction.first->opcode() == HloOpcode::kParameter) {
      auto callers =
          call_graph->GetComputationCallers(instruction.first->parent());
      if (callers[0]->opcode() == HloOpcode::kWhile) {
        update_shape_layout(std::make_pair(callers[0], instruction.second));
      }
    }
  }
  VLOG(3) << "Annotations to update num: " << annotations_to_update.size();
  // Update annotations. The copy needs to be placed before every annotation.
  for (auto& [insertion_point, annotation] : annotations_to_update) {
    VLOG(3) << "\tAnnotation to update (insertion pt): "
            << insertion_point->ToString();
    VLOG(3) << "\tAnnotation to update (instr): " << annotation->ToString();
    processed_annotations.insert(annotation);
    // Move the annotation first just before dynamic-update-slice to avoid shape
    // changes.
    if (insertion_point != annotation) {
      CHECK_EQ(insertion_point->opcode(), HloOpcode::kDynamicUpdateSlice);
      TF_RETURN_IF_ERROR(insertion_point->ReplaceOperandWith(
          1, insertion_point->AddInstruction(annotation->CloneWithNewOperands(
                 insertion_point->operand(1)->shape(),
                 {insertion_point->mutable_operand(1)}))));
      TF_RETURN_IF_ERROR(
          annotation->ReplaceAllUsesWith(annotation->mutable_operand(0)));
      TF_RETURN_IF_ERROR(annotation->parent()->RemoveInstruction(annotation));
      insertion_point = insertion_point->mutable_operand(1);
      annotation = insertion_point;
    }
    Shape instruction_shape = insertion_point->shape();
    *instruction_shape.mutable_layout() = copy_to_move->shape().layout();
    auto* new_copy =
        insertion_point->AddInstruction(copy_to_move->CloneWithNewOperands(
            instruction_shape, {insertion_point->mutable_operand(0)}));
    TF_RETURN_IF_ERROR(
        insertion_point->mutable_operand(0)->ReplaceAllUsesWithDifferentShape(
            new_copy));
  }
  TF_RETURN_IF_ERROR(copy_to_move->ReplaceAllUsesWithDifferentShape(
      copy_to_move->mutable_operand(0)));
  TF_RETURN_IF_ERROR(copy_to_move->parent()->RemoveInstruction(copy_to_move));
  return true;
}

// Fixes layout changing copies in between on the path to users.
StatusOr<bool> FixupInterveningCopies(
    const std::vector<HloInstruction*>& copy_to_host_annotations,
    const CallGraph* call_graph) {
  absl::flat_hash_set<HloInstruction*> processed_annotations;
  bool changed = false;
  for (HloInstruction* instruction : copy_to_host_annotations) {
    if (processed_annotations.count(instruction)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool changed_annotation_for_copy_movement,
                        ProcessAnnotationForCopyMovement(
                            instruction, call_graph, processed_annotations));
    changed |= changed_annotation_for_copy_movement;
  }
  return changed;
}

}  // namespace

StatusOr<bool> HostOffloadLegalize::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Split broadcasts so that each HloUse of a broadcast instruction will get
  // its own copy.
  // TODO(b/319293925): Do not blindly duplicate all broadcasts, instead do it
  // only when necessary.
  TF_ASSIGN_OR_RETURN(bool duplicated_at_least_one_broadcast,
                      DuplicateBroadcastForEachUse(module));
  if (duplicated_at_least_one_broadcast) {
    changed = true;
  }
  if (!after_layout_) {
    return changed;
  }
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  std::vector<HloInstruction*> copy_to_host_annotations;

  // Iterate over all instructions and look for XLA host offload annotations.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }
      if (instruction->custom_call_target() ==
          host_memory_offload_annotations::kMoveToHostCustomCallTarget) {
        copy_to_host_annotations.push_back(instruction);
      }
    }
  }
  // Fixup layout changing copies that are in between memory offloaded sections.
  // Move them before the data is moved to the host.
  TF_ASSIGN_OR_RETURN(
      bool changed_intervening_copies,
      FixupInterveningCopies(copy_to_host_annotations, call_graph.get()));
  changed |= changed_intervening_copies;

  return changed;
}

}  // namespace xla

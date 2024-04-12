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
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

constexpr std::array<HloOpcode, 2> kUsersOpcodes = {HloOpcode::kSlice,
                                                    HloOpcode::kDynamicSlice};

// Find an annotation moving up. Meant to find an annotation from a DUS operand.
HloInstruction* FindToHostAnnotationToUpdate(HloInstruction* instr) {
  while (!instr->IsCustomCall(
      host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
    if ((instr->opcode() != HloOpcode::kBitcast &&
         instr->opcode() != HloOpcode::kCopy &&
         instr->opcode() != HloOpcode::kReshape) ||
        instr->mutable_operand(0)->user_count() != 1) {
      return nullptr;
    }
    instr = instr->mutable_operand(0);
  }
  return instr;
}

// Find an annotation moving up. Meant to find an annotation from a DUS
// instruction.
HloInstruction* FindToDeviceAnnotationToUpdate(HloInstruction* instr) {
  while (!instr->IsCustomCall(
      host_memory_offload_annotations::kMoveToDeviceCustomCallTarget)) {
    if (instr->user_count() != 1 ||
        (instr->opcode() != HloOpcode::kBitcast &&
         instr->opcode() != HloOpcode::kReshape &&
         instr->opcode() != HloOpcode::kCopy &&
         !absl::c_linear_search(kUsersOpcodes, instr->opcode()))) {
      return nullptr;
    }
    instr = instr->users()[0];
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
absl::StatusOr<bool> DuplicateBroadcastForEachUse(HloModule* module) {
  bool split_at_least_one = false;
  for (HloComputation* computation : module->computations()) {
    std::vector<HloInstruction*> broadcasts;
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kBroadcast ||
          !instruction->HasConstantOperand()) {
        continue;
      }
      broadcasts.push_back(instruction);
    }
    for (HloInstruction* instruction : broadcasts) {
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
// Walks one instruction at a time.
absl::StatusOr<std::pair<HloInstruction*, int>> WalkUpMemoryOffload(
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
// Walks one instruction at a time, but returns multiple instructions for each
// conforming user.
absl::StatusOr<std::vector<std::pair<HloInstruction*, int>>>
WalkDownMemoryOffload(const std::pair<HloInstruction*, int64_t>& current_value,
                      const CallGraph& call_graph) {
  // TODO(maggioni): Verify that set of instructions supported in chain by
  // legalization is in sync with host_offloader.
  VLOG(5) << "Current value in progress: " << current_value.first->ToString()
          << " idx: " << current_value.second;
  std::vector<std::pair<HloInstruction*, int>> results;
  auto add_gte_for_idx = [&results](HloInstruction* instr, int idx) -> Status {
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
      results.push_back(std::make_pair(user, -1));
    }
    return OkStatus();
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
      TF_RETURN_IF_ERROR(add_gte_for_idx(callers[0], current_value.second));
      return results;
    }
  }
  if (current_value.first->opcode() == HloOpcode::kParameter &&
      current_value.first->shape().IsTuple()) {
    TF_RETURN_IF_ERROR(
        add_gte_for_idx(current_value.first, current_value.second));
    return results;
  }
  for (HloInstruction* user : current_value.first->users()) {
    switch (user->opcode()) {
      case HloOpcode::kGetTupleElement: {
        CHECK_NE(user->tuple_index(), -1);
        if (user->tuple_index() != current_value.second) {
          continue;
        }
        results.push_back(std::make_pair(user, -1));
        break;
      }
      case HloOpcode::kTuple: {
        auto output_indices = user->OperandIndices(current_value.first);
        if (output_indices.size() != 1) {
          return absl::InvalidArgumentError(
              "Expected operand to be used only once in the tuple.");
        }
        results.push_back(std::make_pair(user, output_indices[0]));
        break;
      }
      case HloOpcode::kOptimizationBarrier: {
        results.push_back(std::make_pair(user, current_value.second));
        break;
      }
      case HloOpcode::kWhile: {
        HloComputation* while_body = user->while_body();
        HloInstruction* parameter = while_body->parameter_instruction(0);
        results.push_back(std::make_pair(parameter, current_value.second));
        break;
      }
      case HloOpcode::kDynamicUpdateSlice: {
        if (user->OperandIndices(current_value.first)[0] != 0) {
          return absl::InvalidArgumentError(
              "Expected to be used by first operand of dynamic-update-slice");
        }
        results.push_back(std::make_pair(user, current_value.second));
        break;
      }
      case HloOpcode::kCustomCall: {
        if (user->IsCustomCall(host_memory_offload_annotations::
                                   kMoveToDeviceCustomCallTarget)) {
          results.push_back(std::make_pair(user, current_value.second));
          break;
        }
        return absl::InvalidArgumentError("Invalid custom-call found.");
      }
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kReshape:
      case HloOpcode::kSlice: {
        results.push_back(std::make_pair(user, current_value.second));
        break;
      }
      default: {
        return absl::InvalidArgumentError("Unrecognized user opcode");
      }
    }
  }
  return results;
}

absl::StatusOr<bool> ProcessAnnotationForCopyMovement(
    HloInstruction* instruction, const CallGraph* call_graph,
    absl::flat_hash_set<HloInstruction*>& processed_annotations,
    std::vector<HloInstruction*>& to_remove) {
  auto is_entry_computation_parameter = [](HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kParameter &&
           instruction->parent()->IsEntryComputation();
  };

  if (instruction->IsRoot()) {
    return false;
  }
  HloInstruction* starting_instr =
      FindDUSFromAnnotation(instruction->users().at(0));
  // If it's the pure copy case reset instruction.
  if (starting_instr->opcode() != HloOpcode::kDynamicUpdateSlice) {
    starting_instr = instruction;
  }
  VLOG(3) << "Dus or Annotation: " << starting_instr->ToString();
  std::pair<HloInstruction*, int> current_value =
      std::make_pair(starting_instr, -1);
  // Found a copy that would block offloading. Walk up to find all annotations
  // to update (required in case there are multiple insertions in the buffer).
  processed_annotations.insert(current_value.first);
  if (!current_value.first->IsCustomCall(
          host_memory_offload_annotations::kMoveToHostCustomCallTarget) &&
      !is_entry_computation_parameter(current_value.first)) {
    CHECK_EQ(current_value.first->opcode(), HloOpcode::kDynamicUpdateSlice);
    while (true) {
      VLOG(10) << "Current value before: " << current_value.first->ToString();
      auto current_value_up = WalkUpMemoryOffload(current_value, *call_graph);
      // Invalid upward walking means the chain is unrecognized.
      if (!current_value_up.ok()) {
        return false;
      }
      // This means we encountered a broadcast with constant 0 expansion.
      if (current_value_up.value() == current_value) {
        break;
      }
      current_value = current_value_up.value();
      VLOG(10) << "Current value after: " << current_value.first->ToString();
      HloInstruction* annotation = current_value.first;
      if (annotation->opcode() == HloOpcode::kDynamicUpdateSlice) {
        HloInstruction* real_annotation =
            FindToHostAnnotationToUpdate(annotation->mutable_operand(1));
        // Check if this dynamic-update-slice doesn't have an annotation
        // attached.
        if (!real_annotation->IsCustomCall(
                host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
          return false;
        }
      }
    }
  }
  std::vector<std::pair<HloInstruction*, int>> copies_to_move;
  // Do a final walkdown from the top to collect all the instructions that need
  // their shape updated.
  std::vector<std::pair<HloInstruction*, int>> stack(1, current_value);
  while (!stack.empty()) {
    VLOG(5) << "Current value before down: " << stack.back().first->ToString();
    if (absl::c_linear_search(kUsersOpcodes, stack.back().first->opcode()) ||
        stack.back().first->IsCustomCall(
            host_memory_offload_annotations::kMoveToDeviceCustomCallTarget)) {
      HloInstruction* annotation =
          FindToDeviceAnnotationToUpdate(stack.back().first);
      if (!annotation ||
          !annotation->IsCustomCall(
              host_memory_offload_annotations::kMoveToDeviceCustomCallTarget)) {
        VLOG(5) << "Couldn't find annotation for consumer instruction in chain";
        return false;
      }

      // Fix up while body's root instruction shape along the way.
      if (annotation->IsCustomCall(
              host_memory_offload_annotations::kMoveToDeviceCustomCallTarget)) {
        for (HloInstruction* user : annotation->users()) {
          HloInstruction* root_instruction =
              annotation->parent()->root_instruction();
          if (root_instruction == user &&
              root_instruction->opcode() == HloOpcode::kTuple) {
            auto callers =
                call_graph->GetComputationCallers(annotation->parent());
            if (callers.size() != 1 ||
                callers[0]->opcode() != HloOpcode::kWhile) {
              return absl::InvalidArgumentError(
                  "Expected to be called only by one caller and caller be a "
                  "While");
            }
            for (int i = 0; i < user->operands().size(); i++) {
              if (user->operands()[i] == annotation &&
                  annotation->operand(0)->opcode() ==
                      HloOpcode::kGetTupleElement &&
                  annotation->operand(0)->operand(0)->opcode() ==
                      HloOpcode::kParameter &&
                  annotation->operand(0)->tuple_index() == i) {
                // A special case where move-to-device is put into the result
                // tuple element at the same index as where the move-to-device
                // gets the data from. In this case, while loop's result tuple
                // should not use move-to-device since at loop entry it's still
                // on host.
                user->ReplaceOperandWith(i, annotation->mutable_operand(0))
                    .IgnoreError();
              }
            }
          }
        }
      }
      stack.pop_back();
      continue;
    }
    auto current_value_down = WalkDownMemoryOffload(stack.back(), *call_graph);
    if (!current_value_down.ok()) {
      VLOG(5) << "Current value down failed: " << current_value_down.status();
      break;
    }
    stack.pop_back();
    stack.insert(stack.end(), current_value_down.value().begin(),
                 current_value_down.value().end());
    for (auto& instruction : current_value_down.value()) {
      VLOG(5) << "Current value last down: " << stack.back().first->ToString();
      if (instruction.first->opcode() == HloOpcode::kCopy) {
        copies_to_move.push_back(instruction);
      }
    }
  }

  auto update_shape_layout =
      [&](const std::pair<HloInstruction*, int>& instruction,
          HloInstruction* copy_to_move) {
        VLOG(5) << "Update shape layout: " << instruction.first->ToString()
                << " " << instruction.second;
        // Update shape. Tuple shape vs array shape.
        if (instruction.second != -1) {
          *instruction.first->mutable_shape()
               ->mutable_tuple_shapes(instruction.second)
               ->mutable_layout() = copy_to_move->operand(0)->shape().layout();
        } else {
          *instruction.first->mutable_shape()->mutable_layout() =
              copy_to_move->operand(0)->shape().layout();
        }

        if (instruction.first->opcode() == HloOpcode::kWhile) {
          // Fix up while body's root instruction shape and condition's
          // parameter shape for while loops.
          Shape new_shape = copy_to_move->operand(0)->shape();
          *instruction.first->while_body()
               ->root_instruction()
               ->mutable_shape()
               ->mutable_tuple_shapes(instruction.second)
               ->mutable_layout() = new_shape.layout();
          *instruction.first->while_condition()
               ->parameter_instruction(0)
               ->mutable_shape()
               ->mutable_tuple_shapes(instruction.second)
               ->mutable_layout() = new_shape.layout();
        }
      };

  // Process all copies one at a time from the last to the first and push it to
  // its specific user.
  while (!copies_to_move.empty()) {
    auto& copy_to_move = copies_to_move.back();
    VLOG(5) << "Copy to move: " << copy_to_move.first->ToString();
    stack.clear();
    stack.push_back(copy_to_move);
    while (!stack.empty()) {
      VLOG(5) << "Current value before down: " << stack.back().first->ToString()
              << " " << stack.back().second;
      auto current_value_down =
          WalkDownMemoryOffload(stack.back(), *call_graph);
      if (!current_value_down.ok()) {
        VLOG(5) << "Current value down failed: " << current_value_down.status();
        break;
      }
      for (auto& instruction : current_value_down.value()) {
        update_shape_layout(instruction, copy_to_move.first);
        if (instruction.first->opcode() == HloOpcode::kParameter) {
          auto callers =
              call_graph->GetComputationCallers(instruction.first->parent());
          if (callers.size() != 1) {
            return absl::InvalidArgumentError(
                "Expected to be called only by one caller");
          }
          auto* caller = callers[0];
          update_shape_layout(std::make_pair(caller, instruction.second),
                              copy_to_move.first);
        }
      }
      stack.pop_back();
      for (auto& instruction : current_value_down.value()) {
        VLOG(5) << "Current value last down: " << instruction.first->ToString();
        CHECK_NE(instruction.first->opcode(), HloOpcode::kCopy)
            << "Copies should be processed in order";
        if (absl::c_linear_search(kUsersOpcodes, instruction.first->opcode()) ||
            instruction.first->IsCustomCall(
                host_memory_offload_annotations::
                    kMoveToDeviceCustomCallTarget)) {
          HloInstruction* annotation =
              FindToDeviceAnnotationToUpdate(instruction.first);
          CHECK_NE(annotation, nullptr)
              << "We already verified we could find an annotation here. "
                 "Something went wrong.";
          HloInstruction* new_annotation = nullptr;
          if (instruction.first->opcode() == HloOpcode::kCustomCall) {
            new_annotation = annotation;
          } else {
            new_annotation = instruction.first->AddInstruction(
                annotation->CloneWithNewOperands(instruction.first->shape(),
                                                 {instruction.first}));
          }
          update_shape_layout(std::make_pair(new_annotation, -1),
                              copy_to_move.first);
          Shape new_copy_shape = new_annotation->shape();
          *new_copy_shape.mutable_layout() =
              copy_to_move.first->shape().layout();
          HloInstruction* new_copy = instruction.first->AddInstruction(
              copy_to_move.first->CloneWithNewOperands(new_copy_shape,
                                                       {new_annotation}));
          std::vector<HloInstruction*> users = instruction.first->users();
          for (auto* use : users) {
            if (use == new_copy || use == new_annotation) {
              continue;
            }
            TF_RETURN_IF_ERROR(
                instruction.first->ReplaceUseWithDifferentShape(use, new_copy));
          }
          // Move the copy here.
          if (new_annotation != annotation) {
            TF_RETURN_IF_ERROR(annotation->ReplaceAllUsesWithDifferentShape(
                annotation->mutable_operand(0)));
            to_remove.push_back(annotation);
          }
          continue;
        }
        // Move the annotation first just before dynamic-update-slice to avoid
        // shape changes.
        if (instruction.first->opcode() == HloOpcode::kDynamicUpdateSlice) {
          HloInstruction* annotation = FindToHostAnnotationToUpdate(
              instruction.first->mutable_operand(1));
          if (annotation == nullptr) {
            CHECK(false);
            return false;
          }
          CHECK(annotation->opcode() == HloOpcode::kCustomCall);
          HloInstruction* new_annotation = instruction.first->AddInstruction(
              annotation->CloneWithNewOperands(
                  instruction.first->operand(1)->shape(),
                  {instruction.first->mutable_operand(1)}));
          TF_RETURN_IF_ERROR(
              instruction.first->ReplaceOperandWith(1, new_annotation));
          TF_RETURN_IF_ERROR(
              annotation->ReplaceAllUsesWith(annotation->mutable_operand(0)));
          processed_annotations.insert(annotation);
          processed_annotations.insert(new_annotation);
          to_remove.push_back(annotation);
        }
        stack.push_back(instruction);
      }
    }
    VLOG(5) << "MOVED: " << copy_to_move.first->ToString();
    TF_RETURN_IF_ERROR(copy_to_move.first->ReplaceAllUsesWithDifferentShape(
        copy_to_move.first->mutable_operand(0)));
    TF_RETURN_IF_ERROR(
        copy_to_move.first->parent()->RemoveInstruction(copy_to_move.first));
    copies_to_move.pop_back();
  }
  return true;
}

// Fixes layout changing copies in between on the path to users.
absl::StatusOr<bool> FixupInterveningCopies(
    const std::vector<HloInstruction*>& copy_to_host_annotations,
    const CallGraph* call_graph) {
  absl::flat_hash_set<HloInstruction*> processed_annotations;
  std::vector<HloInstruction*> annotations_to_remove;
  bool changed = false;
  for (HloInstruction* instruction : copy_to_host_annotations) {
    if (processed_annotations.contains(instruction)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool changed_annotation_for_copy_movement,
                        ProcessAnnotationForCopyMovement(
                            instruction, call_graph, processed_annotations,
                            annotations_to_remove));
    changed |= changed_annotation_for_copy_movement;
  }
  for (HloInstruction* instruction : annotations_to_remove) {
    TF_RETURN_IF_ERROR(instruction->parent()->RemoveInstruction(instruction));
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> HostOffloadLegalize::Run(
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
      if (instruction->opcode() == HloOpcode::kParameter &&
          instruction->parent()->IsEntryComputation()) {
        Shape param_shape =
            module->entry_computation_layout()
                .parameter_layout(instruction->parameter_number())
                .shape();
        // TODO(mingyao): Add support for tuple parameter.
        if (param_shape.has_layout() &&
            param_shape.layout().memory_space() == kHostMemorySpaceColor) {
          copy_to_host_annotations.push_back(instruction);
          continue;
        }
      }

      if (instruction->IsCustomCall(
              host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
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

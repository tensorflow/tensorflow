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

#include "xla/hlo/transforms/host_offload_legalize.h"

#include <array>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_value.h"
#include "xla/service/host_offload_utils.h"
#include "xla/service/memory_annotations.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

bool IsEntryComputationParameter(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kParameter &&
         instruction->parent()->IsEntryComputation();
}

constexpr std::array<HloOpcode, 2> kUsersOpcodes = {HloOpcode::kSlice,
                                                    HloOpcode::kDynamicSlice};

// Find an annotation moving up. Meant to find an annotation from a DUS operand.
HloInstruction* FindToHostAnnotationToUpdate(HloInstruction* instr) {
  while (
      !instr->IsCustomCall(memory_annotations::kMoveToHostCustomCallTarget)) {
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
  while (
      !instr->IsCustomCall(memory_annotations::kMoveToDeviceCustomCallTarget)) {
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

// Find a DUS starting from an annotation on the update operand.
HloInstruction* FindDUSFromAnnotation(HloInstruction* instr) {
  while (instr->opcode() != HloOpcode::kDynamicUpdateSlice) {
    if (instr->user_count() != 1 || (instr->opcode() != HloOpcode::kBitcast &&
                                     instr->opcode() != HloOpcode::kReshape &&
                                     instr->opcode() != HloOpcode::kCopy)) {
      break;
    }
    instr = instr->users()[0];
  }
  return instr;
}

struct InstructionAndIndex {
  HloInstruction* instruction;
  int index;
  InstructionAndIndex(HloInstruction* instruction, int index)
      : instruction(instruction), index(index) {}
  bool operator==(const InstructionAndIndex& other) const {
    return instruction == other.instruction && index == other.index;
  }
};

// Walk up in the chain of memory offloaded instructions. absl::Status not-ok
// when an instructions not supported or end of chain reached. Walks one
// instruction at a time.
// Returns current_value if there is nowhere else to go.
absl::StatusOr<InstructionAndIndex> WalkUpMemoryOffload(
    InstructionAndIndex current_value, const CallGraph& call_graph) {
  // TODO(maggioni): Verify that set of instructions supported in chain by
  // legalization is in sync with host_offloader.
  auto& [instruction, index] = current_value;
  // Walk up to find definition
  switch (instruction->opcode()) {
    case HloOpcode::kGetTupleElement: {
      CHECK_EQ(index, -1);
      return InstructionAndIndex(instruction->mutable_operand(0),
                                 instruction->tuple_index());
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kCopy: {
      return InstructionAndIndex(instruction->mutable_operand(0), index);
    }
    case HloOpcode::kTuple: {
      return InstructionAndIndex(instruction->mutable_operand(index), -1);
    }
    case HloOpcode::kOptimizationBarrier: {
      return InstructionAndIndex(instruction->mutable_operand(0), index);
    }
    case HloOpcode::kWhile: {
      HloComputation* while_body = instruction->while_body();
      HloInstruction* root = while_body->root_instruction();
      CHECK_EQ(root->opcode(), HloOpcode::kTuple);
      return InstructionAndIndex(root, index);
    }
    case HloOpcode::kParameter: {
      if (instruction->parent() ==
          instruction->GetModule()->entry_computation()) {
        // We reached the top. No further to go.
        return current_value;
      }
      std::vector<HloInstruction*> callers =
          call_graph.GetComputationCallers(instruction->parent());
      if (callers.size() != 1) {
        return absl::InvalidArgumentError(
            "Expected to be called only by one caller");
      }
      HloInstruction* caller = callers[0];
      if (caller->opcode() != HloOpcode::kWhile) {
        return absl::InvalidArgumentError(
            "Expected to be called by a while loop");
      }
      return InstructionAndIndex(caller->mutable_operand(0), index);
    }
    case HloOpcode::kDynamicUpdateSlice: {
      return InstructionAndIndex(instruction->mutable_operand(0), index);
    }
    case HloOpcode::kCustomCall: {
      if (!instruction->IsCustomCall("AllocateBuffer") &&
          !instruction->IsCustomCall(
              memory_annotations::kMoveToHostCustomCallTarget)) {
        return absl::InvalidArgumentError(
            "Expected AllocateBuffer or MoveToHost custom-call");
      }
      return InstructionAndIndex(instruction, index);
    }
    case HloOpcode::kBroadcast: {
      HloInstruction* broadcast_operand = instruction->mutable_operand(0);
      if (broadcast_operand->opcode() != HloOpcode::kConstant) {
        return absl::InvalidArgumentError("Expected a constant as operand");
      }
      if (!ShapeUtil::IsEffectiveScalar(broadcast_operand->shape())) {
        return absl::InvalidArgumentError("Expected a scalar broadcast");
      }
      return InstructionAndIndex(instruction, index);
    }
    default: {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid opcode %s", instruction->ToString()));
    }
  }
}

// Walk down in the chain of memory offloaded instructions. absl::Status not-ok
// when an instructions not supported or end of chain reached. Walks one
// instruction at a time, but returns multiple instructions for each conforming
// user.
absl::StatusOr<std::vector<InstructionAndIndex>> WalkDownMemoryOffload(
    const InstructionAndIndex& current_value, const CallGraph& call_graph,
    bool for_move_copy_phase) {
  // TODO(maggioni): Verify that set of instructions supported in chain by
  // legalization is in sync with host_offloader.
  VLOG(6) << "Getting users of: \"" << current_value.instruction->ToString()
          << "\" at index " << current_value.index;
  std::vector<InstructionAndIndex> results;
  auto add_gte_for_idx = [&results](HloInstruction* instr,
                                    int idx) -> absl::Status {
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
      results.emplace_back(user, -1);
    }
    return absl::OkStatus();
  };
  if (current_value.instruction->user_count() == 0) {
    if (current_value.instruction->IsRoot() &&
        !current_value.instruction->parent()->IsEntryComputation()) {
      std::vector<HloInstruction*> callers =
          call_graph.GetComputationCallers(current_value.instruction->parent());
      if (callers.size() != 1 || callers[0]->opcode() != HloOpcode::kWhile) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected computation \"%s\" to be called only by one caller "
            "and that caller to be a While. There are %d caller(s): [%s]",
            current_value.instruction->parent()->name(), callers.size(),
            absl::StrJoin(callers, ", ",
                          [](std::string* out, const HloInstruction* instr) {
                            absl::StrAppend(out, instr->name());
                          })));
      }
      TF_RETURN_IF_ERROR(add_gte_for_idx(callers[0], current_value.index));
      return results;
    }
  }
  if (current_value.instruction->opcode() == HloOpcode::kParameter &&
      current_value.instruction->shape().IsTuple()) {
    TF_RETURN_IF_ERROR(
        add_gte_for_idx(current_value.instruction, current_value.index));
    return results;
  }
  for (HloInstruction* user : current_value.instruction->users()) {
    switch (user->opcode()) {
      case HloOpcode::kGetTupleElement: {
        CHECK_NE(user->tuple_index(), -1);
        if (user->tuple_index() != current_value.index) {
          continue;
        }
        results.emplace_back(user, -1);
        break;
      }
      case HloOpcode::kTuple: {
        auto output_indices = user->OperandIndices(current_value.instruction);
        if (output_indices.size() != 1) {
          return absl::InvalidArgumentError(
              "Expected operand to be used only once in the tuple.");
        }
        results.emplace_back(user, output_indices[0]);
        break;
      }
      case HloOpcode::kOptimizationBarrier: {
        results.emplace_back(user, current_value.index);
        break;
      }
      case HloOpcode::kWhile: {
        HloComputation* while_body = user->while_body();
        HloInstruction* parameter = while_body->parameter_instruction(0);
        results.emplace_back(parameter, current_value.index);
        break;
      }
      case HloOpcode::kDynamicUpdateSlice: {
        if (user->OperandIndices(current_value.instruction)[0] != 0) {
          return absl::InvalidArgumentError(
              "Expected to be used by first operand of dynamic-update-slice");
        }
        results.emplace_back(user, current_value.index);
        break;
      }
      case HloOpcode::kCustomCall: {
        if (user->IsCustomCall(
                memory_annotations::kMoveToDeviceCustomCallTarget)) {
          results.emplace_back(user, current_value.index);
          break;
        }
        return absl::InvalidArgumentError("Invalid custom-call found.");
      }
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kReshape:
      case HloOpcode::kSlice: {
        results.emplace_back(user, current_value.index);
        break;
      }
      case HloOpcode::kAsyncStart: {
        if (user->async_execution_thread() == HloInstruction::kHostThread) {
          // For move copy phase, we need to handle the copy even though we
          // never move the tensor to device yet. For now just throw an error.
          CHECK(!for_move_copy_phase)
              << "Transpose copy going into host call is not supported yet.";

          // For first phase to collect copies to move, it's ok to ignore this
          // path since we don't see copies along the path yet and it's ok to
          // pass host tensor to the async host call.
          break;
        }
        [[fallthrough]];
      }
      default: {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unrecognized user name: %s", user->name()));
      }
    }
  }
  return results;
}

void UpdateInstructionLayout(const InstructionAndIndex& instruction_and_index,
                             const Layout& new_layout) {
  HloInstruction* instruction = instruction_and_index.instruction;
  const int index = instruction_and_index.index;
  VLOG(2) << "  Updating " << instruction->name() << "'s layout "
          << instruction->shape().ToString(true) << " at index " << index
          << " to " << new_layout.ToString();
  // Update shape. Tuple shape vs array shape.
  if (index != -1) {
    *instruction->mutable_shape()
         ->mutable_tuple_shapes(index)
         ->mutable_layout() = new_layout;
  } else {
    VLOG(5) << "  Instruction: " << instruction->ToString();
    VLOG(5) << "  New layout: " << new_layout.ToString();
    *instruction->mutable_shape()->mutable_layout() = new_layout;
  }
  VLOG(3) << "   Shape is now: " << instruction->shape().ToString(true);

  if (instruction->opcode() == HloOpcode::kWhile) {
    // Fix up while body's root instruction shape and condition's
    // parameter shape for while loops.
    *instruction->while_body()
         ->root_instruction()
         ->mutable_shape()
         ->mutable_tuple_shapes(index)
         ->mutable_layout() = new_layout;
    *instruction->while_condition()
         ->parameter_instruction(0)
         ->mutable_shape()
         ->mutable_tuple_shapes(index)
         ->mutable_layout() = new_layout;
  }
}

Shape RemoveMajormostDimension(const Shape& shape) {
  CHECK(shape.has_layout()) << "Shape must have layout.";
  const int size = shape.layout().minor_to_major().size();
  const int64_t majormost_dim = shape.layout().minor_to_major(size - 1);
  return ShapeUtil::DeleteDimension(majormost_dim, shape);
}

Shape AddMajormostDimension(const Shape& shape) {
  CHECK(shape.has_layout()) << "Shape must have layout.";
  Shape new_shape = ShapeUtil::PrependMajorDimension(1, shape);
  for (const Tile& tile : shape.layout().tiles()) {
    *new_shape.mutable_layout()->add_tiles() = tile;
  }
  return new_shape;
}

absl::Status MoveCopyDown(
    const InstructionAndIndex& copy_to_move_instruction_and_index,
    const CallGraph* call_graph,
    absl::flat_hash_set<HloInstruction*>& processed_annotations,
    absl::flat_hash_set<HloInstruction*>& to_remove) {
  HloInstruction* copy_to_move = copy_to_move_instruction_and_index.instruction;
  VLOG(5) << "Moving copy: " << copy_to_move->ToString();
  struct InstructionAndShapes {
    InstructionAndShapes(InstructionAndIndex idx, Shape s_before, Shape s_after)
        : instruction_and_index(idx),
          shape_before_copy(s_before),
          shape_after_copy(s_after) {}
    InstructionAndIndex instruction_and_index;
    Shape shape_before_copy;
    Shape shape_after_copy;
  };
  std::vector<InstructionAndShapes> stack = {InstructionAndShapes(
      copy_to_move_instruction_and_index, copy_to_move->operand(0)->shape(),
      copy_to_move->shape())};
  while (!stack.empty()) {
    InstructionAndShapes current_instruction_and_shapes = stack.back();
    InstructionAndIndex current_instruction_and_index =
        current_instruction_and_shapes.instruction_and_index;
    stack.pop_back();
    VLOG(5) << "Current top of stack: "
            << current_instruction_and_index.instruction->ToString()
            << ", index: " << current_instruction_and_index.index;
    // Get the users of the current instruction.
    absl::StatusOr<std::vector<InstructionAndIndex>> current_value_down =
        WalkDownMemoryOffload(current_instruction_and_index, *call_graph,
                              /*for_move_copy_phase=*/true);
    if (!current_value_down.ok()) {
      VLOG(5) << "WalkDownMemoryOffload failed: "
              << current_value_down.status();
      break;
    }

    for (InstructionAndIndex& instruction_and_index :
         current_value_down.value()) {
      HloInstruction* instruction = instruction_and_index.instruction;
      Shape shape_before_copy =
          current_instruction_and_shapes.shape_before_copy;
      Shape shape_after_copy = current_instruction_and_shapes.shape_after_copy;
      VLOG(5) << "Evaluating successor: " << instruction->ToString();
      const int index = instruction_and_index.index;
      if (instruction->opcode() == HloOpcode::kBitcast) {
        // For now, we only know how to move a copy over a bitcast which
        // "reshapes" away the majormost dimension (which must be a degenerate
        // dimension), or reshapes to add a degenerate majormost dimension.
        const Shape& before_bitcast_shape = instruction->operand(0)->shape();
        const Shape& after_bitcast_shape = instruction->shape();
        if (!Shape::Equal().IgnoreLayout()(copy_to_move->operand(0)->shape(),
                                           copy_to_move->shape())) {
          return absl::InternalError(absl::StrFormat(
              "Expecting copy to only change instructions layout. Copy: %s",
              copy_to_move->ToString()));
        }
        if (after_bitcast_shape.dimensions().size() ==
            before_bitcast_shape.dimensions().size() - 1) {
          if (!(ShapeUtil::IsEffectivelyMostMajorDimension(before_bitcast_shape,
                                                           0) &&
                before_bitcast_shape.dimensions(0) == 1)) {
            return absl::InternalError(absl::StrFormat(
                "Only handling bitcasts with majormost dimension "
                "of size 1. This bitcast is \"%s\"",
                instruction->ToString()));
          }
          const Shape new_bitcast_shape =
              RemoveMajormostDimension(shape_before_copy);
          VLOG(2) << absl::StreamFormat(
              " Encountered bitcast \"%s\", updating current shape from %s to "
              "%s",
              instruction->name(), shape_before_copy.ToString(true),
              new_bitcast_shape.ToString(true));
          shape_before_copy = new_bitcast_shape;
          const Shape new_copy_shape =
              RemoveMajormostDimension(shape_after_copy);
          VLOG(2) << absl::StreamFormat(
              " Also updating shape after copy from %s to %s",
              shape_after_copy.ToString(true), new_copy_shape.ToString(true));
          shape_after_copy = new_copy_shape;
        } else if (after_bitcast_shape.dimensions().size() ==
                   before_bitcast_shape.dimensions().size() + 1) {
          if (!(ShapeUtil::IsEffectivelyMostMajorDimension(after_bitcast_shape,
                                                           0) &&
                after_bitcast_shape.dimensions(0) == 1)) {
            return absl::InternalError(absl::StrFormat(
                "Only handling bitcasts with majormost dimension "
                "of size 1. This bitcast is \"%s\"",
                instruction->ToString()));
          }
          const Shape new_bitcast_shape =
              AddMajormostDimension(shape_before_copy);
          VLOG(2) << absl::StreamFormat(
              " Encountered bitcast \"%s\", updating current shape from %s to "
              "%s",
              instruction->name(), shape_before_copy.ToString(true),
              new_bitcast_shape.ToString(true));
          shape_before_copy = new_bitcast_shape;
          const Shape new_copy_shape = AddMajormostDimension(shape_after_copy);
          VLOG(2) << absl::StreamFormat(
              " Also updating shape after copy from %s to %s",
              shape_after_copy.ToString(true), new_copy_shape.ToString(true));
          shape_after_copy = new_copy_shape;
        } else {
          return absl::InternalError(
              absl::StrFormat("Only handling bitcasts which add or remove a "
                              "0'th dimension. This bitcast is \"%s\"",
                              instruction->ToString()));
        }
      } else if (instruction->opcode() == HloOpcode::kSlice ||
                 instruction->opcode() == HloOpcode::kDynamicSlice) {
        // Since we're moving the copy over a Slice/DynamicSlice, we need to
        // change the shape of the copy to match the shape of the result of the
        // Slice/DynamicSlice. We want to maintain the layout of
        // shape_after_copy though.
        Shape new_copy_shape = instruction->shape();
        *new_copy_shape.mutable_layout() = shape_after_copy.layout();
        VLOG(2) << absl::StreamFormat(
            " Encountered %s \"%s\", updating shape after copy from "
            "%s to %s",
            HloOpcodeString(instruction->opcode()), instruction->name(),
            shape_after_copy.ToString(true), new_copy_shape.ToString(true));
        shape_after_copy = new_copy_shape;
      }

      // Update the shape of this instruction as if the copy never happened.
      UpdateInstructionLayout(instruction_and_index,
                              shape_before_copy.layout());
      if (instruction->opcode() == HloOpcode::kParameter) {
        // Also update the layout of the call site.
        std::vector<HloInstruction*> callers =
            call_graph->GetComputationCallers(instruction->parent());
        if (callers.size() != 1) {
          return absl::InvalidArgumentError(
              "Expected to be called only by one caller");
        }
        HloInstruction* caller = callers[0];
        UpdateInstructionLayout(InstructionAndIndex(caller, index),
                                shape_before_copy.layout());
      }

      CHECK_NE(instruction->opcode(), HloOpcode::kCopy)
          << "Copies should be processed in reverse order so this never "
             "happens";
      if (absl::c_linear_search(kUsersOpcodes, instruction->opcode()) ||
          instruction->IsCustomCall(
              memory_annotations::kMoveToDeviceCustomCallTarget)) {
        HloInstruction* annotation =
            FindToDeviceAnnotationToUpdate(instruction);
        CHECK_NE(annotation, nullptr)
            << "We already verified we could find an annotation here. "
               "Something went wrong.";
        HloInstruction* new_annotation = nullptr;
        if (instruction->opcode() == HloOpcode::kCustomCall) {
          new_annotation = annotation;
        } else {
          new_annotation =
              instruction->AddInstruction(annotation->CloneWithNewOperands(
                  instruction->shape(), {instruction}));
        }
        UpdateInstructionLayout(InstructionAndIndex(new_annotation, -1),
                                shape_before_copy.layout());
        VLOG(3) << absl::StreamFormat("Creating copy with shape %s",
                                      shape_after_copy.ToString(true));
        HloInstruction* new_copy =
            instruction->AddInstruction(copy_to_move->CloneWithNewOperands(
                shape_after_copy, {new_annotation}));
        VLOG(2) << absl::StreamFormat("Inserting copy \"%s\" after \"%s\"",
                                      new_copy->name(), instruction->name());
        std::vector<HloInstruction*> users = instruction->users();
        for (HloInstruction* use : users) {
          if (use == new_copy || use == new_annotation) {
            continue;
          }
          TF_RETURN_IF_ERROR(
              instruction->ReplaceUseWithDifferentShape(use, new_copy));
        }
        // Move the copy here.
        if (new_annotation != annotation) {
          TF_RETURN_IF_ERROR(annotation->ReplaceAllUsesWithDifferentShape(
              annotation->mutable_operand(0)));
          to_remove.insert(annotation);
        }
        continue;
      }
      // Move the annotation first just before dynamic-update-slice to avoid
      // shape changes.
      if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
        HloInstruction* annotation =
            FindToHostAnnotationToUpdate(instruction->mutable_operand(1));
        if (annotation == nullptr) {
          return absl::InternalError("Annotation not found.");
        }
        CHECK(annotation->opcode() == HloOpcode::kCustomCall);
        HloInstruction* new_annotation =
            instruction->AddInstruction(annotation->CloneWithNewOperands(
                instruction->operand(1)->shape(),
                {instruction->mutable_operand(1)}));
        TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(1, new_annotation));
        TF_RETURN_IF_ERROR(
            annotation->ReplaceAllUsesWith(annotation->mutable_operand(0)));
        processed_annotations.insert(annotation);
        processed_annotations.insert(new_annotation);
        to_remove.insert(annotation);

        // Need to make DUS and its update slice's layout consistent by adding
        // a copy on the operand side, which is on device.
        if (instruction->shape().layout().minor_to_major() !=
            instruction->operand(1)->shape().layout().minor_to_major()) {
          HloInstruction* update_slice = instruction->mutable_operand(1);
          CHECK(update_slice->IsCustomCall(
              memory_annotations::kMoveToHostCustomCallTarget));
          *update_slice->mutable_shape()->mutable_layout() =
              instruction->shape().layout();
          HloInstruction* new_copy =
              update_slice->AddInstruction(HloInstruction::CreateUnary(
                  update_slice->shape(), HloOpcode::kCopy,
                  update_slice->mutable_operand(0)));
          TF_RETURN_IF_ERROR(update_slice->ReplaceOperandWith(0, new_copy));
        }
      }
      stack.emplace_back(instruction_and_index, shape_before_copy,
                         shape_after_copy);
    }
  }
  VLOG(2) << absl::StreamFormat("Removing copy \"%s\"",
                                copy_to_move->ToString());
  TF_RETURN_IF_ERROR(copy_to_move->ReplaceAllUsesWithDifferentShape(
      copy_to_move->mutable_operand(0)));
  TF_RETURN_IF_ERROR(copy_to_move->parent()->RemoveInstruction(copy_to_move));
  return absl::OkStatus();
}

// Returns true if the copy should be moved. A copy can be moved if there is
// always a place for it after being moved back to device.
bool ShouldMoveCopyDown(InstructionAndIndex copy_to_move) {
  std::queue<host_offload_utils::InstructionAndShapeIndex> queue;
  queue.push({copy_to_move.instruction, {}});
  while (!queue.empty()) {
    host_offload_utils::InstructionAndShapeIndex current = queue.front();
    queue.pop();
    if (current.instruction->IsRoot() &&
        current.instruction->parent()->IsEntryComputation()) {
      // It reaches entry computation root without being brought back to the
      // device. Do not move this copy since there is no place to do this copy
      // on device.
      return false;
    }

    // Push successors onto the queue to be visited.
    absl::StatusOr<std::vector<host_offload_utils::InstructionAndShapeIndex>>
        successors = host_offload_utils::GetSuccessors(current);
    if (!successors.ok()) {
      return false;
    }
    for (const host_offload_utils::InstructionAndShapeIndex& successor :
         successors.value()) {
      if (successor.instruction->IsCustomCall(
              memory_annotations::kMoveToDeviceCustomCallTarget)) {
        continue;
      }
      queue.push(successor);
    }
  }
  return true;
}

absl::StatusOr<bool> ProcessAnnotationForCopyMovement(
    HloInstruction* instruction, const CallGraph* call_graph,
    absl::flat_hash_set<HloInstruction*>& processed_annotations,
    absl::flat_hash_set<HloInstruction*>& to_remove) {
  VLOG(2) << "Walking down graph starting at instruction "
          << instruction->name();
  if (instruction->IsRoot()) {
    return false;
  }
  if (instruction->user_count() == 0) {
    return false;
  }
  // Look for a DynamicUpdateSlice following this instruction.
  HloInstruction* starting_instr =
      FindDUSFromAnnotation(instruction->users().at(0));
  // If it's the pure copy case reset instruction.
  if (starting_instr->opcode() != HloOpcode::kDynamicUpdateSlice) {
    starting_instr = instruction;
  }
  if (!(starting_instr->IsCustomCall(
            memory_annotations::kMoveToHostCustomCallTarget) ||
        IsEntryComputationParameter(starting_instr) ||
        starting_instr->opcode() == HloOpcode::kDynamicUpdateSlice)) {
    return absl::InternalError(
        "Starting instruction must be a move-to-host annotation, entry "
        "computation parameter, or dynamic-update-slice.");
  }
  VLOG(2) << "Effective starting instruction: " << starting_instr->name();

  InstructionAndIndex current_value(starting_instr, -1);
  // Walk up to find all annotations to update (required in case there are
  // multiple insertions in the buffer).
  processed_annotations.insert(current_value.instruction);

  if (current_value.instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
    // Walk up the graph and find the broadcast which this dynamic-update-slice
    // is updating.
    while (true) {
      VLOG(10) << "Current value before: "
               << current_value.instruction->ToString();
      absl::StatusOr<InstructionAndIndex> current_value_up =
          WalkUpMemoryOffload(current_value, *call_graph);
      // Invalid upward walking means the chain is unrecognized.
      if (!current_value_up.ok()) {
        return false;
      }
      // This means we encountered a broadcast with constant 0 expansion.
      if (current_value_up.value() == current_value) {
        break;
      }
      current_value = current_value_up.value();
      VLOG(10) << "Current value after: "
               << current_value.instruction->ToString();
      HloInstruction* annotation = current_value.instruction;
      if (annotation->opcode() == HloOpcode::kDynamicUpdateSlice) {
        HloInstruction* real_annotation =
            FindToHostAnnotationToUpdate(annotation->mutable_operand(1));
        // Check if this dynamic-update-slice doesn't have an annotation
        // attached.
        if (!real_annotation->IsCustomCall(
                memory_annotations::kMoveToHostCustomCallTarget)) {
          return false;
        }
      }
    }
  }

  // Do a final walkdown from the top to find all the copies which we need to
  // move.
  std::vector<InstructionAndIndex> copies_to_move;
  std::vector<InstructionAndIndex> stack = {current_value};
  while (!stack.empty()) {
    VLOG(5) << "Current value before down: "
            << stack.back().instruction->ToString() << " "
            << stack.back().index;
    if (absl::c_linear_search(kUsersOpcodes,
                              stack.back().instruction->opcode()) ||
        stack.back().instruction->IsCustomCall(
            memory_annotations::kMoveToDeviceCustomCallTarget)) {
      HloInstruction* annotation =
          FindToDeviceAnnotationToUpdate(stack.back().instruction);
      if (!annotation ||
          !annotation->IsCustomCall(
              memory_annotations::kMoveToDeviceCustomCallTarget)) {
        VLOG(5) << "Couldn't find annotation for consumer instruction in chain";
        return false;
      }

      // Fix up while body's root instruction shape along the way.
      if (annotation->IsCustomCall(
              memory_annotations::kMoveToDeviceCustomCallTarget)) {
        for (HloInstruction* user : annotation->users()) {
          HloInstruction* root_instruction =
              annotation->parent()->root_instruction();
          if (root_instruction == user &&
              root_instruction->opcode() == HloOpcode::kTuple &&
              !root_instruction->parent()->IsEntryComputation()) {
            std::vector<HloInstruction*> callers =
                call_graph->GetComputationCallers(annotation->parent());
            if (callers.size() != 1 ||
                callers[0]->opcode() != HloOpcode::kWhile) {
              return absl::InvalidArgumentError(absl::StrFormat(
                  "Expected computation \"%s\" to be called only by one caller "
                  "and that caller to be a While. There are %d caller(s): [%s]",
                  current_value.instruction->parent()->name(), callers.size(),
                  absl::StrJoin(
                      callers, ", ",
                      [](std::string* out, const HloInstruction* instr) {
                        absl::StrAppend(out, instr->name());
                      })));
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
    absl::StatusOr<std::vector<InstructionAndIndex>> current_value_down =
        WalkDownMemoryOffload(stack.back(), *call_graph,
                              /*for_move_copy_phase=*/false);
    if (!current_value_down.ok()) {
      VLOG(5) << "Current value down failed: " << current_value_down.status();
      break;
    }
    stack.pop_back();
    stack.insert(stack.end(), current_value_down.value().begin(),
                 current_value_down.value().end());
    for (InstructionAndIndex& instruction_and_index :
         current_value_down.value()) {
      VLOG(5) << "Current value last down: "
              << stack.back().instruction->ToString();
      if (instruction_and_index.instruction->opcode() == HloOpcode::kCopy) {
        VLOG(1) << absl::StreamFormat(
            " Found a copy to move: \"%s\"",
            instruction_and_index.instruction->name());
        copies_to_move.push_back(instruction_and_index);
      }
    }
  }

  if (copies_to_move.empty()) {
    return false;
  }

  bool changed = false;
  // Process all copies one at a time from the last to the first and push it to
  // its specific user.
  for (auto it = copies_to_move.rbegin(); it != copies_to_move.rend(); ++it) {
    InstructionAndIndex& copy_to_move_and_index = *it;
    HloInstruction* copy_to_move = copy_to_move_and_index.instruction;
    if (ShouldMoveCopyDown(copy_to_move_and_index)) {
      TF_RETURN_IF_ERROR(MoveCopyDown(copy_to_move_and_index, call_graph,
                                      processed_annotations, to_remove));
      changed = true;
    } else {
      // We should not move this copy down; maybe we can move it up. For now, we
      // only check if we can easily move it up by swapping places with its
      // operand.
      if (copy_to_move->operand(0)->IsCustomCall(
              memory_annotations::kMoveToHostCustomCallTarget)) {
        HloInstruction* custom_call = copy_to_move->mutable_operand(0);
        TF_RETURN_IF_ERROR(copy_to_move->ReplaceAllUsesWith(custom_call));
        TF_RETURN_IF_ERROR(copy_to_move->ReplaceOperandWith(
            0, custom_call->mutable_operand(0)));
        TF_RETURN_IF_ERROR(custom_call->ReplaceOperandWith(0, copy_to_move));
        copy_to_move->mutable_shape()->mutable_layout()->set_memory_space(
            Layout::kDefaultMemorySpace);
        *custom_call->mutable_shape()->mutable_layout() =
            copy_to_move->shape().layout();
        changed = true;
      }
    }
  }
  return changed;
}

// Fixes layout changing copies in between on the path to users.
absl::StatusOr<bool> FixupInterveningCopies(
    const std::vector<HloInstruction*>& starting_instructions,
    const CallGraph* call_graph) {
  absl::flat_hash_set<HloInstruction*> processed_annotations;
  absl::flat_hash_set<HloInstruction*> annotations_to_remove;
  bool changed = false;
  for (HloInstruction* instruction : starting_instructions) {
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

std::vector<HloInstruction*>
HostOffloadLegalize::FindStartingInstructionsOfHostMemoryOffload(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  std::vector<HloInstruction*> starting_instructions;
  // Iterate over all instructions and look for XLA host offload annotations.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsEntryComputationParameter(instruction)) {
        Shape param_shape =
            module->entry_computation_layout()
                .parameter_layout(instruction->parameter_number())
                .shape();
        // TODO(mingyao): Add support for tuple parameter.
        if (param_shape.has_layout() &&
            param_shape.layout().memory_space() == Layout::kHostMemorySpace) {
          starting_instructions.push_back(instruction);
          continue;
        }
      }

      if (instruction->IsCustomCall(
              memory_annotations::kMoveToHostCustomCallTarget)) {
        starting_instructions.push_back(instruction);
      }
    }
  }
  return starting_instructions;
}

absl::StatusOr<bool> HostOffloadLegalize::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  // Look for layout changing copies which happen during host memory offload. If
  // any are found, move them outside of the offload section.
  std::vector<HloInstruction*> starting_instructions =
      FindStartingInstructionsOfHostMemoryOffload(module, execution_threads);
  VLOG(1) << absl::StreamFormat(
      "Starting instructions for host memory offload: [%s]",
      absl::StrJoin(starting_instructions, ", ",
                    [](std::string* out, HloInstruction* instruction) {
                      return absl::StrAppend(out, instruction->name());
                    }));
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_ASSIGN_OR_RETURN(
      bool changed_intervening_copies,
      FixupInterveningCopies(starting_instructions, call_graph.get()));
  changed |= changed_intervening_copies;

  return changed;
}

}  // namespace xla

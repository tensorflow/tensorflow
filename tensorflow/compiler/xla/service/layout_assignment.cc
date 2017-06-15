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

#include "tensorflow/compiler/xla/service/layout_assignment.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <set>
#include <string>
#include <tuple>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

std::ostream& operator<<(std::ostream& out,
                         const LayoutConstraint& constraint) {
  out << constraint.ToString();
  return out;
}

BufferLayoutConstraint::BufferLayoutConstraint(const Layout& layout,
                                               const LogicalBuffer& buffer,
                                               bool mandatory)
    : LayoutConstraint(mandatory), layout_(layout), buffer_(&buffer) {
  CHECK(LayoutUtil::ValidateLayoutForShape(layout, buffer.shape()).ok());
}

string BufferLayoutConstraint::ToString() const {
  return tensorflow::strings::Printf("BufferLayoutConstraint %s: %s",
                                     buffer_->ToString().c_str(),
                                     LayoutUtil::HumanString(layout_).c_str());
}

OperandLayoutConstraint::OperandLayoutConstraint(
    const ShapeLayout& shape_layout, const HloInstruction* instruction,
    int64 operand_no, bool mandatory)
    : LayoutConstraint(mandatory),
      shape_layout_(shape_layout),
      instruction_(instruction),
      operand_no_(operand_no) {
  CHECK(shape_layout_.LayoutIsSet());
  CHECK(ShapeUtil::Compatible(shape_layout.shape(),
                              instruction->operand(operand_no)->shape()));
}

string OperandLayoutConstraint::ToString() const {
  return tensorflow::strings::Printf(
      "OperandLayoutConstraint %s, operand %lld: %s",
      instruction_->name().c_str(), operand_no_,
      shape_layout_.ToString().c_str());
}

string ResultLayoutConstraint::ToString() const {
  return tensorflow::strings::Printf("ResultLayoutConstraint: %s",
                                     shape_layout_.ToString().c_str());
}

LayoutConstraints::LayoutConstraints(
    const TuplePointsToAnalysis& points_to_analysis,
    const HloComputation* computation)
    : points_to_analysis_(points_to_analysis), computation_(computation) {
  // Gather all array-shaped logical buffers into unconstrained_buffer_ids.
  for (auto& buffer : points_to_analysis_.logical_buffers()) {
    // The points to analysis is computed per module, restrict constraints to
    // array buffers in this computation.
    if (buffer->IsArray() && buffer->instruction()->parent() == computation) {
      unconstrained_buffer_ids_.insert(buffer->id());
    }
  }
}

bool LayoutConstraints::OperandBufferForwarded(
    const HloInstruction* instruction, int64 operand_no) const {
  // The operand is potentially forwarded if the intersection of points-to sets
  // of the operand and the instruction is non-empty.
  auto output_buffers =
      points_to_analysis_.GetPointsToSet(instruction).CreateFlattenedSet();
  auto operand_buffers =
      points_to_analysis_.GetPointsToSet(instruction->operand(operand_no))
          .CreateFlattenedSet();
  for (const LogicalBuffer* output_buffer : output_buffers) {
    if (operand_buffers.count(output_buffer) > 0) {
      return true;
    }
  }
  return false;
}

Status LayoutConstraints::SetBufferLayout(const Layout& layout,
                                          const LogicalBuffer& buffer,
                                          bool mandatory) {
  VLOG(3) << "SetBufferLayout : " << buffer << " : "
          << LayoutUtil::HumanString(layout);

  TF_RETURN_IF_ERROR(points_to_analysis_.VerifyBuffer(buffer));
  if (!buffer.IsArray()) {
    return FailedPrecondition(
        "Layout of buffer %s cannot be constrained because buffer is not "
        "array-shaped, has shape: %s",
        buffer.ToString().c_str(),
        ShapeUtil::HumanString(buffer.shape()).c_str());
  }
  TF_RETURN_IF_ERROR(
      LayoutUtil::ValidateLayoutForShape(layout, buffer.shape()));

  const BufferLayoutConstraint* curr_constraint =
      GetBufferLayoutConstraint(buffer);
  if (curr_constraint != nullptr) {
    if (LayoutUtil::Equal(curr_constraint->layout(), layout)) {
      // New constraint matches existing constraint. Nothing to do.
      return Status::OK();
    }
    if (curr_constraint->mandatory()) {
      return FailedPrecondition(
          "Buffer %s already has the layout constraint %s, cannot add "
          "incompatible constraint %s",
          buffer.ToString().c_str(),
          LayoutUtil::HumanString(curr_constraint->layout()).c_str(),
          LayoutUtil::HumanString(layout).c_str());
    }
  }

  auto iter = buffer_constraints_.find(&buffer);
  bool overwrite = iter != buffer_constraints_.end();
  if (!overwrite) {
    iter = buffer_constraints_
               .insert(std::make_pair(
                   &buffer, BufferLayoutConstraint(layout, buffer, mandatory)))
               .first;
  } else {
    iter->second = BufferLayoutConstraint(layout, buffer, /*mandatory=*/true);
  }
  added_constraints_.push_back(&iter->second);

  // Remove buffer from the set of unconstrained buffers.
  TF_RET_CHECK(unconstrained_buffer_ids_.count(buffer.id()) ==
               static_cast<int>(!overwrite));
  unconstrained_buffer_ids_.erase(buffer.id());

  return Status::OK();
}

Status LayoutConstraints::SetOperandLayout(const Shape& shape_with_layout,
                                           const HloInstruction* instruction,
                                           int64 operand_no, bool mandatory) {
  VLOG(3) << "SetOperandLayout : " << instruction->name() << ", operand "
          << operand_no << " : "
          << ShapeUtil::HumanStringWithLayout(shape_with_layout);

  const OperandLayoutConstraint* curr_shape_layout =
      GetOperandLayoutConstraint(instruction, operand_no);
  if (curr_shape_layout != nullptr) {
    if (curr_shape_layout->shape_layout().MatchesLayoutInShape(
            shape_with_layout)) {
      // New constraint matches existing constraint. Nothing to do.
      return Status::OK();
    }
    if (curr_shape_layout->mandatory()) {
      return FailedPrecondition(
          "Operand %lld of instruction %s already has a layout constraint "
          "%s, cannot add incompatible constraint %s",
          operand_no, instruction->name().c_str(),
          curr_shape_layout->shape_layout().ToString().c_str(),
          ShapeUtil::HumanStringWithLayout(shape_with_layout).c_str());
    }
  }

  // If any buffers in the operand occur in the output of the instruction, then
  // return an error. This case is not handled because such a constraint changes
  // layouts beyond this immediate use and is complicated to handle.
  if (OperandBufferForwarded(instruction, operand_no)) {
    return FailedPrecondition(
        "Cannot constraint layout of operand %lld of instruction %s "
        "because instruction forwards operand's LogicalBuffer(s)",
        operand_no, instruction->name().c_str());
  }

  auto key = std::make_pair(instruction, operand_no);
  auto iter = operand_constraints_.find(key);
  if (iter == operand_constraints_.end()) {
    auto pair = std::make_pair(
        key, OperandLayoutConstraint(ShapeLayout(shape_with_layout),
                                     instruction, operand_no, mandatory));
    iter = operand_constraints_.insert(pair).first;
  } else {
    iter->second =
        OperandLayoutConstraint(ShapeLayout(shape_with_layout), instruction,
                                operand_no, /*mandatory=*/true);
  }
  added_constraints_.push_back(&iter->second);

  return Status::OK();
}

Status LayoutConstraints::SetArrayOperandLayout(
    const Layout& layout, const HloInstruction* instruction, int64 operand_no,
    bool mandatory) {
  const HloInstruction* operand = instruction->operand(operand_no);
  TF_RET_CHECK(ShapeUtil::IsArray(operand->shape()));
  Shape shape(operand->shape());
  *shape.mutable_layout() = layout;
  TF_RETURN_IF_ERROR(LayoutUtil::ValidateLayoutInShape(shape));
  return SetOperandLayout(shape, instruction, operand_no, mandatory);
}

Status LayoutConstraints::SetResultLayout(const Shape& shape_with_layout) {
  VLOG(3) << "SetResultLayout : "
          << ShapeUtil::HumanStringWithLayout(shape_with_layout);

  const ShapeLayout* curr_shape_layout = ResultLayout();
  if (curr_shape_layout != nullptr) {
    if (!curr_shape_layout->MatchesLayoutInShape(shape_with_layout)) {
      return FailedPrecondition(
          "Result of computation %s already has the layout constraint %s, "
          "cannot add incompatible constraint %s",
          computation_->name().c_str(), curr_shape_layout->ToString().c_str(),
          ShapeUtil::HumanStringWithLayout(shape_with_layout).c_str());
    }
    // New constraint matches existing constraint. Nothing to do.
    return Status::OK();
  }

  result_constraint_.reset(
      new ResultLayoutConstraint(ShapeLayout(shape_with_layout)));
  added_constraints_.push_back(result_constraint_.get());

  return Status::OK();
}

Status LayoutConstraints::SetInstructionLayout(
    const Shape& shape_with_layout, const HloInstruction* instruction) {
  VLOG(3) << "SetInstructionLayout : " << instruction->name() << ", "
          << ShapeUtil::HumanStringWithLayout(shape_with_layout);

  if (!ShapeUtil::Compatible(shape_with_layout, instruction->shape())) {
    return FailedPrecondition(
        "Instruction %s of shape %s cannot be assigned incompatible layout %s",
        instruction->name().c_str(),
        ShapeUtil::HumanString(instruction->shape()).c_str(),
        ShapeUtil::HumanStringWithLayout(shape_with_layout).c_str());
  }

  // Create a BufferLayoutConstraint for each array shape in the output of the
  // instruction.
  return ShapeUtil::ForEachSubshapeWithStatus(
      shape_with_layout,
      [this, instruction](const Shape& subshape,
                          const ShapeIndex& index) -> Status {
        // The precondition for this method is that the instruction defines all
        // buffers in its output.
        auto buffers =
            points_to_analysis_.GetPointsToSet(instruction).element(index);
        CHECK_EQ(1, buffers.size());
        CHECK_EQ(buffers[0]->instruction(), instruction);

        if (ShapeUtil::IsArray(subshape)) {
          return SetBufferLayout(subshape.layout(), *buffers[0]);
        } else {
          return Status::OK();
        }
      });
}

const Layout* LayoutConstraints::BufferLayout(
    const LogicalBuffer& buffer) const {
  if (const auto* constraint = GetBufferLayoutConstraint(buffer)) {
    return &constraint->layout();
  }
  return nullptr;
}
const BufferLayoutConstraint* LayoutConstraints::GetBufferLayoutConstraint(
    const LogicalBuffer& buffer) const {
  auto it = buffer_constraints_.find(&buffer);
  return it == buffer_constraints_.end() ? nullptr : &it->second;
}

const ShapeLayout* LayoutConstraints::OperandLayout(
    const HloInstruction* instruction, int64 operand_no) const {
  if (const auto* constraint =
          GetOperandLayoutConstraint(instruction, operand_no)) {
    return &constraint->shape_layout();
  }
  return nullptr;
}
const OperandLayoutConstraint* LayoutConstraints::GetOperandLayoutConstraint(
    const HloInstruction* instruction, int64 operand_no) const {
  auto it = operand_constraints_.find(std::make_pair(instruction, operand_no));
  return it == operand_constraints_.end() ? nullptr : &it->second;
}

const ShapeLayout* LayoutConstraints::ResultLayout() const {
  return result_constraint_ ? &result_constraint_->shape_layout() : nullptr;
}

string LayoutConstraints::ToString() const {
  string output;
  tensorflow::strings::StrAppend(&output, "LayoutConstraints for computation ",
                                 computation_->name(), ":\n");
  for (auto* instruction : computation_->MakeInstructionPostOrder()) {
    tensorflow::strings::StrAppend(&output, "  ", instruction->ToShortString(),
                                   "\n");
    for (int64 i = 0; i < instruction->operand_count(); ++i) {
      if (OperandLayout(instruction, i) != nullptr) {
        tensorflow::strings::StrAppend(
            &output, "    operand (", i,
            "): ", OperandLayout(instruction, i)->ToString(), "\n");
      }
    }
    for (const LogicalBuffer* buffer :
         points_to_analysis_.GetBuffersDefinedByInstruction(instruction)) {
      if (BufferLayout(*buffer) != nullptr) {
        tensorflow::strings::StrAppend(
            &output, "    ", buffer->ToString(), " : ",
            LayoutUtil::HumanString(*BufferLayout(*buffer)), "\n");
      }
    }
  }

  if (ResultLayout() != nullptr) {
    tensorflow::strings::StrAppend(&output, "  => ", ResultLayout()->ToString(),
                                   "\n");
  }
  return output;
}

Status LayoutAssignment::AddMandatoryConstraints(
    const ComputationLayout& computation_layout, HloComputation* computation,
    LayoutConstraints* constraints) {
  VLOG(3) << "Adding mandatory layout constraints to computation "
          << computation->name();

  // Constrain layouts of instructions which define values with pre-existing
  // layouts.
  for (auto& instruction : computation->instructions()) {
    Shape const* shape_with_layout = nullptr;
    if (instruction->opcode() == HloOpcode::kConstant) {
      // Constant layouts must match the layout of their literal.
      shape_with_layout = &instruction->literal().shape();
    } else if (instruction->opcode() == HloOpcode::kInfeed) {
      // Infeed layouts must match the layout of the original inserted
      // instruction.
      // TODO(b/31425034): Change infeeds to be more like parameters, with
      // shapes in the ComputationLayout.
      shape_with_layout = &instruction->shape();
    } else if (instruction->opcode() == HloOpcode::kOutfeed) {
      // Constrain the input to the Outfeed instruction to be the expected
      // layout of the Outfeed.
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
          instruction->outfeed_shape(), instruction.get(), 0,
          /*mandatory=*/true));
    } else if (instruction->opcode() == HloOpcode::kParameter) {
      // Parameter layouts must match the respective layout in
      // ComputationLayout.
      shape_with_layout =
          &computation_layout.parameter_layout(instruction->parameter_number())
               .shape();
    }
    if (shape_with_layout != nullptr) {
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(*shape_with_layout,
                                                           instruction.get()));
    }
  }

  // Constrain layouts of instructions which call computations which have
  // already been assigned layouts. Instructions which call computations in a
  // parallel element-wise context (eg, map or reduce) do not need layout
  // constraints because they operate on scalars.
  for (auto& instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      // kCall instruction operands and output must match the ComputationLayout
      // of the called computation.
      const ComputationLayout& called_computation_layout =
          FindOrDie(computation_layouts_, instruction->to_apply());
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(
          called_computation_layout.result_layout().shape(),
          instruction.get()));
      TF_RET_CHECK(instruction->operand_count() ==
                   called_computation_layout.parameter_count());
      for (int64 i = 0; i < instruction->operand_count(); ++i) {
        TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
            called_computation_layout.parameter_layout(i).shape(),
            instruction.get(), i, /*mandatory=*/true));
      }
    } else if (instruction->opcode() == HloOpcode::kWhile) {
      // Layout of input and output of kWhile instruction must be equal and must
      // match both input and output of body computation. Also, the input of
      // condition computation must match kWhile layout.
      HloComputation* body = instruction->while_body();
      HloComputation* condition = instruction->while_condition();
      const HloInstruction* init = instruction->operand(0);
      const ComputationLayout& body_layout =
          FindOrDie(computation_layouts_, body);
      const ComputationLayout& condition_layout =
          FindOrDie(computation_layouts_, condition);

      // Check a few invariants irrespective of layout.
      CHECK_EQ(1, instruction->operand_count());
      CHECK_EQ(1, body->num_parameters());
      CHECK_EQ(1, condition->num_parameters());
      DCHECK(ShapeUtil::Compatible(body_layout.result_shape(),
                                   body_layout.parameter_shape(0)));
      DCHECK(ShapeUtil::Compatible(body_layout.result_shape(),
                                   condition_layout.parameter_shape(0)));
      DCHECK(ShapeUtil::Compatible(body_layout.result_shape(), init->shape()));

      // Return error if earlier layout assignment of the embedded computations
      // has produced conflicting layouts.
      if (!ShapeUtil::Equal(body_layout.result_shape(),
                            body_layout.parameter_shape(0))) {
        return InternalError(
            "Parameter and result of body computation %s of while instruction "
            "%s have different layouts: %s vs %s",
            body->name().c_str(), instruction->name().c_str(),
            ShapeUtil::HumanString(body_layout.result_shape()).c_str(),
            ShapeUtil::HumanString(body_layout.parameter_shape(0)).c_str());
      }
      if (!ShapeUtil::Equal(body->root_instruction()->shape(),
                            condition->parameter_instruction(0)->shape())) {
        return InternalError(
            "Parameter of condition computation %s of while instruction "
            "%s does not match body computation %s result: %s vs %s",
            condition->name().c_str(), instruction->name().c_str(),
            body->name().c_str(),
            ShapeUtil::HumanString(condition_layout.parameter_shape(0)).c_str(),
            ShapeUtil::HumanString(body_layout.result_shape()).c_str());
      }

      // Constrain the output and the operand of the while instruction to match
      // the computations.
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(
          body_layout.result_shape(), instruction.get()));
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
          body_layout.result_shape(), instruction.get(), 0,
          /*mandatory=*/true));
    } else if (instruction->opcode() == HloOpcode::kCustomCall) {
      // Add constraints for kCustomCall instruction operands and instructions.
      // For now we only support row major layouts for all inputs and outputs.
      auto row_major_shape = [](const Shape& old_shape) {
        Shape new_shape(old_shape);
        std::vector<int64> dimension_order(new_shape.dimensions_size());
        std::iota(dimension_order.rbegin(), dimension_order.rend(), 0);
        *new_shape.mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
        return new_shape;
      };

      Shape result_shape(row_major_shape(instruction->shape()));
      TF_RETURN_IF_ERROR(
          constraints->SetInstructionLayout(result_shape, instruction.get()));
      for (int64 i = 0; i < instruction->operand_count(); ++i) {
        const Shape& operand_shape = instruction->operand(i)->shape();
        // Opaque operands don't get a layout constraint.
        if (ShapeUtil::IsOpaque(operand_shape)) {
          continue;
        }

        Shape row_major_operand_shape(row_major_shape(operand_shape));
        TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
            row_major_operand_shape, instruction.get(), i, /*mandatory=*/true));
      }
    }
  }

  // Finally set the result layout to match ComputationLayout.
  return constraints->SetResultLayout(
      computation_layout.result_layout().shape());
}

namespace {

// The operands of a call must match the layouts of parameters in the
// ComputationLayout, and the call instruction itself must match the result
// layout in the ComputationLayout.
Status CheckCallLayout(HloInstruction* call,
                       const ComputationLayout& computation_layout) {
  HloComputation* computation = call->to_apply();
  TF_RET_CHECK(computation->num_parameters() == call->operand_count());
  for (int64 i = 0; i < computation->num_parameters(); ++i) {
    TF_RET_CHECK(computation_layout.parameter_layout(i).MatchesLayoutInShape(
        call->operand(i)->shape()));
  }
  TF_RET_CHECK(
      computation_layout.result_layout().MatchesLayoutInShape(call->shape()));
  return Status::OK();
}

// Custom calls have fixed input and output layouts.
Status CheckCustomCallLayout(HloInstruction* custom_call) {
  for (const HloInstruction* operand : custom_call->operands()) {
    TF_RET_CHECK(
        LayoutUtil::IsMonotonicWithDim0Major(operand->shape().layout()));
  }
  TF_RET_CHECK(
      LayoutUtil::IsMonotonicWithDim0Major(custom_call->shape().layout()));
  return Status::OK();
}

// For a while instruction, all the following layouts must be the same:
//   (1) init operand
//   (2) condition computation parameter
//   (3) body computation parameter
//   (4) body computation result
//   (5) while instruction result
Status CheckWhileLayout(HloInstruction* while_inst,
                        const ComputationLayout& condition_computation_layout,
                        const ComputationLayout& body_computation_layout) {
  auto init_shape = while_inst->operand(0)->shape();
  TF_RET_CHECK(
      condition_computation_layout.parameter_layout(0).MatchesLayoutInShape(
          init_shape));
  TF_RET_CHECK(body_computation_layout.parameter_layout(0).MatchesLayoutInShape(
      init_shape));
  TF_RET_CHECK(
      body_computation_layout.result_layout().MatchesLayoutInShape(init_shape));
  TF_RET_CHECK(
      LayoutUtil::LayoutsInShapesEqual(init_shape, while_inst->shape()));
  return Status::OK();
}

// Fusion parameters must match the layout of the fusion instructions operands,
// and the root of the fusion expression must match the layout of the fusion
// instruction.
Status CheckFusionLayout(HloInstruction* fusion) {
  TF_RET_CHECK(HloOpcode::kFusion == fusion->opcode());

  TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
      fusion->shape(), fusion->fused_expression_root()->shape()));
  for (int64 i = 0; i < fusion->operand_count(); ++i) {
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        fusion->fused_parameter(i)->shape(), fusion->operand(i)->shape()));
  }
  return Status::OK();
}

// The layout of a parameter must match the respective layout in the
// computation's ComputationLayout.
Status CheckParameterLayout(HloInstruction* parameter,
                            const ComputationLayout& computation_layout) {
  const ShapeLayout& parameter_layout =
      computation_layout.parameter_layout(parameter->parameter_number());
  if (!parameter_layout.MatchesLayoutInShape(parameter->shape())) {
    return InternalError(
        "parameter instruction %s does not match layout of computation "
        "shape: %s",
        parameter->ToString().c_str(), parameter_layout.ToString().c_str());
  }
  return Status::OK();
}

// The layout of a constant instruction must match the layout of its literal.
Status CheckConstantLayout(HloInstruction* constant) {
  if (!LayoutUtil::LayoutsInShapesEqual(constant->literal().shape(),
                                        constant->shape())) {
    return InternalError(
        "constant instruction %s does not match the layout of its literal %s",
        constant->ToString().c_str(),
        ShapeUtil::HumanStringWithLayout(constant->literal().shape()).c_str());
  }
  return Status::OK();
}

// Check that all layouts in the module have been set and satisfy all necessary
// conditions.
Status CheckLayouts(
    HloModule* module,
    const std::map<HloComputation*, ComputationLayout>& computation_layouts) {
  TF_ASSIGN_OR_RETURN(auto points_to_analysis,
                      TuplePointsToAnalysis::Run(module));
  for (auto& computation : module->computations()) {
    for (auto& instruction : computation->instructions()) {
      // Verify every instruction has a layout and the layout is valid for the
      // shape.
      TF_RET_CHECK(LayoutUtil::HasLayout(instruction->shape()));
      TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

      // Use points-to analysis to verify that every subshape element in the
      // output of the instruction matches the layout of the logical buffer
      // which could be the source of the subshape value.
      const PointsToSet& points_to_set =
          points_to_analysis->GetPointsToSet(instruction.get());
      TF_RETURN_IF_ERROR(points_to_set.ForEachElementWithStatus(
          [&instruction](
              ShapeIndex index,
              const std::vector<const LogicalBuffer*>& buffers) -> Status {
            if (ShapeUtil::IsLeafIndex(instruction->shape(), index)) {
              const Shape& instruction_subshape =
                  ShapeUtil::GetSubshape(instruction->shape(), index);
              for (const LogicalBuffer* buffer : buffers) {
                if (!ShapeUtil::Equal(instruction_subshape, buffer->shape())) {
                  return InternalError(
                      "Layout of instruction %s at index {%s} does not match "
                      "source LogicalBuffer %s: %s vs %s",
                      instruction->name().c_str(),
                      tensorflow::str_util::Join(index, ",").c_str(),
                      buffer->ToString().c_str(),
                      ShapeUtil::HumanStringWithLayout(instruction_subshape)
                          .c_str(),
                      ShapeUtil::HumanStringWithLayout(buffer->shape())
                          .c_str());
                }
              }
            }
            return Status::OK();
          }));

      // Verify instructions that have special layout constraints.
      switch (instruction->opcode()) {
        case HloOpcode::kCall:
          TF_RETURN_IF_ERROR(CheckCallLayout(
              instruction.get(),
              FindOrDie(computation_layouts, instruction->to_apply())));
          break;
        case HloOpcode::kCustomCall:
          TF_RETURN_IF_ERROR(CheckCustomCallLayout(instruction.get()));
          break;
        case HloOpcode::kFusion:
          TF_RETURN_IF_ERROR(CheckFusionLayout(instruction.get()));
          break;
        case HloOpcode::kParameter:
          TF_RETURN_IF_ERROR(CheckParameterLayout(
              instruction.get(),
              FindOrDie(computation_layouts, instruction->parent())));
          break;
        case HloOpcode::kConstant:
          TF_RETURN_IF_ERROR(CheckConstantLayout(instruction.get()));
          break;
        case HloOpcode::kWhile:
          TF_RETURN_IF_ERROR(CheckWhileLayout(
              instruction.get(),
              FindOrDie(computation_layouts, instruction->while_condition()),
              FindOrDie(computation_layouts, instruction->while_body())));
          break;
        default:
          break;
      }
    }
  }

  // Finally verify the result layout matches the layout of the entry
  // computation root.
  TF_RET_CHECK(ShapeUtil::Equal(
      module->entry_computation()->root_instruction()->shape(),
      FindOrDie(computation_layouts, module->entry_computation())
          .result_layout()
          .shape()));

  return Status::OK();
}

}  // namespace

LayoutAssignment::LayoutAssignment(ComputationLayout* entry_computation_layout)
    : entry_computation_layout_(entry_computation_layout) {
  VLOG(1) << "entry computation layout given to layout assignment: "
          << entry_computation_layout_->ToString();
  // Layouts of all parameter instructions must be set.
  for (const ShapeLayout& parameter_layout :
       entry_computation_layout_->parameter_layouts()) {
    CHECK(parameter_layout.LayoutIsSet());
  }
  // If the result layout is not set, then choose the default.
  // TODO(b/29118294): Choose a better layout in this case.
  if (!entry_computation_layout_->result_layout().LayoutIsSet()) {
    entry_computation_layout_->mutable_result_layout()->SetToDefaultLayout();
  }
}

std::unique_ptr<Layout> LayoutAssignment::ChooseOperandLayoutFromOutputLayout(
    const Layout& output_layout, const HloInstruction* instruction,
    int64 operand_no) {
  const HloInstruction* operand = instruction->operand(operand_no);

  CHECK(ShapeUtil::IsArray(instruction->shape()) &&
        ShapeUtil::IsArray(operand->shape()));

  if (instruction->IsElementwiseOnOperand(operand_no) &&
      !ShapeUtil::IsScalar(operand->shape()) &&
      ShapeUtil::Rank(operand->shape()) ==
          ShapeUtil::Rank(instruction->shape())) {
    // Assign operands the same layout as the instruction, so that
    // 1) the elementwise operation can reuse its operand's buffer, and
    // 2) the input and output elements can reuse the same linear index.
    //
    // TODO(jingyue): Other operations, such as kSlice and kConcat, can benefit
    // from assigning the same layout to input and output.
    return MakeUnique<Layout>(output_layout);
  }

  if (instruction->opcode() == HloOpcode::kReshape) {
    // Prefer the operand layout that makes the reshape an bitcast. If any
    // dimension bound is 1 in the operand shape, there may be several such
    // layouts. So if 'output_layout' is a MajorToMinor layout, try if the
    // reshape is a bitcast when using the same layout. This may avoid copy
    // operations.
    const Shape& output_shape = instruction->shape();
    Shape output_shape_with_layout = ShapeUtil::MakeShapeWithLayout(
        output_shape.element_type(), AsInt64Slice(output_shape.dimensions()),
        AsInt64Slice(output_layout.minor_to_major()));
    const Shape& operand_shape = operand->shape();
    if (LayoutUtil::IsMonotonicWithDim0Major(output_layout)) {
      Shape operand_shape_with_layout =
          ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
              operand_shape.element_type(),
              AsInt64Slice(operand_shape.dimensions()));
      if (ShapeUtil::ReshapeIsBitcast(operand_shape_with_layout,
                                      output_shape_with_layout)) {
        return MakeUnique<Layout>(operand_shape_with_layout.layout());
      }
    }
    auto aligned_operand_shape =
        ShapeUtil::AlignLayouts(output_shape_with_layout, operand_shape);
    if (aligned_operand_shape) {
      auto operand_layout = aligned_operand_shape.value().layout();
      TF_CHECK_OK(
          LayoutUtil::ValidateLayoutForShape(operand_layout, operand_shape));
      return MakeUnique<Layout>(operand_layout);
    }
  }

  if (instruction->opcode() == HloOpcode::kTranspose) {
    // Pick the operand layout that makes the transpose a bitcast.
    std::vector<int64> perm =
        ComposePermutations(instruction->dimensions(),
                            AsInt64Slice(output_layout.minor_to_major()));
    Layout operand_layout = LayoutUtil::MakeLayout(perm);
    TF_CHECK_OK(
        LayoutUtil::ValidateLayoutForShape(operand_layout, operand->shape()));
    return MakeUnique<Layout>(operand_layout);
  }

  return nullptr;
}

std::unique_ptr<Layout> LayoutAssignment::ChooseOutputLayoutFromOperandLayout(
    const Layout& operand_layout, const HloInstruction* user,
    int64 operand_no) {
  const HloInstruction* operand = user->operand(operand_no);

  CHECK(ShapeUtil::IsArray(user->shape()) &&
        ShapeUtil::IsArray(operand->shape()));

  if (user->IsElementwiseOnOperand(operand_no) &&
      !ShapeUtil::IsScalar(operand->shape()) &&
      ShapeUtil::Rank(operand->shape()) == ShapeUtil::Rank(user->shape())) {
    // Assign users the same layout as the operand.
    return MakeUnique<Layout>(operand_layout);
  }

  if (user->opcode() == HloOpcode::kReshape) {
    // Prefer the user layout that makes the reshape an bitcast. If any
    // dimension bound is 1 in the user shape, there may be several such
    // layouts. So if 'operand_layout' is a MajorToMinor layout, try if the
    // reshape is a bitcast when using the same layout. This may avoid copy
    // operations.
    Shape operand_shape_with_layout = ShapeUtil::MakeShapeWithLayout(
        operand->shape().element_type(),
        AsInt64Slice(operand->shape().dimensions()),
        AsInt64Slice(operand_layout.minor_to_major()));
    const Shape& output_shape = user->shape();
    if (LayoutUtil::IsMonotonicWithDim0Major(operand_layout)) {
      Shape output_shape_with_layout =
          ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
              output_shape.element_type(),
              AsInt64Slice(output_shape.dimensions()));
      if (ShapeUtil::ReshapeIsBitcast(output_shape_with_layout,
                                      operand_shape_with_layout)) {
        return MakeUnique<Layout>(output_shape_with_layout.layout());
      }
    }
    auto aligned_user_shape =
        ShapeUtil::AlignLayouts(operand_shape_with_layout, output_shape);
    if (aligned_user_shape) {
      auto user_layout = aligned_user_shape.value().layout();
      TF_CHECK_OK(
          LayoutUtil::ValidateLayoutForShape(user_layout, output_shape));
      return MakeUnique<Layout>(user_layout);
    }
  }

  if (user->opcode() == HloOpcode::kTranspose) {
    // Pick the user layout that makes the reshape a bitcast.
    // To become a bitcast, the layouts need to satisfy
    //   collapsing_order * output_layout = input_layout
    // so output_layout = inverse(collapsing_order) * input_layout
    std::vector<int64> perm =
        Permute(InversePermutation(user->dimensions()),
                AsInt64Slice(operand_layout.minor_to_major()));
    Layout user_layout = LayoutUtil::MakeLayout(perm);
    TF_CHECK_OK(LayoutUtil::ValidateLayoutForShape(user_layout, user->shape()));
    return MakeUnique<Layout>(user_layout);
  }

  return nullptr;
}

Status LayoutAssignment::PropagateConstraints(LayoutConstraints* constraints) {
  // Gathers all initial constraints in a worklist and propagates them in
  // depth-first order. DFS order seems to be better than BFS because a
  // constraint is propagated as far as possible before propagating unrelated
  // constraints which makes it less likely that conflicting constraints will be
  // propagated to instructions. However, we should experiment with other orders
  // too.
  std::deque<const LayoutConstraint*> worklist;

  // Lambda for moving newly added constraints to the worklist.
  auto add_new_constraints_to_worklist = [constraints, &worklist]() {
    // Add constraints to the front of the deque for DFS ordering.
    for (auto* constraint : constraints->ConsumeAddedConstraints()) {
      worklist.push_front(constraint);
    }
  };
  add_new_constraints_to_worklist();

  while (!worklist.empty()) {
    const LayoutConstraint* layout_constraint = worklist.front();
    worklist.pop_front();
    VLOG(2) << "Propagating " << layout_constraint->ToString()
            << " to its neighbors.";
    if (auto* buffer_constraint =
            dynamic_cast<const BufferLayoutConstraint*>(layout_constraint)) {
      TF_RETURN_IF_ERROR(
          PropagateBufferConstraint(*buffer_constraint, constraints));
    } else if (auto* operand_constraint =
                   dynamic_cast<const OperandLayoutConstraint*>(
                       layout_constraint)) {
      TF_RETURN_IF_ERROR(
          PropagateOperandConstraint(*operand_constraint, constraints));
    } else if (auto* result_constraint =
                   dynamic_cast<const ResultLayoutConstraint*>(
                       layout_constraint)) {
      TF_RETURN_IF_ERROR(
          PropagateResultConstraint(*result_constraint, constraints));
    } else {
      LOG(FATAL) << "Invalid constraint type: " << *layout_constraint;
    }

    add_new_constraints_to_worklist();
  }
  return Status::OK();
}

namespace {

// Returns a vector containing all array-shaped uses (instruction and operand
// number) of the given logical buffer or its aliases.
std::vector<std::pair<const HloInstruction*, int64>> GetArrayUsesOfBuffer(
    const LogicalBuffer& buffer,
    const TuplePointsToAnalysis& points_to_analysis) {
  CHECK(buffer.IsArray());
  std::vector<std::pair<const HloInstruction*, int64>> uses;
  for (const auto& buffer_alias : points_to_analysis.GetBufferAliases(buffer)) {
    if (!ShapeUtil::IsArray(buffer_alias.instruction()->shape())) {
      continue;
    }
    // This alias must be the top-level (index == {}) of the instruction's
    // result because the instruction produces an array.
    CHECK(buffer_alias.index().empty());

    // Add all uses of the instruction's output.
    for (const HloInstruction* user : buffer_alias.instruction()->users()) {
      for (int64 operand_no :
           user->OperandIndices(buffer_alias.instruction())) {
        uses.emplace_back(user, operand_no);
      }
    }
  }
  return uses;
}

}  // namespace

Status LayoutAssignment::PropagateUseConstraintToDefs(
    const ShapeLayout& shape_layout, const HloInstruction* instruction,
    LayoutConstraints* constraints) {
  // Try to set all logical buffers which may be sources of the given operand to
  // match the given layout.
  const PointsToSet& points_to_set =
      constraints->points_to_analysis().GetPointsToSet(instruction);
  return points_to_set.ForEachElementWithStatus(
      [this, &shape_layout, constraints](
          const ShapeIndex& index,
          const std::vector<const LogicalBuffer*>& buffers) -> Status {
        if (ShapeUtil::IsLeafIndex(shape_layout.shape(), index)) {
          for (const LogicalBuffer* buffer : buffers) {
            if (constraints->BufferLayout(*buffer) == nullptr &&
                ShapeUtil::IsArray(buffer->shape())) {
              TF_RETURN_IF_ERROR(constraints->SetBufferLayout(
                  ShapeUtil::GetSubshape(shape_layout.shape(), index).layout(),
                  *buffer));
            }
          }
        }
        return Status::OK();
      });
}

Status LayoutAssignment::PropagateOperandConstraint(
    const OperandLayoutConstraint& operand_constraint,
    LayoutConstraints* constraints) {
  // Try to set the layout of the logical buffers in the given operand to match
  // the constrained layout. This avoids copies.
  TF_RETURN_IF_ERROR(
      PropagateUseConstraintToDefs(operand_constraint.shape_layout(),
                                   operand_constraint.operand(), constraints));

  // For array-shaped operands and user instructions try to pick a minimum cost
  // layout. For example, if the operand of a elementwise instruction is
  // constained to a certain layout we want the output of the instruction to
  // have the same layout.
  const HloInstruction* operand = operand_constraint.operand();
  const HloInstruction* user = operand_constraint.instruction();
  if (!ShapeUtil::IsArray(operand->shape()) ||
      !ShapeUtil::IsArray(user->shape())) {
    return Status::OK();
  }

  // Only try to choose a low cost layout if the instruction 'user' defines its
  // output (ie, doesn't forward a buffer from elsewhere).
  if (constraints->OperandBufferForwarded(user,
                                          operand_constraint.operand_no())) {
    return Status::OK();
  }
  TF_ASSIGN_OR_RETURN(
      const LogicalBuffer* buffer,
      constraints->points_to_analysis().GetBufferDefinedAt(user, /*index=*/{}));

  if (constraints->BufferLayout(*buffer) == nullptr) {
    std::unique_ptr<Layout> layout = ChooseOutputLayoutFromOperandLayout(
        operand_constraint.shape_layout().layout(), user,
        operand_constraint.operand_no());
    if (layout != nullptr) {
      TF_RETURN_IF_ERROR(
          constraints->SetBufferLayout(*layout, *buffer, /*mandatory=*/false));
    }
  }
  return Status::OK();
}

Status LayoutAssignment::PropagateBufferConstraint(
    const BufferLayoutConstraint& buffer_constraint,
    LayoutConstraints* constraints) {
  // Only propagate array layouts.
  const LogicalBuffer& buffer = buffer_constraint.buffer();
  if (!buffer.IsArray()) {
    return Status::OK();
  }

  // If this buffer is the result of an array-shaped op (as opposed to an array
  // element in a tuple) try to propagate the layout to its operands.
  if (buffer.IsTopLevel()) {
    const HloInstruction* instruction = buffer.instruction();
    // Propagate the def-constraint on an instruction to the use-constraints on
    // its operands (use-def propagation).
    for (int64 operand_no = 0; operand_no < instruction->operand_count();
         ++operand_no) {
      if (constraints->OperandLayout(instruction, operand_no) == nullptr &&
          ShapeUtil::IsArray(instruction->operand(operand_no)->shape())) {
        std::unique_ptr<Layout> operand_layout =
            ChooseOperandLayoutFromOutputLayout(buffer_constraint.layout(),
                                                instruction, operand_no);
        if (operand_layout != nullptr) {
          TF_RETURN_IF_ERROR(constraints->SetArrayOperandLayout(
              *operand_layout, instruction, operand_no, /*mandatory=*/true));
        }
      }
    }
  }
  return PropagateBufferConstraintToUses(buffer_constraint, constraints);
}

Status LayoutAssignment::PropagateBufferConstraintToUses(
    const BufferLayoutConstraint& buffer_constraint,
    LayoutConstraints* constraints) {
  const LogicalBuffer& buffer = buffer_constraint.buffer();
  TF_RET_CHECK(buffer.IsArray());

  // Propagate the layout to all array uses of the logical buffer. This skips
  // uses of the buffer where the buffer is the element of a tuple.
  for (const auto& user_operand_no :
       GetArrayUsesOfBuffer(buffer, constraints->points_to_analysis())) {
    const HloInstruction* user = user_operand_no.first;
    int64 operand_no = user_operand_no.second;
    // Only add an operand constraint if the user does not forward the buffer
    // because this case is not handled is SetOperandLayout.
    if (constraints->OperandLayout(user, operand_no) == nullptr &&
        !constraints->OperandBufferForwarded(user, operand_no)) {
      TF_RETURN_IF_ERROR(constraints->SetArrayOperandLayout(
          buffer_constraint.layout(), user, operand_no, /*mandatory=*/false));
    }
  }

  return Status::OK();
}

Status LayoutAssignment::PropagateResultConstraint(
    const ResultLayoutConstraint& result_constraint,
    LayoutConstraints* constraints) {
  // Propagate the use constraint of the root instruction up to the logical
  // buffers which make up the result.
  return PropagateUseConstraintToDefs(
      result_constraint.shape_layout(),
      constraints->computation()->root_instruction(), constraints);
}

namespace {

// Infers the layout of the array at the given index in the given instruction's
// output using points-to analysis. Precondition: The given instruction must
// not produce this array value (that is, the array is forwarded from the
// instruction's operands).
StatusOr<Layout> InferArrayLayout(
    const TuplePointsToAnalysis& points_to_analysis,
    HloInstruction* instruction, const ShapeIndex& index) {
  // This function should only be called for array shapes which don't yet have
  // layouts.
  const Shape& subshape = ShapeUtil::GetSubshape(instruction->shape(), index);
  TF_RET_CHECK(ShapeUtil::IsArray(subshape));
  TF_RET_CHECK(!subshape.has_layout());

  // The instruction should not define the buffer at this index.
  TF_RET_CHECK(
      !points_to_analysis.InstructionDefinesBufferAtIndex(instruction, index));

  const std::vector<const LogicalBuffer*>& source_buffers =
      points_to_analysis.GetPointsToSet(instruction).element(index);
  TF_RET_CHECK(!source_buffers.empty());

  // Verify the layout is the same for every LogicalBuffer which this location
  // ('instruction' and 'index') points to.
  const Layout* first_buffer_layout = nullptr;
  for (const LogicalBuffer* source_buffer : source_buffers) {
    if (!source_buffer->shape().has_layout()) {
      // This should not happen because we've assigned layouts to all
      // instructions preceding this one.
      return InternalError("LogicalBuffer %s does not have a layout",
                           source_buffer->ToString().c_str());
    }

    if (first_buffer_layout == nullptr) {
      first_buffer_layout = &source_buffer->shape().layout();
    } else if (!LayoutUtil::Equal(source_buffer->shape().layout(),
                                  *first_buffer_layout)) {
      // The points-to set is ambiguous for this index and the different source
      // buffers have different layouts. This case is possible in valid XLA
      // computations because we do not propagate BufferLayoutConstraints to all
      // LogicalBuffers which may alias the constrained LogicalBuffer at some
      // point in the computation.
      return FailedPrecondition(
          "Array at index {%s} in instruction %s aliases buffers %s "
          "and %s which have different layouts",
          tensorflow::str_util::Join(index, ",").c_str(),
          instruction->name().c_str(), source_buffers[0]->ToString().c_str(),
          source_buffer->ToString().c_str());
    }
  }

  return *first_buffer_layout;
}

// Creates and returns a copy of the given instruction with a different
// layout. Tuple-shaped instructions will be deep-copied, and the last Tuple
// instruction producing the copy is returned.
StatusOr<HloInstruction*> CreateCopyWithNewLayout(
    const Shape& shape_with_layout, HloInstruction* instruction) {
  TF_RET_CHECK(LayoutUtil::HasLayout(shape_with_layout));
  DCHECK(ShapeUtil::Compatible(shape_with_layout, instruction->shape()));

  if (ShapeUtil::IsTuple(instruction->shape())) {
    // Deep-copy tuples.
    std::vector<HloInstruction*> element_copies;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(instruction->shape());
         ++i) {
      HloInstruction* gte = instruction->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(
              ShapeUtil::GetSubshape(instruction->shape(), {i}), instruction,
              i));

      // Recurse to copy each elements.
      TF_ASSIGN_OR_RETURN(
          HloInstruction * element_copy,
          CreateCopyWithNewLayout(
              ShapeUtil::GetSubshape(shape_with_layout, {i}), gte));
      element_copies.push_back(element_copy);
    }
    // Gather element copies into a tuple with a new Tuple instruction.
    HloInstruction* tuple_copy = instruction->parent()->AddInstruction(
        HloInstruction::CreateTuple(element_copies));
    LayoutUtil::ClearLayout(tuple_copy->mutable_shape());
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        shape_with_layout, tuple_copy->mutable_shape()));
    return tuple_copy;
  } else if (ShapeUtil::IsArray(instruction->shape())) {
    HloInstruction* copy =
        instruction->parent()->AddInstruction(HloInstruction::CreateUnary(
            instruction->shape(), HloOpcode::kCopy, instruction));
    LayoutUtil::ClearLayout(copy->mutable_shape());
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        shape_with_layout, copy->mutable_shape()));

    return copy;
  } else {
    return FailedPrecondition(
        "Can only copy array and tuple shaped instructions");
  }
}

// Creates a copy of the given operand if the operand's layout does not match
// the given layout. This copy replaces the use in the given instruction. Tuple
// operands will be deep-copied.
Status CopyOperandIfLayoutsDiffer(const ShapeLayout& operand_layout,
                                  HloInstruction* instruction,
                                  int64 operand_no) {
  HloInstruction* operand = instruction->mutable_operand(operand_no);
  TF_RET_CHECK(operand_layout.LayoutIsSet());
  TF_RET_CHECK(LayoutUtil::HasLayout(operand->shape()));

  if (ShapeUtil::Equal(operand_layout.shape(), operand->shape())) {
    // Operand layout already matches our constraint. Nothing to do.
    return Status::OK();
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * operand_copy,
                      CreateCopyWithNewLayout(operand_layout.shape(), operand));

  return instruction->ReplaceOperandWith(operand_no, operand_copy);
}

// For fusion instructions, set the layout of each fused parameter instruction
// to match the layout of its corresponding fusion instruction operand. Also,
// set the layout of the fused root to match the layout of the fusion
// instruction itself.
// Fused GetTupleElement requires a layout so that TBAA metadata for the tuple
// element array pointer load can be added.
Status SetFusionLayouts(HloInstruction* fusion) {
  TF_RET_CHECK(fusion->opcode() == HloOpcode::kFusion);
  for (auto& fused_instruction : fusion->fused_instructions()) {
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      const HloInstruction* fusion_operand =
          fusion->operand(fused_instruction->parameter_number());
      DCHECK(ShapeUtil::Compatible(fusion_operand->shape(),
                                   fused_instruction->shape()));
      TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
          fusion_operand->shape(), fused_instruction->mutable_shape()));
    } else if (fused_instruction.get() == fusion->fused_expression_root()) {
      // The layout of the root of the fused expression must match the fusion
      // instruction layout.
      DCHECK(
          ShapeUtil::Compatible(fusion->shape(), fused_instruction->shape()));
      TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
          fusion->shape(), fused_instruction->mutable_shape()));
    } else if (fused_instruction->opcode() != HloOpcode::kConstant &&
               fused_instruction->opcode() != HloOpcode::kGetTupleElement &&
               fused_instruction->opcode() != HloOpcode::kInfeed) {
      // Internal fused instructions with the exception of constants
      // and infeed need no layout.
      LayoutUtil::ClearLayout(fused_instruction->mutable_shape());
    }
  }

  return Status::OK();
}

}  // namespace

Status LayoutAssignment::AssignLayouts(const LayoutConstraints& constraints,
                                       HloComputation* computation) {
  VLOG(2) << "Assigning layouts to computation: " << computation->name();
  XLA_VLOG_LINES(2, computation->ToString());
  XLA_VLOG_LINES(2, constraints.ToString());

  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    LayoutUtil::ClearLayout(instruction->mutable_shape());

    // Create a copy of an operand if the operand instruction's layout does not
    // match the use constraint (OperandLayoutConstraint).
    for (int64 operand_no = 0; operand_no < instruction->operand_count();
         ++operand_no) {
      const ShapeLayout* operand_layout =
          constraints.OperandLayout(instruction, operand_no);
      if (operand_layout != nullptr) {
        TF_RETURN_IF_ERROR(CopyOperandIfLayoutsDiffer(*operand_layout,
                                                      instruction, operand_no));
      }
    }

    // Set the layouts of the array shapes this instruction defines as
    // indicated by the respective BufferLayoutConstraints. Any array shapes
    // in the output of the instruction which are not defined by the instruction
    // (eg, array elements in a Tuple instruction) will be assigned below via
    // inference.
    for (const LogicalBuffer* buffer :
         constraints.points_to_analysis().GetBuffersDefinedByInstruction(
             instruction)) {
      if (!ShapeUtil::IsArray(buffer->shape())) {
        continue;
      }

      TF_RET_CHECK(buffer->instruction() == instruction);
      Shape* buffer_subshape = ShapeUtil::GetMutableSubshape(
          instruction->mutable_shape(), buffer->index());
      const Layout* buffer_layout = constraints.BufferLayout(*buffer);
      TF_RET_CHECK(buffer_layout != nullptr);
      *buffer_subshape->mutable_layout() = *buffer_layout;
    }

    // Any remaining layouts in the output of the instruction must be
    // inferrable using points-to analysis.
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachMutableSubshapeWithStatus(
        instruction->mutable_shape(),
        [instruction, &constraints](Shape* subshape, const ShapeIndex& index) {
          if (subshape->has_layout() || !ShapeUtil::IsArray(*subshape)) {
            return Status::OK();
          }
          // Set Layout of subshape to match layout of LogicalBuffer which
          // produces it.
          TF_ASSIGN_OR_RETURN(*subshape->mutable_layout(),
                              InferArrayLayout(constraints.points_to_analysis(),
                                               instruction, index));
          return Status::OK();
        }));

    // Fusion instructions require some layouts to be set on fused instructions
    // inside the fusion instruction.
    if (instruction->opcode() == HloOpcode::kFusion) {
      TF_RETURN_IF_ERROR(SetFusionLayouts(instruction));
    }

    // Execute extra verification step once the layout has been finalized.
    TF_RETURN_IF_ERROR(Verify(instruction));

    // Verify all layouts in the shape have been set.
    TF_RET_CHECK(LayoutUtil::HasLayout(instruction->shape()));
  }

  // Copy the root instrucion's result if the it does not match the result
  // layout constraint
  if (constraints.ResultLayout() != nullptr &&
      !constraints.ResultLayout()->MatchesLayoutInShape(
          computation->root_instruction()->shape())) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_root,
        CreateCopyWithNewLayout(constraints.ResultLayout()->shape(),
                                computation->root_instruction()));
    computation->set_root_instruction(new_root);
  }

  return Status::OK();
}

Status LayoutAssignment::RunOnComputation(
    const ComputationLayout& computation_layout, HloComputation* computation) {
  DCHECK(computation_layout.LayoutIsSet());
  InsertOrDie(&computation_layouts_, computation, computation_layout);
  VLOG(2) << "LayoutAssignment::RunOnComputation(" << computation->name()
          << ")";
  VLOG(2) << "  ComputationLayout = " << computation_layout.ToString();

  TF_ASSIGN_OR_RETURN(auto points_to_analysis,
                      TuplePointsToAnalysis::Run(computation->parent()));

  // Construct LayoutConstraints with all layout constraints of the computation.
  LayoutConstraints constraints(*points_to_analysis, computation);

  // Add constraints required for correctness on all backends (eg, entry
  // parameter layout constraints).
  TF_RETURN_IF_ERROR(
      AddMandatoryConstraints(computation_layout, computation, &constraints));

  // Add any backend-specific constraints.
  TF_RETURN_IF_ERROR(AddBackendConstraints(&constraints));

  // Propagates layouts from an HLO to its neighbors.
  TF_RETURN_IF_ERROR(PropagateConstraints(&constraints));

  // While any unconstrained buffers remain, pick an arbitrary buffer, give it a
  // layout and propagate the change.
  while (!constraints.unconstrained_buffer_ids().empty()) {
    int unconstrained_count = constraints.unconstrained_buffer_ids().size();

    // Arbitrarily pick the first unconstrained buffer and give it the default
    // layout. By construction unconstrained_buffers() has a stable sort based
    // on LogicalBuffer::Id.
    const LogicalBuffer& buffer = points_to_analysis->GetBuffer(
        *constraints.unconstrained_buffer_ids().begin());
    TF_RETURN_IF_ERROR(constraints.SetBufferLayout(
        LayoutUtil::GetDefaultLayoutForShape(buffer.shape()), buffer,
        /*mandatory=*/false));

    TF_RETURN_IF_ERROR(PropagateConstraints(&constraints));

    // To verify progress has been made, check that the number of unconstrained
    // buffers has been reduced.
    CHECK_LT(constraints.unconstrained_buffer_ids().size(),
             unconstrained_count);
  }

  // All logical buffers should have constraints at this point. All that
  // remains is assign the constraints to the buffers and infer layouts for
  // aliased buffers.
  return AssignLayouts(constraints, computation);
}

StatusOr<bool> LayoutAssignment::Run(HloModule* module) {
  VLOG(2) << "Running layout assignment on module " << module->name();
  XLA_VLOG_LINES(3, module->ToString());
  if (VLOG_IS_ON(10)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "before layout assignment",
                                /*show_addresses=*/false,
                                /*show_layouts=*/true);
  }

  // Assign layouts to computations in an order such that a callee computation
  // is handled before its caller computation. This ensures that the layout of
  // all callers of a computation will agree.
  for (auto* computation : module->MakeComputationPostOrder()) {
    if (computation == module->entry_computation()) {
      TF_RETURN_IF_ERROR(RunOnComputation(*entry_computation_layout_,
                                          module->entry_computation()));
    } else {
      ComputationLayout computation_layout(computation->ComputeProgramShape());
      // Setting all embedded computations to the default layout is potentially
      // suboptimal.
      computation_layout.SetToDefaultLayout();
      TF_RETURN_IF_ERROR(RunOnComputation(computation_layout, computation));
    }
  }

  TF_RETURN_IF_ERROR(CheckLayouts(module, computation_layouts_));

  VLOG(3) << "After layout assignment:";
  XLA_VLOG_LINES(3, module->ToString());
  if (VLOG_IS_ON(10)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "after layout assignment",
                                /*show_addresses=*/false,
                                /*show_layouts=*/true);
  }

  // All layouts are reset then reassigned by this pass.
  return true;
}

}  // namespace xla

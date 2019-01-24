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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
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
                                               bool mandatory, bool dfs)
    : LayoutConstraint(mandatory, dfs), layout_(layout), buffer_(&buffer) {
  CHECK(LayoutUtil::ValidateLayoutForShape(layout, buffer.shape()).ok());
}

string BufferLayoutConstraint::ToString() const {
  return absl::StrFormat("BufferLayoutConstraint %s: %s", buffer_->ToString(),
                         LayoutUtil::HumanString(layout_));
}

OperandLayoutConstraint::OperandLayoutConstraint(
    const ShapeLayout& shape_layout, const HloInstruction* instruction,
    int64 operand_no, bool mandatory, bool dfs)
    : LayoutConstraint(mandatory, dfs),
      shape_layout_(shape_layout),
      instruction_(instruction),
      operand_no_(operand_no) {
  CHECK(shape_layout_.LayoutIsSet());
  CHECK(ShapeUtil::Compatible(shape_layout.shape(),
                              instruction->operand(operand_no)->shape()))
      << shape_layout.shape() << " is not compatible with "
      << instruction->operand(operand_no)->shape() << " (for operand "
      << operand_no << " of instruction " << instruction->ToString() << ")";
}

string OperandLayoutConstraint::ToString() const {
  return absl::StrFormat("OperandLayoutConstraint %s, operand %d: %s",
                         instruction_->name(), operand_no_,
                         shape_layout_.ToString());
}

string ResultLayoutConstraint::ToString() const {
  return absl::StrFormat("ResultLayoutConstraint: %s",
                         shape_layout_.ToString());
}

LayoutConstraints::LayoutConstraints(
    const TuplePointsToAnalysis& points_to_analysis,
    HloComputation* computation)
    : points_to_analysis_(points_to_analysis), computation_(computation) {
  // Gather all array-shaped logical buffers into unconstrained_buffer_ids.
  for (HloInstruction* inst : computation_->instructions()) {
    points_to_analysis_.GetPointsToSet(inst).ForEachElement(
        [&](const ShapeIndex&, const PointsToSet::BufferList& buffers) {
          for (const LogicalBuffer* buffer : buffers) {
            // The points to analysis is computed per module, restrict
            // constraints to array buffers in this computation.
            if (buffer->IsArray() &&
                buffer->instruction()->parent() == computation) {
              unconstrained_buffer_ids_.insert(buffer->id());
            }
          }
        });
  }
}

PointsToSet::BufferSet* LayoutConstraints::GetBufferSet(
    const HloInstruction* instruction) const {
  auto it = buffer_sets_cache_.find(instruction);
  if (it != buffer_sets_cache_.end()) {
    return it->second.get();
  }
  auto& buffer_set =
      buffer_sets_cache_
          .emplace(instruction, absl::make_unique<PointsToSet::BufferSet>())
          .first->second;
  const auto& points_to_set = points_to_analysis_.GetPointsToSet(instruction);
  points_to_set.ForEachElement(
      [&buffer_set](const ShapeIndex& /*index*/,
                    const PointsToSet::BufferList& buffers) {
        buffer_set->insert(buffers.begin(), buffers.end());
      });
  return buffer_set.get();
}

bool LayoutConstraints::OperandBufferForwarded(
    const HloInstruction* instruction, int64 operand_no) const {
  // The operand is potentially forwarded if the intersection of points-to sets
  // of the operand and the instruction is non-empty.
  PointsToSet::BufferSet* output_buffers = GetBufferSet(instruction);
  PointsToSet::BufferSet* operand_buffers =
      GetBufferSet(instruction->operand(operand_no));
  return absl::c_any_of(*output_buffers, [&](const LogicalBuffer* b) {
    return operand_buffers->count(b) > 0;
  });
}

Status LayoutConstraints::SetBufferLayout(const Layout& layout,
                                          const LogicalBuffer& buffer,
                                          bool mandatory, bool dfs) {
  VLOG(3) << "SetBufferLayout : " << buffer << " : "
          << LayoutUtil::HumanString(layout);

  TF_RETURN_IF_ERROR(points_to_analysis_.VerifyBuffer(buffer));
  if (!buffer.IsArray()) {
    return FailedPrecondition(
        "Layout of buffer %s cannot be constrained because buffer is not "
        "array-shaped, has shape: %s",
        buffer.ToString(), ShapeUtil::HumanString(buffer.shape()));
  }
  TF_RETURN_IF_ERROR(
      LayoutUtil::ValidateLayoutForShape(layout, buffer.shape()));

  auto iter = buffer_constraints_.find(&buffer);
  if (iter != buffer_constraints_.end()) {
    const BufferLayoutConstraint& curr_constraint = iter->second;
    if (LayoutUtil::Equal(curr_constraint.layout(), layout)) {
      // New constraint matches existing constraint. Nothing to do.
      return Status::OK();
    }
    if (curr_constraint.mandatory()) {
      return FailedPrecondition(
          "Buffer %s already has the layout constraint %s, cannot add "
          "incompatible constraint %s",
          buffer.ToString(), LayoutUtil::HumanString(curr_constraint.layout()),
          LayoutUtil::HumanString(layout));
    }
    iter->second = BufferLayoutConstraint(layout, buffer, mandatory, dfs);
  } else {
    TF_RET_CHECK(unconstrained_buffer_ids_.erase(buffer.id()) == 1)
        << buffer.ToString();
    iter = buffer_constraints_
               .insert(std::make_pair(
                   &buffer,
                   BufferLayoutConstraint(layout, buffer, mandatory, dfs)))
               .first;
  }
  added_constraints_.push_back(&iter->second);
  return Status::OK();
}

Status LayoutConstraints::SetOperandLayout(const Shape& shape_with_layout,
                                           const HloInstruction* instruction,
                                           int64 operand_no, bool mandatory,
                                           bool dfs) {
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
          "Operand %d of instruction %s already has a layout constraint "
          "%s, cannot add incompatible constraint %s",
          operand_no, instruction->name(),
          curr_shape_layout->shape_layout().ToString(),
          ShapeUtil::HumanStringWithLayout(shape_with_layout));
    }
  }

  // If any buffers in the operand occur in the output of the instruction, then
  // return an error. This case is not handled because such a constraint changes
  // layouts beyond this immediate use and is complicated to handle.
  if (OperandBufferForwarded(instruction, operand_no)) {
    return FailedPrecondition(
        "Cannot constraint layout of operand %d of instruction %s "
        "because instruction forwards operand's LogicalBuffer(s)",
        operand_no, instruction->name());
  }

  auto key = std::make_pair(instruction, operand_no);
  auto iter = operand_constraints_.find(key);
  if (iter == operand_constraints_.end()) {
    auto pair = std::make_pair(
        key, OperandLayoutConstraint(ShapeLayout(shape_with_layout),
                                     instruction, operand_no, mandatory, dfs));
    iter = operand_constraints_.insert(pair).first;
  } else {
    iter->second =
        OperandLayoutConstraint(ShapeLayout(shape_with_layout), instruction,
                                operand_no, mandatory, dfs);
  }
  added_constraints_.push_back(&iter->second);

  return Status::OK();
}

Status LayoutConstraints::SetArrayOperandLayout(
    const Layout& layout, const HloInstruction* instruction, int64 operand_no,
    bool mandatory, bool dfs) {
  const HloInstruction* operand = instruction->operand(operand_no);
  TF_RET_CHECK(operand->shape().IsArray());
  Shape shape(operand->shape());
  *shape.mutable_layout() = layout;
  TF_RETURN_IF_ERROR(LayoutUtil::ValidateLayoutInShape(shape));
  return SetOperandLayout(shape, instruction, operand_no, mandatory, dfs);
}

Status LayoutConstraints::SetResultLayout(const Shape& shape_with_layout,
                                          bool dfs) {
  VLOG(3) << "SetResultLayout : "
          << ShapeUtil::HumanStringWithLayout(shape_with_layout);

  const ShapeLayout* curr_shape_layout = ResultLayout();
  if (curr_shape_layout != nullptr) {
    if (!curr_shape_layout->MatchesLayoutInShape(shape_with_layout)) {
      return FailedPrecondition(
          "Result of computation %s already has the layout constraint %s, "
          "cannot add incompatible constraint %s",
          computation_->name(), curr_shape_layout->ToString(),
          ShapeUtil::HumanStringWithLayout(shape_with_layout));
    }
    // New constraint matches existing constraint. Nothing to do.
    return Status::OK();
  }

  result_constraint_.reset(
      new ResultLayoutConstraint(ShapeLayout(shape_with_layout), dfs));
  added_constraints_.push_back(result_constraint_.get());

  return Status::OK();
}

Status LayoutConstraints::SetInstructionLayout(
    const Shape& shape_with_layout, const HloInstruction* instruction,
    bool mandatory, bool dfs) {
  VLOG(3) << "SetInstructionLayout : " << instruction->name() << ", "
          << ShapeUtil::HumanStringWithLayout(shape_with_layout);

  if (!ShapeUtil::Compatible(shape_with_layout, instruction->shape())) {
    return FailedPrecondition(
        "Instruction %s of shape %s cannot be assigned incompatible layout %s",
        instruction->name(), ShapeUtil::HumanString(instruction->shape()),
        ShapeUtil::HumanStringWithLayout(shape_with_layout));
  }

  // Create a BufferLayoutConstraint for each array shape in the output of the
  // instruction.
  return ShapeUtil::ForEachSubshapeWithStatus(
      shape_with_layout,
      [this, instruction, mandatory](const Shape& subshape,
                                     const ShapeIndex& index) -> Status {
        // The precondition for this method is that the instruction defines all
        // buffers in its output.
        auto buffers =
            points_to_analysis_.GetPointsToSet(instruction).element(index);
        CHECK_EQ(1, buffers.size());
        CHECK_EQ(buffers[0]->instruction(), instruction);

        if (subshape.IsArray()) {
          return SetBufferLayout(subshape.layout(), *buffers[0], mandatory);
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
  absl::StrAppend(&output, "LayoutConstraints for computation ",
                  computation_->name(), ":\n");
  for (auto* instruction : computation_->MakeInstructionPostOrder()) {
    absl::StrAppend(&output, "  ", instruction->ToShortString(), "\n");
    for (int64 i = 0; i < instruction->operand_count(); ++i) {
      if (OperandLayout(instruction, i) != nullptr) {
        absl::StrAppend(&output, "    operand (", i,
                        "): ", OperandLayout(instruction, i)->ToString(), "\n");
      }
    }
    for (const LogicalBuffer* buffer :
         points_to_analysis_.GetBuffersDefinedByInstruction(instruction)) {
      if (BufferLayout(*buffer) != nullptr) {
        absl::StrAppend(&output, "    ", buffer->ToString(), " : ",
                        LayoutUtil::HumanString(*BufferLayout(*buffer)), "\n");
      }
    }
  }

  if (ResultLayout() != nullptr) {
    absl::StrAppend(&output, "  => ", ResultLayout()->ToString(), "\n");
  }
  return output;
}

namespace {

bool IsHostSendRecv(const HloInstruction* instruction) {
  const HloSendRecvInstruction* send_recv_instr =
      DynCast<HloSendRecvInstruction>(instruction);
  return send_recv_instr != nullptr && send_recv_instr->is_host_transfer();
}

}  // namespace

Status LayoutAssignment::BuildHostChannelConstraints(
    HloComputation* computation) {
  for (auto* instruction : computation->instructions()) {
    const HloSendRecvInstruction* send_recv_instr =
        DynCast<HloSendRecvInstruction>(instruction);
    if (send_recv_instr == nullptr || !send_recv_instr->is_host_transfer()) {
      continue;
    }

    // For host transfers the Send and Recv instruction carry the layout.
    if (instruction->opcode() == HloOpcode::kSend ||
        instruction->opcode() == HloOpcode::kRecv) {
      const Shape& data_shape =
          ShapeUtil::GetTupleElementShape(send_recv_instr->shape(), 0);
      TF_RET_CHECK(data_shape.IsArray());
      TF_RET_CHECK(LayoutUtil::HasLayout(data_shape));
      const Layout* prev_layout = host_channel_constraints_.ConstrainChannel(
          send_recv_instr->channel_id(), data_shape.layout());
      TF_RET_CHECK(prev_layout == nullptr)
          << "Cannot constrain host transfer layout as it was set to "
          << LayoutUtil::HumanString(*prev_layout) << ": "
          << send_recv_instr->ToString();
    }
  }
  return Status::OK();
}

namespace {

bool IsLayoutConstrainedCustomCall(HloInstruction* instruction) {
  const HloCustomCallInstruction* custom_call =
      DynCast<HloCustomCallInstruction>(instruction);
  return custom_call != nullptr && custom_call->layout_constrained();
}

}  // namespace

Status LayoutAssignment::AddMandatoryConstraints(
    const ComputationLayout* computation_layout,
    ChannelLayoutConstraints* channel_constraints, HloComputation* computation,
    LayoutConstraints* constraints) {
  VLOG(3) << "Adding mandatory layout constraints to computation "
          << computation->name();

  auto get_channel_constraints = [&](const HloInstruction* instruction) {
    return IsHostSendRecv(instruction) ? &host_channel_constraints_
                                       : channel_constraints;
  };

  // Constrain layouts of instructions which define values with pre-existing
  // layouts.
  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kInfeed) {
      // Infeed layouts must match the layout of the original inserted
      // instruction.
      // TODO(b/31425034): Change infeeds to be more like parameters, with
      // shapes in the ComputationLayout.
      TF_RETURN_IF_ERROR(
          constraints->SetInstructionLayout(instruction->shape(), instruction));
    } else if (instruction->opcode() == HloOpcode::kOutfeed) {
      // Constrain the input to the Outfeed instruction to be the expected
      // layout of the Outfeed.
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
          instruction->outfeed_shape(), instruction, 0));
    } else if (instruction->opcode() == HloOpcode::kParameter) {
      if (computation_layout != nullptr) {
        const ShapeLayout& parameter_layout =
            computation_layout->parameter_layout(
                instruction->parameter_number());
        if (parameter_layout.LayoutIsSet()) {
          // Parameter layouts must match the respective layout in
          // ComputationLayout, if there is one.
          TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(
              parameter_layout.shape(), instruction));
        }
      }
    } else if (IsLayoutConstrainedCustomCall(instruction)) {
      const HloCustomCallInstruction* custom_call =
          DynCast<HloCustomCallInstruction>(instruction);
      TF_RETURN_IF_ERROR(
          constraints->SetInstructionLayout(custom_call->shape(), custom_call));
      for (int64 i = 0; i < custom_call->operand_count(); ++i) {
        TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
            custom_call->operand_shapes_with_layout()[i], custom_call, i));
      }
    } else if (instruction->opcode() == HloOpcode::kSend ||
               instruction->opcode() == HloOpcode::kRecv) {
      CHECK(get_channel_constraints(instruction))
          << "Multi-module layout assignment requires ChannelLayoutConstraints";
      int64 channel_id = instruction->channel_id();
      if (!get_channel_constraints(instruction)
               ->IsChannelConstrained(channel_id)) {
        continue;
      }
      if (instruction->opcode() == HloOpcode::kSend) {
        // TODO(b/68493863): Change to use SetOperandLayout().
        const Shape send_buffer_shape = instruction->operand(0)->shape();
        TF_RET_CHECK(send_buffer_shape.IsArray());
        Shape new_buffer_shape =
            get_channel_constraints(instruction)
                ->LayoutShapeForChannel(send_buffer_shape,
                                        instruction->channel_id());
        TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(
            new_buffer_shape, instruction->operand(0)));
      } else {
        const Shape recv_buffer_shape =
            ShapeUtil::GetTupleElementShape(instruction->shape(), 0);
        TF_RET_CHECK(recv_buffer_shape.IsArray());
        TF_ASSIGN_OR_RETURN(
            const LogicalBuffer* buffer,
            constraints->points_to_analysis().GetBufferDefinedAt(instruction,
                                                                 {0}));
        Shape new_shape = get_channel_constraints(instruction)
                              ->LayoutShapeForChannel(
                                  recv_buffer_shape, instruction->channel_id());
        TF_RETURN_IF_ERROR(
            constraints->SetBufferLayout(new_shape.layout(), *buffer));
      }
    } else if (instruction->IsCrossModuleAllReduce()) {
      CHECK(get_channel_constraints(instruction))
          << "Multi-module layout assignment requires ChannelLayoutConstraints";
      int64 all_reduce_id = instruction->all_reduce_id().value();
      if (!get_channel_constraints(instruction)
               ->IsChannelConstrained(all_reduce_id)) {
        continue;
      }
      // TODO(b/68493863): Change to use SetOperandLayout().
      const Shape& buffer_shape = instruction->operand(0)->shape();
      TF_RET_CHECK(buffer_shape.IsArray());
      Shape new_buffer_shape =
          get_channel_constraints(instruction)
              ->LayoutShapeForChannel(buffer_shape, all_reduce_id);
      TF_RETURN_IF_ERROR(
          constraints->SetInstructionLayout(new_buffer_shape, instruction));
    }
  }

  // Constrain layouts of instructions which call computations which have
  // already been assigned layouts. Instructions which call computations in a
  // parallel element-wise context (eg, map or reduce) do not need layout
  // constraints because they operate on scalars.
  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      // kCall instruction operands and output must match the ComputationLayout
      // of the called computation.
      const ComputationLayout& called_computation_layout =
          FindOrDie(computation_layouts_, instruction->to_apply());
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(
          called_computation_layout.result_layout().shape(), instruction));
      TF_RET_CHECK(instruction->operand_count() ==
                   called_computation_layout.parameter_count());
      for (int64 i = 0; i < instruction->operand_count(); ++i) {
        TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
            called_computation_layout.parameter_layout(i).shape(), instruction,
            i));
      }
    } else if (instruction->opcode() == HloOpcode::kWhile) {
      // Layout of input and output of kWhile instruction must be equal and must
      // match both input and output of body computation. Also, the input of
      // condition computation must match kWhile layout.
      HloComputation* body = instruction->while_body();
      HloComputation* condition = instruction->while_condition();
      const HloInstruction* init = instruction->operand(0);
      ComputationLayout& body_layout = FindOrDie(computation_layouts_, body);
      ComputationLayout& condition_layout =
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

      if (body_layout.result_layout() != body_layout.parameter_layout(0)) {
        VLOG(2) << "Reset %while body parameter layout: body=" << body->name()
                << " while=" << instruction->name()
                << " shape=" << body_layout.result_layout().ToString();
        *body_layout.mutable_parameter_layout(0) = body_layout.result_layout();
      }
      if (condition_layout.parameter_layout(0) !=
          body_layout.parameter_layout(0)) {
        VLOG(2) << "Reset %while condition parameter layout: cond="
                << condition->name() << " while=" << instruction->name()
                << " shape=" << body_layout.parameter_layout(0).ToString();
        *condition_layout.mutable_parameter_layout(0) =
            body_layout.parameter_layout(0);
      }

      // Constrain the output and the operand of the while instruction to match
      // the computations.
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(
          body_layout.result_shape(), instruction));
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
          body_layout.result_shape(), instruction, 0));
    } else if (instruction->opcode() == HloOpcode::kConditional) {
      // The layout of the true and false computations must match, and must
      // be the layout of the kConditional instruction.
      TF_RET_CHECK(instruction->operand_count() == 3);

      HloComputation* true_computation = instruction->true_computation();
      HloComputation* false_computation = instruction->false_computation();
      const HloInstruction* true_operand = instruction->operand(1);
      const HloInstruction* false_operand = instruction->operand(2);

      TF_RET_CHECK(true_computation->num_parameters() == 1);
      TF_RET_CHECK(false_computation->num_parameters() == 1);
      ComputationLayout& true_computation_layout =
          FindOrDie(computation_layouts_, true_computation);
      ComputationLayout& false_computation_layout =
          FindOrDie(computation_layouts_, false_computation);

      DCHECK(ShapeUtil::Compatible(true_operand->shape(),
                                   true_computation_layout.parameter_shape(0)));
      DCHECK(ShapeUtil::Compatible(
          false_operand->shape(), false_computation_layout.parameter_shape(0)));
      if (true_computation_layout.result_layout() !=
          false_computation_layout.result_layout()) {
        // We assign layouts in DFS fashion, so the true and false computations
        // might have negotiated a different layout. But for the conditional
        // instruction POV the layout must match, so we run again on the false
        // computation, this time with proper computation layout.
        VLOG(2) << "Reset %conditional false computation result layout: "
                   "false_computation="
                << false_computation->name()
                << " conditional=" << instruction->name() << " shape="
                << true_computation_layout.result_layout().ToString();
        *false_computation_layout.mutable_result_layout() =
            true_computation_layout.result_layout();
      }
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(
          true_computation_layout.result_shape(), instruction));
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
          true_computation_layout.parameter_shape(0), instruction, 1,
          /*mandatory=*/true));
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
          false_computation_layout.parameter_shape(0), instruction, 2,
          /*mandatory=*/true));
    }
  }
  // Finally set the result layout to match ComputationLayout, if there is one.
  if (computation_layout != nullptr) {
    const ShapeLayout& result_layout = computation_layout->result_layout();
    if (result_layout.LayoutIsSet()) {
      TF_RETURN_IF_ERROR(constraints->SetResultLayout(result_layout.shape()));
    }
  }
  return Status::OK();
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

// Operands of layout-constrained custom calls must match the expected
// constrained layouts.
Status CheckCustomCallLayout(HloInstruction* instruction) {
  if (IsLayoutConstrainedCustomCall(instruction)) {
    const HloCustomCallInstruction* custom_call =
        DynCast<HloCustomCallInstruction>(instruction);
    for (int64 i = 0; i < custom_call->operand_count(); ++i) {
      TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
          custom_call->operand(i)->shape(),
          custom_call->operand_shapes_with_layout()[i]));
    }
  }
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

Status CheckConditionalLayout(
    HloInstruction* instruction,
    const ComputationLayout& true_computation_layout,
    const ComputationLayout& false_computation_layout) {
  HloComputation* true_computation = instruction->true_computation();
  HloComputation* false_computation = instruction->false_computation();
  const HloInstruction* true_operand = instruction->operand(1);
  const HloInstruction* false_operand = instruction->operand(2);

  TF_RET_CHECK(true_computation_layout.result_layout() ==
               false_computation_layout.result_layout());
  TF_RET_CHECK(true_computation_layout.result_layout().MatchesLayoutInShape(
      instruction->shape()));
  TF_RET_CHECK(true_computation_layout.result_layout().MatchesLayoutInShape(
      true_computation->root_instruction()->shape()));
  TF_RET_CHECK(false_computation_layout.result_layout().MatchesLayoutInShape(
      instruction->shape()));
  TF_RET_CHECK(false_computation_layout.result_layout().MatchesLayoutInShape(
      false_computation->root_instruction()->shape()));
  TF_RET_CHECK(true_computation_layout.parameter_layout(0).MatchesLayoutInShape(
      true_operand->shape()));
  TF_RET_CHECK(
      false_computation_layout.parameter_layout(0).MatchesLayoutInShape(
          false_operand->shape()));
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
  if (parameter_layout.LayoutIsSet() &&
      !parameter_layout.MatchesLayoutInShape(parameter->shape())) {
    return InternalError(
        "parameter instruction %s does not match layout of computation "
        "shape: %s",
        parameter->ToString(), parameter_layout.ToString());
  }
  return Status::OK();
}

// The layout of a constant instruction must match the layout of its literal.
Status CheckConstantLayout(HloInstruction* constant) {
  if (!LayoutUtil::LayoutsInShapesEqual(constant->literal().shape(),
                                        constant->shape())) {
    return InternalError(
        "constant instruction %s does not match the layout of its literal %s",
        constant->ToString(),
        ShapeUtil::HumanStringWithLayout(constant->literal().shape()));
  }
  return Status::OK();
}

}  // namespace

StatusOr<HloInstruction*> LayoutAssignment::CreateCopyWithNewLayout(
    const Shape& shape_with_layout, HloInstruction* instruction) {
  TF_RET_CHECK(LayoutUtil::HasLayout(shape_with_layout));
  DCHECK(ShapeUtil::Compatible(shape_with_layout, instruction->shape()))
      << ShapeUtil::HumanString(shape_with_layout) << " "
      << ShapeUtil::HumanString(instruction->shape())
      << " instruction: " << instruction->ToString();

  if (instruction->shape().IsTuple()) {
    // Copy tuple elements which have differing layouts.
    std::vector<HloInstruction*> element_copies;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(instruction->shape());
         ++i) {
      const Shape& target_shape =
          ShapeUtil::GetSubshape(shape_with_layout, {i});
      const Shape& instr_shape =
          ShapeUtil::GetSubshape(instruction->shape(), {i});
      HloInstruction* gte = instruction->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(instr_shape, instruction, i));

      if (ShapeUtil::Equal(target_shape, instr_shape)) {
        // Shapes and layouts are equal, no need to copy.
        element_copies.push_back(gte);
      } else {
        SetupCopiedInstruction(*instruction, gte, {i});
        // Recurse to copy each element.
        TF_ASSIGN_OR_RETURN(HloInstruction * element_copy,
                            CreateCopyWithNewLayout(target_shape, gte));
        element_copies.push_back(element_copy);
      }
    }
    // Gather element copies into a tuple with a new Tuple instruction.
    HloInstruction* tuple_copy = instruction->parent()->AddInstruction(
        HloInstruction::CreateTuple(element_copies));
    SetupCopiedInstruction(*instruction, tuple_copy, {});
    LayoutUtil::ClearLayout(tuple_copy->mutable_shape());
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        shape_with_layout, tuple_copy->mutable_shape()));
    return tuple_copy;
  } else if (instruction->shape().IsArray()) {
    HloInstruction* copy =
        instruction->parent()->AddInstruction(HloInstruction::CreateUnary(
            instruction->shape(), HloOpcode::kCopy, instruction));
    RegisterAddedCopy(copy);
    SetupCopiedInstruction(*instruction, copy, {});
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
Status LayoutAssignment::CopyOperandIfLayoutsDiffer(
    const ShapeLayout& operand_layout, HloInstruction* instruction,
    int64 operand_no) {
  HloInstruction* operand = instruction->mutable_operand(operand_no);
  TF_RET_CHECK(operand_layout.LayoutIsSet());
  TF_RET_CHECK(LayoutUtil::HasLayout(operand->shape()));

  if (ShapeUtil::Equal(operand_layout.shape(), operand->shape())) {
    VLOG(5) << "Operand " << operand->ToString() << " layout matches in "
            << instruction->ToString();
    // Operand layout already matches our constraint. Nothing to do.
    return Status::OK();
  }
  VLOG(4) << "Operand " << operand->ToString() << " layout does not match "
          << operand_layout.ToString() << " in " << instruction->ToString();

  TF_ASSIGN_OR_RETURN(HloInstruction * operand_copy,
                      CreateCopyWithNewLayout(operand_layout.shape(), operand));

  VLOG(4) << "New copy of " << operand->ToString() << " is "
          << operand_copy->ToString();
  return instruction->ReplaceOperandWith(operand_no, operand_copy);
}

void LayoutAssignment::SetupCopiedInstruction(const HloInstruction& instruction,
                                              HloInstruction* copy,
                                              const ShapeIndex& index) {
  if (instruction.has_sharding()) {
    // If the index is empty, we want to copy the whole sharding, in case the
    // sharding is a tuple sharding.
    HloSharding sharding =
        !index.empty() && instruction.sharding().IsTuple()
            ? instruction.sharding().GetSubSharding(instruction.shape(), index)
            : instruction.sharding();
    // We propagate the sharding to the copied instruction only if it is a
    // special sharding, like tiled ones.
    // Otherwise it is preferable to leave the new instruction without device,
    // and let the automatic device placer to choose the best location.
    auto device = sharding.UniqueDevice();
    if (!device || HloSharding::IsReservedDevice(*device)) {
      copy->set_sharding(sharding);
    }
  }
  copy->set_metadata(instruction.metadata());
}

Status LayoutAssignment::CheckLayouts(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto points_to_analysis,
                      TuplePointsToAnalysis::Run(module));
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->instructions()) {
      // Verify every instruction has a layout and the layout is valid for the
      // shape.
      TF_RET_CHECK(LayoutUtil::HasLayout(instruction->shape()));
      TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

      // Use points-to analysis to verify that every subshape element in the
      // output of the instruction matches the layout of the logical buffer
      // which could be the source of the subshape value.
      const PointsToSet& points_to_set =
          points_to_analysis->GetPointsToSet(instruction);
      TF_RETURN_IF_ERROR(points_to_set.ForEachElementWithStatus(
          [&instruction](ShapeIndex index,
                         const PointsToSet::BufferList& buffers) -> Status {
            if (ShapeUtil::IsLeafIndex(instruction->shape(), index)) {
              const Shape& instruction_subshape =
                  ShapeUtil::GetSubshape(instruction->shape(), index);
              for (const LogicalBuffer* buffer : buffers) {
                if (!ShapeUtil::Equal(instruction_subshape, buffer->shape())) {
                  return InternalError(
                      "Layout of instruction %s at index {%s} does not match "
                      "source LogicalBuffer %s: %s vs %s",
                      instruction->name(), absl::StrJoin(index, ","),
                      buffer->ToString(),
                      ShapeUtil::HumanStringWithLayout(instruction_subshape),
                      ShapeUtil::HumanStringWithLayout(buffer->shape()));
                }
              }
            }
            return Status::OK();
          }));

      // Verify instructions that have special layout constraints.
      switch (instruction->opcode()) {
        case HloOpcode::kCall:
          TF_RETURN_IF_ERROR(CheckCallLayout(
              instruction,
              FindOrDie(computation_layouts_, instruction->to_apply())));
          break;
        case HloOpcode::kCustomCall:
          TF_RETURN_IF_ERROR(CheckCustomCallLayout(instruction));
          break;
        case HloOpcode::kFusion:
          TF_RETURN_IF_ERROR(CheckFusionLayout(instruction));
          break;
        case HloOpcode::kParameter:
          TF_RETURN_IF_ERROR(CheckParameterLayout(
              instruction,
              FindOrDie(computation_layouts_, instruction->parent())));
          break;
        case HloOpcode::kConstant:
          TF_RETURN_IF_ERROR(CheckConstantLayout(instruction));
          break;
        case HloOpcode::kWhile:
          TF_RETURN_IF_ERROR(CheckWhileLayout(
              instruction,
              FindOrDie(computation_layouts_, instruction->while_condition()),
              FindOrDie(computation_layouts_, instruction->while_body())));
          break;
        case HloOpcode::kConditional:
          TF_RETURN_IF_ERROR(CheckConditionalLayout(
              instruction,
              FindOrDie(computation_layouts_, instruction->true_computation()),
              FindOrDie(computation_layouts_,
                        instruction->false_computation())));
          break;
        default:
          break;
      }
    }
  }
  // Finally verify the result layout, if set, matches the layout of the entry
  // computation root.
  const ShapeLayout& result_layout =
      FindOrDie(computation_layouts_, module->entry_computation())
          .result_layout();
  if (result_layout.LayoutIsSet()) {
    TF_RET_CHECK(
        ShapeUtil::Equal(module->result_shape(), result_layout.shape()));
  }
  return Status::OK();
}

LayoutAssignment::LayoutAssignment(
    ComputationLayout* entry_computation_layout,
    std::function<bool(const HloInstruction*)>
        instruction_can_change_layout_func,
    ChannelLayoutConstraints* channel_constraints)
    : entry_computation_layout_(entry_computation_layout),

      saved_entry_computation_layout_(*entry_computation_layout),
      channel_layout_constraints_(channel_constraints),
      instruction_can_change_layout_func_(
          std::move(instruction_can_change_layout_func)) {
  if (channel_layout_constraints_ != nullptr) {
    // Save a copy of the input ChannelLayoutConstraints so that we can reset it
    // if we have to undo previous operations (ClearPreviousPassSideEffects()).
    channel_constraints_ = *channel_layout_constraints_;
  }
  VLOG(1) << "Entry computation layout given to layout assignment: "
          << entry_computation_layout_->ToString();
}

std::unique_ptr<Layout> LayoutAssignment::ChooseOperandLayoutFromOutputLayout(
    const Layout& output_layout, const HloInstruction* instruction,
    int64 operand_no) {
  const HloInstruction* operand = instruction->operand(operand_no);
  CHECK(instruction->shape().IsArray());
  CHECK(operand->shape().IsArray());
  if (!ShapeUtil::IsScalar(operand->shape()) &&
      operand->shape().rank() == instruction->shape().rank() &&
      !instruction_can_change_layout_func_(instruction)) {
    // Propagate the result layout to the operand layout if the instruction
    // requires the same layout out for the result and the operand.
    //
    // For elementwise operations, using the same layout for the operands and
    // the result also has the following benefits:
    // 1) the elementwise operation can reuse its operand's buffer, and
    // 2) the input and output elements can reuse the same linear index.
    return absl::make_unique<Layout>(output_layout);
  }

  if (instruction->opcode() == HloOpcode::kReshape) {
    // Prefer the operand layout that makes the reshape an bitcast. If any
    // dimension bound is 1 in the operand shape, there may be several such
    // layouts. So if 'output_layout' is the default layout, try if the
    // reshape is a bitcast when using the same layout. This may avoid copy
    // operations. For similar reasons, if the operand and output have the same
    // rank, try to match the operand's layout to the output.
    if (ShapeUtil::TrueRank(operand->shape()) == 1 &&
        instruction->shape().rank() == 1) {
      // Don't assign a layout in case of R1 -> effective R1 reshape.
      return nullptr;
    }
    const Shape& output_shape = instruction->shape();
    Shape output_shape_with_layout = ShapeUtil::MakeShapeWithLayout(
        output_shape.element_type(), AsInt64Slice(output_shape.dimensions()),
        LayoutUtil::MinorToMajor(output_layout));
    Shape operand_shape = operand->shape();
    *operand_shape.mutable_layout() =
        LayoutUtil::GetDefaultLayoutForShape(operand_shape);
    if (ShapeUtil::ReshapeIsBitcast(operand_shape, output_shape_with_layout)) {
      return absl::make_unique<Layout>(operand_shape.layout());
    }
    if (operand_shape.rank() == output_shape.rank()) {
      *operand_shape.mutable_layout() = output_layout;
      if (ShapeUtil::ReshapeIsBitcast(operand_shape,
                                      output_shape_with_layout)) {
        return absl::make_unique<Layout>(output_layout);
      }
    }
    auto aligned_operand_shape =
        ShapeUtil::AlignLayouts(output_shape_with_layout, operand_shape);
    if (aligned_operand_shape) {
      auto operand_layout = aligned_operand_shape.value().layout();
      TF_CHECK_OK(
          LayoutUtil::ValidateLayoutForShape(operand_layout, operand_shape));
      return absl::make_unique<Layout>(operand_layout);
    }
  }

  if (instruction->opcode() == HloOpcode::kTranspose) {
    // Pick the operand layout that makes the transpose a bitcast.
    int64 rank = instruction->shape().rank();
    std::vector<int64> new_minor_to_major(rank);
    for (int64 i = 0; i < rank; ++i) {
      int64 output_dim = LayoutUtil::Minor(output_layout, i);
      int64 operand_dim = instruction->dimensions(output_dim);
      new_minor_to_major[i] = operand_dim;
    }
    Layout operand_layout = LayoutUtil::MakeLayout(new_minor_to_major);
    TF_CHECK_OK(
        LayoutUtil::ValidateLayoutForShape(operand_layout, operand->shape()));
    return absl::make_unique<Layout>(operand_layout);
  }

  return nullptr;
}

std::unique_ptr<Layout> LayoutAssignment::ChooseOutputLayoutFromOperandLayout(
    const Layout& operand_layout, const HloInstruction* user,
    int64 operand_no) {
  const HloInstruction* operand = user->operand(operand_no);

  CHECK(user->shape().IsArray() && operand->shape().IsArray());

  if (!ShapeUtil::IsScalar(operand->shape()) &&
      operand->shape().rank() == user->shape().rank() &&
      !instruction_can_change_layout_func_(user)) {
    // Assign users the same layout as the operand.
    return absl::make_unique<Layout>(operand_layout);
  }

  if (user->opcode() == HloOpcode::kReshape) {
    // Prefer the user layout that makes the reshape an bitcast. If any
    // dimension bound is 1 in the user shape, there may be several such
    // layouts. So if 'operand_layout' is the default layout, try if the
    // reshape is a bitcast when using the same layout. This may avoid copy
    // operations. For similar reasons, if the operand and output have the same
    // rank, try to match the outputs's layout to the operand.
    if (operand->shape().rank() == 1 &&
        ShapeUtil::TrueRank(user->shape()) == 1) {
      // Don't assign a layout in case of R1 -> effective R1 reshape.
      return nullptr;
    }
    Shape operand_shape_with_layout = ShapeUtil::MakeShapeWithLayout(
        operand->shape().element_type(),
        AsInt64Slice(operand->shape().dimensions()),
        LayoutUtil::MinorToMajor(operand_layout));
    Shape output_shape = user->shape();
    *output_shape.mutable_layout() =
        LayoutUtil::GetDefaultLayoutForShape(output_shape);
    if (ShapeUtil::ReshapeIsBitcast(output_shape, operand_shape_with_layout)) {
      return absl::make_unique<Layout>(output_shape.layout());
    }
    if (operand->shape().rank() == output_shape.rank()) {
      *output_shape.mutable_layout() = operand_layout;
      if (ShapeUtil::ReshapeIsBitcast(output_shape,
                                      operand_shape_with_layout)) {
        return absl::make_unique<Layout>(operand_layout);
      }
    }
    auto aligned_user_shape =
        ShapeUtil::AlignLayouts(operand_shape_with_layout, output_shape);
    if (aligned_user_shape) {
      auto user_layout = aligned_user_shape.value().layout();
      TF_CHECK_OK(
          LayoutUtil::ValidateLayoutForShape(user_layout, output_shape));
      return absl::make_unique<Layout>(user_layout);
    }
  }

  if (user->opcode() == HloOpcode::kTranspose) {
    // Pick the user layout that makes the transpose a bitcast.
    int64 rank = user->shape().rank();
    std::vector<int64> new_minor_to_major(rank);
    auto inverse_dimensions = InversePermutation(user->dimensions());
    for (int64 i = 0; i < rank; ++i) {
      int64 operand_dim = LayoutUtil::Minor(operand_layout, i);
      int64 user_dim = inverse_dimensions[operand_dim];
      new_minor_to_major[i] = user_dim;
    }
    Layout user_layout = LayoutUtil::MakeLayout(new_minor_to_major);
    TF_CHECK_OK(LayoutUtil::ValidateLayoutForShape(user_layout, user->shape()));
    return absl::make_unique<Layout>(user_layout);
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
      if (constraint->dfs()) {
        worklist.push_front(constraint);
      } else {
        worklist.push_back(constraint);
      }
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
    if (!buffer_alias.instruction()->shape().IsArray()) {
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
      [&shape_layout, constraints](
          const ShapeIndex& index,
          const PointsToSet::BufferList& buffers) -> Status {
        if (ShapeUtil::IsLeafIndex(shape_layout.shape(), index)) {
          for (const LogicalBuffer* buffer : buffers) {
            if (constraints->BufferLayout(*buffer) == nullptr &&
                buffer->shape().IsArray()) {
              TF_RETURN_IF_ERROR(constraints->SetBufferLayout(
                  ShapeUtil::GetSubshape(shape_layout.shape(), index).layout(),
                  *buffer, /*mandatory=*/true));
            }
          }
        }
        return Status::OK();
      });
}

namespace {
// A transpose or a reshape that only changes trivial dimensions have meaningful
// layouts that are valuable to propagate in a depthfirst manner to avoid
// unassigned layouts in the graph.
bool InstructionShouldPropagateDepthFirst(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kReshape:
      return std::get<0>(hlo.ReshapeMerelyInsertsOrDeletes1SizedDimensions());
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}

}  // namespace

Status LayoutAssignment::PropagateOperandConstraint(
    const OperandLayoutConstraint& operand_constraint,
    LayoutConstraints* constraints) {
  // Try to set the layout of the logical buffers in the given operand to match
  // the constrained layout. This avoids copies.
  TF_RETURN_IF_ERROR(
      PropagateUseConstraintToDefs(operand_constraint.shape_layout(),
                                   operand_constraint.operand(), constraints));

  // For array-shaped operands and user instructions try to pick a minimum cost
  // layout. For example, if the operand of an elementwise instruction is
  // constrained to a certain layout we want the output of the instruction to
  // have the same layout.
  //
  // If the user is not array-shaped, we still want to propagate the layout
  // to siblings if the instruction can't change layout. This is to represent
  // the information that non-layout-changing instructions should have the same
  // layout for the operands with the same ranks.
  const HloInstruction* operand = operand_constraint.operand();
  const HloInstruction* user = operand_constraint.instruction();
  if (!operand->shape().IsArray()) {
    return Status::OK();
  }
  if (instruction_can_change_layout_func_(user) && !user->shape().IsArray()) {
    return Status::OK();
  }

  // Only try to choose a low cost layout if the instruction 'user' defines its
  // output (ie, doesn't forward a buffer from elsewhere).
  if (constraints->OperandBufferForwarded(user,
                                          operand_constraint.operand_no())) {
    return Status::OK();
  }

  int64 operand_rank = operand->shape().rank();
  if (operand_rank <= 1) {
    return Status::OK();
  }

  // Propagate layouts between operands of the same instruction. This is a
  // constraint on non-layout-changing instructions.
  if (!instruction_can_change_layout_func_(user)) {
    // Make sure all siblings have the same layout as the operand.
    for (int64 operand_no = 0; operand_no < user->operand_count();
         ++operand_no) {
      if (user->operand(operand_no) == operand) {
        continue;
      }
      const HloInstruction* sibling = user->operand(operand_no);
      const int64 sibling_rank = sibling->shape().rank();
      if (sibling_rank <= 1) {
        continue;
      }
      if (operand_rank != sibling_rank) {
        continue;
      }
      const OperandLayoutConstraint* constraint =
          constraints->GetOperandLayoutConstraint(user, operand_no);
      if (constraint != nullptr) {
        // Due to the DFS of the propagation we can end up here when operand_no
        // has a layout set that hasn't been propagated yet (is still on the
        // stack of layouts to propagate).
        // We can continue here and leave the operands with different layouts,
        // as we will either:
        // - overwrite the current operand when the DFS gets back to propagating
        //   operand(operand_no) to its siblings
        // - overwrite operand(operand_no)'s layout with a mandatory layout if
        //   we continue to propagate our layout to the result, and then
        //   backwards into all operands (if the result is an array of rank > 1)
        continue;
      }
      TF_RETURN_IF_ERROR(constraints->SetArrayOperandLayout(
          operand_constraint.shape_layout().layout(), user, operand_no,
          /*mandatory=*/false));
    }
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        user->shape(),
        [&](const Shape& subshape, const ShapeIndex& shape_index) {
          if (subshape.IsTuple()) {
            return Status::OK();
          }
          if (subshape.rank() <= 1) {
            return Status::OK();
          }

          // Assign the right layout to input fusion of higher rank reduce
          // operations.
          if (subshape.rank() != operand->shape().rank()) {
            return Status::OK();
          }
          // TODO(b/67641796): Are there cases except fusion that use this code
          // path?
          TF_ASSIGN_OR_RETURN(
              const LogicalBuffer* buffer,
              constraints->points_to_analysis().GetBufferDefinedAt(
                  user, shape_index));
          // Make sure the output has the same layout as the operand.
          const BufferLayoutConstraint* constraint =
              constraints->GetBufferLayoutConstraint(*buffer);
          // If we already have a constraint for the buffer it was assigned but
          // hasn't propagated yet. This can happen with diamond-shaped graphs
          // where one path is first evaluated in depth-first order (we're here)
          // and the other path is propagated later. We don't set the layout
          // here as it will always be overwritten later.
          if (constraint == nullptr) {
            TF_RETURN_IF_ERROR(constraints->SetBufferLayout(
                operand_constraint.shape_layout().layout(), *buffer,
                /*mandatory=*/false));
          }
          return Status::OK();
        }));
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      user->shape(), [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (subshape.IsTuple()) {
          return Status::OK();
        }
        if (subshape.rank() <= 1) {
          return Status::OK();
        }
        TF_ASSIGN_OR_RETURN(
            const LogicalBuffer* buffer,
            constraints->points_to_analysis().GetBufferDefinedAt(user,
                                                                 shape_index));
        if (constraints->BufferLayout(*buffer) == nullptr ||
            !constraints->GetBufferLayoutConstraint(*buffer)->mandatory()) {
          std::unique_ptr<Layout> layout = ChooseOutputLayoutFromOperandLayout(
              operand_constraint.shape_layout().layout(), user,
              operand_constraint.operand_no());
          if (layout != nullptr) {
            TF_RETURN_IF_ERROR(constraints->SetBufferLayout(
                *layout, *buffer,
                /*mandatory=*/user->opcode() == HloOpcode::kReduce,
                /*dfs=*/InstructionShouldPropagateDepthFirst(*user)));
          }
        }
        return Status::OK();
      }));
  return Status::OK();
}

Status LayoutAssignment::PropagateBufferConstraintToOperands(
    const BufferLayoutConstraint& buffer_constraint,
    LayoutConstraints* constraints) {
  VLOG(5) << "PropagateBufferConstraintToOperands: "
          << buffer_constraint.ToString();
  const LogicalBuffer& buffer = buffer_constraint.buffer();

  const HloInstruction* instruction = buffer.instruction();
  if (IsAtMostRank1(instruction->shape())) {
    return Status::OK();
  }

  for (int64 operand_no = 0; operand_no < instruction->operand_count();
       ++operand_no) {
    const HloInstruction* operand = instruction->operand(operand_no);
    if (IsAtMostRank1(operand->shape())) {
      continue;
    }
    if (!instruction_can_change_layout_func_(instruction)) {
      // Copy the layout to the operand.
      if (buffer.IsArray() && operand->shape().IsArray() &&
          operand->shape().rank() ==
              LayoutUtil::MinorToMajor(buffer_constraint.layout()).size()) {
        TF_RETURN_IF_ERROR(constraints->SetArrayOperandLayout(
            buffer_constraint.layout(), instruction, operand_no,
            /*mandatory=*/true));
      }
    } else {
      if (!buffer.IsTopLevel() ||
          !instruction->operand(operand_no)->shape().IsArray()) {
        continue;  // Don't touch buffers that are internal to a tuple.
      }
      VLOG(6) << "Propagating constraint to operand " << operand_no << " of "
              << instruction->ToShortString();
      // Assign a layout if there is no constraint already.
      const OperandLayoutConstraint* constraint =
          constraints->GetOperandLayoutConstraint(instruction, operand_no);
      if (constraint == nullptr || !constraint->mandatory()) {
        std::unique_ptr<Layout> operand_layout =
            ChooseOperandLayoutFromOutputLayout(buffer_constraint.layout(),
                                                instruction, operand_no);
        if (operand_layout != nullptr) {
          TF_RETURN_IF_ERROR(constraints->SetArrayOperandLayout(
              *operand_layout, instruction, operand_no, /*mandatory=*/false,
              /*dfs=*/InstructionShouldPropagateDepthFirst(*instruction)));
        }
      } else {
        VLOG(6) << "Operand already has a constraint "
                << constraint->ToString();
      }
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
  TF_RETURN_IF_ERROR(
      PropagateBufferConstraintToUses(buffer_constraint, constraints));
  return PropagateBufferConstraintToOperands(buffer_constraint, constraints);
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
    const ResultLayoutConstraint& layout_constraint,
    LayoutConstraints* constraints) {
  // Propagate the use constraint of the root instruction up to the logical
  // buffers which make up the result.
  return PropagateUseConstraintToDefs(
      layout_constraint.shape_layout(),
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
  TF_RET_CHECK(subshape.IsArray());
  TF_RET_CHECK(!subshape.has_layout());

  // The instruction should not define the buffer at this index.
  TF_RET_CHECK(
      !points_to_analysis.InstructionDefinesBufferAtIndex(instruction, index))
      << instruction->ToString();

  const auto& source_buffers =
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
                           source_buffer->ToString());
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
          absl::StrJoin(index, ","), instruction->name(),
          source_buffers[0]->ToString(), source_buffer->ToString());
    }
  }

  return *first_buffer_layout;
}

// For fusion instructions, set the layout of each fused parameter instruction
// to match the layout of its corresponding fusion instruction operand. Also,
// set the layout of the fused root to match the layout of the fusion
// instruction itself.
Status SetFusionLayouts(HloInstruction* fusion) {
  TF_RET_CHECK(fusion->opcode() == HloOpcode::kFusion);
  for (auto* fused_instruction :
       fusion->fused_instructions_computation()->MakeInstructionPostOrder()) {
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      const HloInstruction* fusion_operand =
          fusion->operand(fused_instruction->parameter_number());
      DCHECK(ShapeUtil::Compatible(fusion_operand->shape(),
                                   fused_instruction->shape()));
      TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
          fusion_operand->shape(), fused_instruction->mutable_shape()));
    } else if (fused_instruction == fusion->fused_expression_root()) {
      // The layout of the root of the fused expression must match the fusion
      // instruction layout.
      DCHECK(
          ShapeUtil::Compatible(fusion->shape(), fused_instruction->shape()));
      TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
          fusion->shape(), fused_instruction->mutable_shape()));
    } else if (fused_instruction->opcode() == HloOpcode::kGetTupleElement) {
      // A GTE inherits its layout from its operand (which should ultimately be
      // a parameter).
      TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
          fused_instruction->operand(0)->shape().tuple_shapes(
              fused_instruction->tuple_index()),
          fused_instruction->mutable_shape()));
    } else if (fused_instruction->opcode() == HloOpcode::kConstant) {
      // Give constants the layout of their literal.
      TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
          fused_instruction->literal().shape(),
          fused_instruction->mutable_shape()));
    } else if (fused_instruction->opcode() == HloOpcode::kInfeed) {
      // Nop; leave the infeed layout alone.
    } else if (fusion->fusion_kind() != HloInstruction::FusionKind::kCustom) {
      // Other instructions don't have layouts inside of fusion nodes.
      // But do not clear layouts for other instructions in custom fusion nodes.
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

    // Set the layouts of the array shapes this instruction defines as indicated
    // by the respective BufferLayoutConstraints. Any array shapes in the output
    // of the instruction which are not defined by the instruction (eg, array
    // elements in a Tuple instruction) will be assigned below via inference.
    for (const LogicalBuffer* buffer :
         constraints.points_to_analysis().GetBuffersDefinedByInstruction(
             instruction)) {
      if (!buffer->shape().IsArray()) {
        continue;
      }

      TF_RET_CHECK(buffer->instruction() == instruction);
      const Layout* buffer_layout = constraints.BufferLayout(*buffer);
      TF_RET_CHECK(buffer_layout != nullptr);

      if (instruction->opcode() == HloOpcode::kConstant) {
        // For constants, we also need to change the layout of the internal
        // literal.
        instruction->RelayoutConstant(*buffer_layout, buffer->index());
      } else {
        Shape* buffer_subshape = ShapeUtil::GetMutableSubshape(
            instruction->mutable_shape(), buffer->index());
        *buffer_subshape->mutable_layout() = *buffer_layout;
      }
    }

    // Any remaining layouts in the output of the instruction must be
    // inferrable using points-to analysis.
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachMutableSubshapeWithStatus(
        instruction->mutable_shape(),
        [instruction, &constraints](Shape* subshape, const ShapeIndex& index) {
          if (subshape->has_layout() || !subshape->IsArray()) {
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

    // Shape must be valid.
    TF_RETURN_IF_ERROR(
        ShapeUtil::ValidateShapeWithOptionalLayout(instruction->shape()));

    // Verify all layouts in the shape have been set.
    TF_RET_CHECK(LayoutUtil::HasLayout(instruction->shape()));
  }
  return Status::OK();
}

Status LayoutAssignment::CalculateComputationLayout(
    HloComputation* computation) {
  ComputationLayout computation_layout(computation->ComputeProgramShape(),
                                       /*ignore_layouts=*/false);
  InsertOrDie(&computation_layouts_, computation, computation_layout);
  VLOG(2) << "  Calculated ComputationLayout = "
          << computation_layout.ToString();
  return Status::OK();
}

Status LayoutAssignment::ClearComputationLayouts(HloComputation* computation) {
  // Clear existing layouts of the instructions.  All layouts must be assigned
  // by the LayoutAssignment pass, except for those on parameters, the
  // computation result, and a couple special cases. The former two are
  // specified in computation_layout.  Clearing the layouts here avoids hiding
  // potential bugs in the layout assignment pass that may accidentally use the
  // existing layout.
  for (HloInstruction* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kBitcast) {
      // bitcasts are inherently layout sensitive and so a bitcast instruction
      // present in the IR before layout assignment is a bug.
      return InternalError(
          "Unexpected bitcast operation seen during layout assignment: %s.",
          instruction->ToString());
    }
    // Some instructions carry mandatory layouts in their shape.
    if (instruction->opcode() != HloOpcode::kInfeed &&
        !IsLayoutConstrainedCustomCall(instruction)) {
      LayoutUtil::ClearLayout(instruction->mutable_shape());
    }
  }
  return Status::OK();
}

Status LayoutAssignment::RunOnComputation(
    ComputationLayout* computation_layout,
    const TuplePointsToAnalysis& points_to_analysis,
    HloComputation* computation,
    ChannelLayoutConstraints* channel_constraints) {
  VLOG(2) << "LayoutAssignment::RunOnComputation(" << computation->name()
          << ")";

  // Must be run before clearing layouts.
  TF_RETURN_IF_ERROR(BuildHostChannelConstraints(computation));

  TF_RETURN_IF_ERROR(ClearComputationLayouts(computation));
  if (computation_layout != nullptr) {
    auto it = computation_layouts_.find(computation);
    if (it == computation_layouts_.end()) {
      VLOG(2) << "  New ComputationLayout = " << computation_layout->ToString();
      computation_layouts_.emplace(computation, *computation_layout);
    } else {
      TF_RET_CHECK(computation_layout == &it->second ||
                   computation_layout == entry_computation_layout_);
      VLOG(2) << "  Existing ComputationLayout = "
              << computation_layout->ToString();
    }
  } else {
    VLOG(2) << "  No ComputationLayout specified (will be calculated)";
  }

  // Construct LayoutConstraints with all layout constraints of the computation.
  LayoutConstraints constraints(points_to_analysis, computation);

  // Add constraints required for correctness on all backends (eg, entry
  // parameter layout constraints).
  TF_RETURN_IF_ERROR(AddMandatoryConstraints(
      computation_layout, channel_constraints, computation, &constraints));

  // Add any backend-specific constraints.
  TF_RETURN_IF_ERROR(AddBackendConstraints(&constraints));

  // Propagates layouts from mandatory and backend constraints.
  TF_RETURN_IF_ERROR(PropagateConstraints(&constraints));

  // Prior to applying default layouts, we take note of all HLO instructions
  // which lack a layout constraint.
  for (LogicalBuffer::Id buffer_id : constraints.unconstrained_buffer_ids()) {
    unconstrained_layout_instructions_.insert(
        points_to_analysis.GetBuffer(buffer_id).instruction());
  }

  // While any unconstrained buffers remain, pick an arbitrary buffer, give it a
  // layout and propagate the change.
  while (!constraints.unconstrained_buffer_ids().empty()) {
    int unconstrained_count = constraints.unconstrained_buffer_ids().size();

    // Arbitrarily pick the first unconstrained buffer and give it the default
    // layout (or the literal layout, in case of constants). By construction
    // unconstrained_buffers() has a stable sort based on LogicalBuffer::Id.
    const LogicalBuffer& buffer = points_to_analysis.GetBuffer(
        *constraints.unconstrained_buffer_ids().begin());
    const HloInstruction* instruction = buffer.instruction();
    Layout new_layout =
        instruction->opcode() == HloOpcode::kConstant
            ? ShapeUtil::GetSubshape(instruction->literal().shape(),
                                     buffer.index())
                  .layout()
            : LayoutUtil::GetDefaultLayoutForShape(buffer.shape());
    TF_RETURN_IF_ERROR(constraints.SetBufferLayout(new_layout, buffer,
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
  TF_RETURN_IF_ERROR(AssignLayouts(constraints, computation));

  // If the computation layout wasn't specified, now it is the time to compute
  // it according to the parameters and root instruction layouts.
  // This allows the first pass through this API to record the best flowing
  // layout to parameters and root instruction.
  if (computation_layout == nullptr) {
    TF_RETURN_IF_ERROR(CalculateComputationLayout(computation));
  }

  // Record the layouts assigned for any communication ops in
  // channel_constraints so that they are constrained for future modules.
  if (channel_constraints != nullptr) {
    TF_RETURN_IF_ERROR(
        ConstrainChannelLayouts(computation, channel_constraints));
  }

  // Copy the root instruction's result if its layout does not match the result
  // layout constraint.
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

Status LayoutAssignment::ConstrainChannelLayouts(
    HloComputation* computation,
    ChannelLayoutConstraints* channel_constraints) {
  auto get_channel_constraints = [&](const HloInstruction* instruction) {
    return IsHostSendRecv(instruction) ? &host_channel_constraints_
                                       : channel_constraints;
  };
  // We go through the kRecvDone before. These must either impose their layout,
  // or find a matching one already existing (ConstrainChannel() returns
  // nullptr).
  for (HloInstruction* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kRecvDone) {
      const Layout* layout =
          get_channel_constraints(instruction)
              ->ConstrainChannel(
                  instruction->channel_id(),
                  ShapeUtil::GetSubshape(instruction->shape(), {0}).layout());
      TF_RET_CHECK(layout == nullptr)
          << instruction->ToString()
          << " cannot constrain layout as it was set to "
          << LayoutUtil::HumanString(*layout);
    }
  }
  // After that we go through the kSend. These are likely going to have a kCopy
  // as operand (otherwise we add it), so in case the constrained layout does
  // not match, we can change the kCopy layout (and the kSend one as well).
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kSend) {
      HloInstruction* operand = instruction->mutable_operand(0);
      const Layout* layout = get_channel_constraints(instruction)
                                 ->ConstrainChannel(instruction->channel_id(),
                                                    operand->shape().layout());
      if (layout != nullptr) {
        // We found an already constrained layout which does not match the one
        // the kSend wants to impose. Either add a new kCopy, or use the
        // existing one to marshal the correct shape.
        Shape shape = operand->shape();
        *shape.mutable_layout() = *layout;
        if (operand->opcode() != HloOpcode::kCopy) {
          HloInstruction* copy = operand->parent()->AddInstruction(
              HloInstruction::CreateUnary(shape, HloOpcode::kCopy, operand));
          RegisterAddedCopy(copy);
          SetupCopiedInstruction(*operand, copy, {});
          TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(0, copy));
          operand = copy;
        } else {
          *operand->mutable_shape() = shape;
        }
        Shape* send_shape =
            ShapeUtil::GetMutableSubshape(instruction->mutable_shape(), {0});
        *send_shape = shape;
      }
    } else if (instruction->IsCrossModuleAllReduce()) {
      const Layout* layout =
          get_channel_constraints(instruction)
              ->ConstrainChannel(instruction->all_reduce_id().value(),
                                 instruction->shape().layout());
      if (layout != nullptr) {
        // We found an already constrained layout which does not match the one
        // the channel wants to impose. Either add a new kCopy, or use the
        // existing one to marshal the correct shape.
        HloInstruction* operand = instruction->mutable_operand(0);
        Shape shape = operand->shape();
        *shape.mutable_layout() = *layout;
        if (operand->opcode() != HloOpcode::kCopy) {
          HloInstruction* copy = operand->parent()->AddInstruction(
              HloInstruction::CreateUnary(shape, HloOpcode::kCopy, operand));
          RegisterAddedCopy(copy);
          SetupCopiedInstruction(*operand, copy, {});
          TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(0, copy));
          operand = copy;
        } else {
          *operand->mutable_shape() = shape;
        }
        *instruction->mutable_shape() = shape;
      }
    }
  }
  return Status::OK();
}

Status LayoutAssignment::PropagateComputationLayouts(
    HloComputation* computation, ComputationLayout* computation_layout) {
  ComputationLayout computed_computation_layout(
      computation->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  for (int64 i = 0; i < computed_computation_layout.parameter_count(); ++i) {
    ShapeLayout* param_layout = computation_layout->mutable_parameter_layout(i);
    if (!param_layout->LayoutIsSet()) {
      VLOG(4) << "Assigning layout to parameter " << i << " of computation "
              << computation->name() << ": "
              << computed_computation_layout.parameter_layout(i).ToString();
      *param_layout = computed_computation_layout.parameter_layout(i);
    } else {
      TF_RET_CHECK(computed_computation_layout.parameter_layout(i) ==
                   *param_layout);
    }
  }
  ShapeLayout* result_layout = computation_layout->mutable_result_layout();
  if (!result_layout->LayoutIsSet()) {
    VLOG(4) << "Assigning result layout of computation " << computation->name()
            << ": " << computed_computation_layout.result_layout().ToString();
    *result_layout = computed_computation_layout.result_layout();
  } else {
    TF_RET_CHECK(computed_computation_layout.result_layout() == *result_layout);
  }
  return Status::OK();
}

StatusOr<bool> LayoutAssignment::Run(HloModule* module) {
  VLOG(2) << "Running layout assignment on module " << module->name();
  XLA_VLOG_LINES(3, module->ToString());
  if (VLOG_IS_ON(10)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "before layout assignment",
                                module->config().debug_options());
  }
  TF_RETURN_IF_ERROR(Init());

  // Verify computation layout is sane.
  const HloComputation* entry = module->entry_computation();
  TF_RET_CHECK(entry_computation_layout_->parameter_count() ==
               entry->num_parameters());
  for (int64 i = 0; i < entry->num_parameters(); ++i) {
    TF_RET_CHECK(
        ShapeUtil::Compatible(entry_computation_layout_->parameter_shape(i),
                              entry->parameter_instruction(i)->shape()));
  }
  TF_RET_CHECK(ShapeUtil::Compatible(entry_computation_layout_->result_shape(),
                                     entry->root_instruction()->shape()));

  // We do two passes. The first one we pass a nullptr ComputationLayout to
  // the RunOnComputation() calls (for non entry computations), and we register
  // the ComputationLayout which are naturally flowing in DFS fashion to the
  // parameters and root instruction.
  // Walking in DFS mode though, means that we can end up with incorrect layouts
  // when seen from an outer instruction, which has across-computation
  // constraints to impose.
  // For example, the kWhile instruction needs to enforce the same layouts for
  // the parameters and root of the body, as well as the condition parameters.
  // Similarly, the kConditional instruction needs to enforce the same layouts
  // for the root of the true and false computations.
  // So in the first pass, while allowing the layouts to flow to parameters and
  // root, we also fix up the eventually inconsistent ComputationLayout, which
  // will be then made mandatory by the second pass.
  for (int64 i = 0; i < 2; ++i) {
    VLOG(5) << "Running " << (i == 0 ? "un" : "") << "constrained pass";
    TF_RETURN_IF_ERROR(ClearPreviousPassSideEffects(module));
    TF_ASSIGN_OR_RETURN(auto points_to_analysis,
                        TuplePointsToAnalysis::Run(module));
    for (auto* computation : module->MakeComputationPostOrder()) {
      if (computation->IsFusionComputation()) {
        continue;
      }
      if (computation == module->entry_computation()) {
        TF_RETURN_IF_ERROR(RunOnComputation(
            entry_computation_layout_, *points_to_analysis,
            module->entry_computation(), channel_layout_constraints_));
      } else {
        ComputationLayout* computation_layout =
            (i == 0) ? nullptr : &FindOrDie(computation_layouts_, computation);
        TF_RETURN_IF_ERROR(RunOnComputation(computation_layout,
                                            *points_to_analysis, computation,
                                            channel_layout_constraints_));
      }
    }
  }
  TF_RETURN_IF_ERROR(PropagateComputationLayouts(module->entry_computation(),
                                                 entry_computation_layout_));
  TF_RETURN_IF_ERROR(CheckLayouts(module));

  VLOG(3) << "After layout assignment:";
  XLA_VLOG_LINES(3, module->ToString());
  if (VLOG_IS_ON(10)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "after layout assignment",
                                module->config().debug_options());
  }
  // All layouts are reset then reassigned by this pass.
  return true;
}

/* static */
bool LayoutAssignment::InstructionCanChangeLayout(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAddDependency:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kComplex:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConditional:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kDivide:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kEq:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFft:
    case HloOpcode::kFloor:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLe:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLt:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kPad:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRemainder:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kScatter:
    case HloOpcode::kSelect:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kSubtract:
    case HloOpcode::kTanh:
    case HloOpcode::kTupleSelect:
    case HloOpcode::kWhile:
      return false;
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCall:
    case HloOpcode::kConstant:
    case HloOpcode::kConvolution:
    case HloOpcode::kCopy:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kOutfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduce:
    case HloOpcode::kReshape:
    case HloOpcode::kRng:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kAfterAll:
    case HloOpcode::kTrace:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
    case HloOpcode::kGetDimensionSize:
      return true;
  }
}

/* static */
bool LayoutAssignment::IsAtMostRank1(const Shape& shape) {
  if (shape.IsArray()) {
    return shape.rank() <= 1;
  }
  return absl::c_all_of(shape.tuple_shapes(), [](const Shape& subshape) {
    return IsAtMostRank1(subshape);
  });
}

Status LayoutAssignment::Init() {
  computation_layouts_.clear();
  *entry_computation_layout_ = saved_entry_computation_layout_;
  return Status::OK();
}

Status LayoutAssignment::ClearPreviousPassSideEffects(HloModule* module) {
  VLOG(5) << "Clearing previous side effects";
  // Clear all the copies which have been added, and all the related
  // instructions (like GTE and tuples).
  int64 removed_copies = 0;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kCopy &&
          added_copies_.contains(instruction)) {
        VLOG(5) << "Removing added copy: " << instruction->ToString();
        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        ++removed_copies;
      }
    }
  }
  added_copies_.clear();
  unconstrained_layout_instructions_.clear();
  if (removed_copies > 0) {
    TupleSimplifier tuple_simplifier;
    HloDCE dce;
    TF_RETURN_IF_ERROR(tuple_simplifier.Run(module).status());
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }
  ResetChannelConstraints();
  return Status::OK();
}

Status LayoutAssignment::AddCopyForOperand(HloInstruction* instruction,
                                           int64 operand_number) {
  HloInstruction* operand = instruction->mutable_operand(operand_number);
  if (operand->opcode() != HloOpcode::kCopy || operand->user_count() > 1) {
    HloInstruction* copy =
        instruction->parent()->AddInstruction(HloInstruction::CreateUnary(
            operand->shape(), HloOpcode::kCopy, operand));
    SetupCopiedInstruction(*operand, copy, {});
    LayoutUtil::ClearLayout(copy->mutable_shape());
    TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(operand_number, copy));
  }
  return Status::OK();
}

}  // namespace xla

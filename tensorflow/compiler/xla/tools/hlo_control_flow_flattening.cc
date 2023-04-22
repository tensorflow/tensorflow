/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/tuple_util.h"

namespace xla {
namespace {

// Create a constant (recursively for tuples) of the given shape and add it to
// the computation.
HloInstruction* CreateConstant(const Shape& shape,
                               HloComputation* computation) {
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> tuple_arguments(shape.tuple_shapes_size());
    for (int index = 0; index < shape.tuple_shapes_size(); ++index) {
      tuple_arguments[index] =
          CreateConstant(shape.tuple_shapes(index), computation);
    }
    return computation->AddInstruction(
        HloInstruction::CreateTuple(tuple_arguments));
  } else {
    return computation->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateFromShape(shape)));
  }
}

// Extracts an instruction that satisfies filter from a fusion instruction.
// Returns nullptr if the fusion doesn't contain any instruction that satisfies
// filter.
const HloInstruction* ExtractInstruction(
    const HloInstruction* hlo,
    const std::function<bool(const HloInstruction*)>& filter) {
  if (filter(hlo)) {
    return hlo;
  }
  if (hlo->opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  for (HloInstruction* inst :
       hlo->fused_instructions_computation()->instructions()) {
    if (filter(inst)) {
      return inst;
    }
  }
  return nullptr;
}

}  // namespace

Status HloControlFlowFlattening::FlattenWhileLoop(
    HloInstruction* while_hlo) const {
  CHECK_EQ(while_hlo->opcode(), HloOpcode::kWhile);
  HloComputation* computation = while_hlo->parent();
  // Add a new induction variable.
  HloInstruction* initialization = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(0)));
  // Create a new while operand with the induction variable added.
  HloInstruction* old_tuple = while_hlo->mutable_operand(0);
  HloInstruction* new_tuple =
      TupleUtil::AppendSuffix(old_tuple, {initialization});
  int new_tuple_size = new_tuple->shape().tuple_shapes().size();
  TF_RETURN_IF_ERROR(while_hlo->ReplaceOperandWithDifferentShape(0, new_tuple));

  auto change_op_shape = [&](HloInstruction* instruction) {
    Shape* shape = instruction->mutable_shape();
    CHECK(shape->IsTuple());
    CHECK_EQ(shape->tuple_shapes().size(), new_tuple_size - 1);
    Shape* subshape = shape->add_tuple_shapes();
    return ShapeUtil::PopulateShape(S32, {}, subshape);
  };

  {
    // Add the new variable to the while loop condition.
    HloComputation* condition = while_hlo->while_condition();
    TF_RETURN_IF_ERROR(change_op_shape(condition->parameter_instruction(0)));

    HloInstruction* limit =
        condition->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int>(while_execution_count_)));
    Shape shape = initialization->shape();
    HloInstruction* induction_variable =
        condition->AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, condition->parameter_instruction(0), new_tuple_size - 1));
    HloInstruction* compare =
        condition->AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), induction_variable, limit,
            ComparisonDirection::kLt));
    TF_RETURN_IF_ERROR(
        condition->ReplaceInstruction(condition->root_instruction(), compare));
  }

  {
    // Add the new variable to the while loop body.
    HloComputation* body = while_hlo->while_body();
    TF_RETURN_IF_ERROR(change_op_shape(body->parameter_instruction(0)));
    HloInstruction* old_root = body->root_instruction();
    Shape shape = initialization->shape();
    HloInstruction* induction_variable =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, body->parameter_instruction(0), new_tuple_size - 1));
    HloInstruction* increment = body->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(1)));
    induction_variable = body->AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, induction_variable, increment));
    HloInstruction* new_root =
        TupleUtil::AppendSuffix(old_root, {induction_variable});
    body->set_root_instruction(new_root, /*accept_different_shape=*/true);
  }

  // Snapshot the users of while hlo before we add new users.
  std::vector<HloInstruction*> while_users(while_hlo->users().begin(),
                                           while_hlo->users().end());

  // Take care of the users of this while loop.
  TF_RETURN_IF_ERROR(change_op_shape(while_hlo));
  HloInstruction* prefix =
      TupleUtil::ExtractPrefix(while_hlo, new_tuple_size - 1);
  for (HloInstruction* user : while_users) {
    TF_RETURN_IF_ERROR(while_hlo->ReplaceUseWithDifferentShape(user, prefix));
  }

  // If the while loop had been the root of its computation, make the prefix new
  // root.
  if (while_hlo->parent()->root_instruction() == while_hlo) {
    // We need to set accept_different_shape=true to reset the root shape to the
    // original, because we have already changed the shape of the old root
    // (while).
    while_hlo->parent()->set_root_instruction(prefix,
                                              /*accept_different_shape=*/true);
  }

  return Status::OK();
}

Status HloControlFlowFlattening::RemoveInfeed(
    HloInstruction* infeed_hlo) const {
  CHECK_EQ(infeed_hlo->opcode(), HloOpcode::kInfeed);
  HloComputation* computation = infeed_hlo->parent();
  CHECK_EQ(infeed_hlo->shape().tuple_shapes_size(), 2);
  const Shape& infeed_shape = ShapeUtil::GetSubshape(infeed_hlo->shape(), {0});

  HloInstruction* constant = CreateConstant(infeed_shape, computation);

  // Create a new tuple consisting op the constant and the token that was
  // originally the operand of infeed, and replace the infeed operation.
  auto new_tuple =
      HloInstruction::CreateTuple({constant, infeed_hlo->mutable_operand(0)});
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(infeed_hlo, std::move(new_tuple)));

  return Status::OK();
}

Status HloControlFlowFlattening::RemoveOutfeed(
    HloInstruction* outfeed_hlo) const {
  CHECK_EQ(outfeed_hlo->opcode(), HloOpcode::kOutfeed);
  HloComputation* computation = outfeed_hlo->parent();
  // Replace the outfeed with a no-op custom call with side effect to ensure the
  // operands aren't DCE'd.
  auto custom_call = HloInstruction::CreateCustomCall(
      outfeed_hlo->shape(), outfeed_hlo->operands(), "NopReturnToken");
  Cast<HloCustomCallInstruction>(custom_call.get())
      ->set_custom_call_has_side_effect(true);
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      outfeed_hlo, std::move(custom_call)));

  return Status::OK();
}

Status HloControlFlowFlattening::RemoveAllReduce(HloInstruction* hlo) const {
  HloComputation* computation = hlo->parent();
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          ShapeUtil::MakeTokenShape(), hlo->operands(), "NopReturnToken"));
  Cast<HloCustomCallInstruction>(custom_call)
      ->set_custom_call_has_side_effect(true);

  HloInstruction* constant = CreateConstant(hlo->shape(), computation);
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, constant));
  return Status::OK();
}

StatusOr<bool> HloControlFlowFlattening::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (flatten_while_loop_ && instruction->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(FlattenWhileLoop(instruction));
        changed = true;
      } else if (remove_infeed_outfeed_ &&
                 instruction->opcode() == HloOpcode::kInfeed) {
        TF_RETURN_IF_ERROR(RemoveInfeed(instruction));
        changed = true;
      } else if (remove_infeed_outfeed_ &&
                 instruction->opcode() == HloOpcode::kOutfeed) {
        TF_RETURN_IF_ERROR(RemoveOutfeed(instruction));
        changed = true;
      } else if (remove_all_reduce_ &&
                 (instruction->opcode() == HloOpcode::kAllReduce ||
                  (instruction->opcode() == HloOpcode::kFusion &&
                   ExtractInstruction(
                       instruction, [](const HloInstruction* hlo) {
                         return hlo->opcode() == HloOpcode::kAllReduce;
                       }) != nullptr))) {
        TF_RETURN_IF_ERROR(RemoveAllReduce(instruction));
        changed = true;
      }
    }
  }

  // Fix the schedule if the module was scheduled.
  if (changed && module->has_schedule()) {
    TF_RETURN_IF_ERROR(module->schedule().Update());
  }
  XLA_VLOG_LINES(1, module->ToString());
  return changed;
}

}  // namespace xla

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
#include "xla/hlo/transforms/collectives/while_loop_all_reduce_code_motion_setup.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/statusor.h"

namespace xla {

bool ReorderReduceTranspose::InstructionMatchesPattern(
    HloInstruction* instruction) {
  // Instruction must be in while loop body.
  if (!instruction->parent()->IsWhileBodyComputation()) {
    return false;
  }
  // Search for Reduce Scatter Transpose pairs with optional convert in between
  if (instruction->opcode() != HloOpcode::kTranspose) {
    return false;
  }

  HloInstruction* operand = instruction->mutable_operand(0);

  // Check if the operand is a convert instruction
  if (operand->opcode() == HloOpcode::kConvert) {
    operand = operand->mutable_operand(0);
  }

  // Transpose operand is ReduceScatter
  if (operand->opcode() != HloOpcode::kReduceScatter) {
    return false;
  }

  VLOG(2) << "Found Reduce Scatter (Convert) Transpose Pair:"
          << operand->ToString() << "\n"
          << instruction->ToString();

  if (operand->operand_count() != 1) {
    VLOG(2) << "Reject Reduce Scatter (Convert) Transpose Pair because Reduce "
               "Scatter "
            << "has operand count " << operand->operand_count()
            << " more than 1 supported by this pass";
    return false;
  }
  if (instruction->user_count() == 0) {
    return false;
  }

  // RepeatedTransformers case
  // reduce-scatter->transpose->reshape->dynamic-update-slice
  if (instruction->users()[0]->opcode() == HloOpcode::kReshape) {
    // Look for the dynamic update slice
    auto* reshape = instruction->users()[0];
    if (reshape->user_count() == 0) {
      return false;
    }
    return reshape->users()[0]->opcode() == HloOpcode::kDynamicUpdateSlice;
  }

  // Check if the Transpose is used in an Add operation
  if (instruction->users()[0]->opcode() != HloOpcode::kAdd) {
    return false;
  }

  HloInstruction* add_instruction = instruction->users()[0];

  // Check if the first or second operand of the Add is a GetTupleElement whose
  // operand is a Parameter
  HloInstruction* second_operand =
      add_instruction->operand(0)->opcode() == HloOpcode::kGetTupleElement
          ? add_instruction->mutable_operand(0)
          : add_instruction->mutable_operand(1);
  if (second_operand->opcode() != HloOpcode::kGetTupleElement) {
    return false;
  }
  HloInstruction* gte_operand = second_operand->mutable_operand(0);
  if (gte_operand->opcode() != HloOpcode::kParameter) {
    return false;
  }
  return true;
}

absl::StatusOr<HloInstruction*> ReorderReduceTranspose::ExpandInstruction(
    HloInstruction* instruction) {
  auto* transpose = Cast<HloTransposeInstruction>(instruction);
  HloInstruction* operand = instruction->mutable_operand(0);

  // Check if the operand is a convert instruction
  bool has_convert = operand->opcode() == HloOpcode::kConvert;
  auto* reduce_scatter =
      has_convert
          ? Cast<HloReduceScatterInstruction>(operand->mutable_operand(0))
          : Cast<HloReduceScatterInstruction>(operand);

  // Create a new Convert instruction if the original pattern had one
  HloInstruction* new_convert = nullptr;
  if (has_convert) {
    new_convert =
        instruction->parent()->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(
                reduce_scatter->mutable_operand(0)->shape(),
                operand->shape().element_type()),
            reduce_scatter->mutable_operand(0)));
  }

  // Create a new Transpose instruction that uses the same dimension
  // for permutation as before, but on the converted operand (if applicable)
  // or the original reduce-scatter operand.
  TF_ASSIGN_OR_RETURN(
      auto* new_transpose,
      MakeTransposeHlo(
          has_convert ? new_convert : reduce_scatter->mutable_operand(0),
          transpose->dimensions()));

  // Create a new ReduceScatter instruction that uses the same replica
  // groups as before, but on the new transpose. The scatter dimension has
  // now changed based on the transpose, so find it through the transpose
  // permutation.
  int64_t new_scatter_dim = -1;
  for (int i = 0; i < transpose->shape().rank(); i++) {
    if (transpose->dimensions()[i] == reduce_scatter->scatter_dimension()) {
      new_scatter_dim = i;
      break;
    }
  }

  return instruction->parent()->AddInstruction(
      HloInstruction::CreateReduceScatter(
          transpose->shape(), {new_transpose},
          reduce_scatter->called_computations()[0],
          reduce_scatter->replica_groups(), reduce_scatter->constrain_layout(),
          reduce_scatter->channel_id(), reduce_scatter->use_global_device_ids(),
          new_scatter_dim));
}

bool ReorderConvertReduceAdd::InstructionMatchesPattern(
    HloInstruction* instruction) {
  // Instruction must be in while loop body.
  if (!instruction->parent()->IsWhileBodyComputation()) {
    return false;
  }
  // Check if the instruction is an add operation
  if (instruction->opcode() != HloOpcode::kAdd) {
    return false;
  }

  // Check if one of the operands is a convert operation
  HloInstruction* convert_operand = nullptr;
  HloInstruction* get_tuple_element_operand = nullptr;
  for (HloInstruction* operand : instruction->operands()) {
    if (operand->opcode() == HloOpcode::kConvert) {
      convert_operand = operand;
    } else if (operand->opcode() == HloOpcode::kGetTupleElement) {
      get_tuple_element_operand = operand;
    }
  }
  if (convert_operand == nullptr || get_tuple_element_operand == nullptr) {
    return false;
  }

  // Check if the operand of the convert operation is a reduce-scatter or
  // all-reduce
  HloInstruction* reduce_op_operand = convert_operand->mutable_operand(0);
  if (reduce_op_operand->opcode() != HloOpcode::kReduceScatter &&
      reduce_op_operand->opcode() != HloOpcode::kAllReduce) {
    return false;
  }
  // Check if the reduce_op_operand is a reduce-scatter and
  // enable_reduce_scatter_ is true.
  if (!enable_reduce_scatter_ &&
      reduce_op_operand->opcode() == HloOpcode::kReduceScatter) {
    return false;
  }

  // Check if the get-tuple-element instruction is operating on a parameter
  // tuple
  HloInstruction* tuple_operand = get_tuple_element_operand->mutable_operand(0);
  if (tuple_operand->opcode() != HloOpcode::kParameter) {
    return false;
  }

  VLOG(2) << "Found pattern: reduce-scatter/all-reduce, convert, add, with "
             "get-tuple-element on parameter tuple";
  return true;
}

absl::StatusOr<HloInstruction*> ReorderConvertReduceAdd::ExpandInstruction(
    HloInstruction* instruction) {
  VLOG(2) << "Entering ExpandInstruction";

  // Get the add, convert, and reduce-scatter/all-reduce instructions
  HloInstruction* add = instruction;
  HloInstruction* convert = nullptr;
  HloInstruction* other_operand = nullptr;
  for (HloInstruction* operand : add->operands()) {
    if (operand->opcode() == HloOpcode::kConvert) {
      convert = operand;
    } else {
      other_operand = operand;
    }
  }
  // Pattern matched in `InstructionMatchesPattern`.
  CHECK(convert != nullptr && other_operand != nullptr);
  HloInstruction* reduce_op = convert->mutable_operand(0);

  VLOG(2) << "Found add: " << add->ToString();
  VLOG(2) << "Found convert: " << convert->ToString();
  VLOG(2) << "Found reduce_op: " << reduce_op->ToString();
  VLOG(2) << "Found other_operand: " << other_operand->ToString();

  // Create a new convert instruction with the reduce-scatter/all-reduce operand
  PrimitiveType new_data_type = convert->shape().element_type();
  HloInstruction* new_convert =
      instruction->parent()->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(reduce_op->operand(0)->shape(),
                                       new_data_type),
          reduce_op->mutable_operand(0)));

  VLOG(2) << "Created new_convert: " << new_convert->ToString();

  // Create a new reduce-scatter/all-reduce instruction with the converted data
  // type
  HloInstruction* new_reduce_op;
  if (reduce_op->opcode() == HloOpcode::kReduceScatter) {
    auto* reduce_scatter = Cast<HloReduceScatterInstruction>(reduce_op);
    Shape new_reduce_scatter_shape =
        ShapeUtil::ChangeElementType(reduce_scatter->shape(), new_data_type);

    new_reduce_op = instruction->parent()->AddInstruction(
        HloInstruction::CreateReduceScatter(
            new_reduce_scatter_shape, {new_convert},
            reduce_scatter->called_computations()[0],
            reduce_scatter->replica_groups(),
            reduce_scatter->constrain_layout(), reduce_scatter->channel_id(),
            reduce_scatter->use_global_device_ids(),
            reduce_scatter->scatter_dimension()));
    VLOG(2) << "Created new_reduce_op (ReduceScatter): "
            << new_reduce_op->ToString();
  } else {
    auto* all_reduce = Cast<HloAllReduceInstruction>(reduce_op);
    Shape new_all_reduce_shape =
        ShapeUtil::ChangeElementType(all_reduce->shape(), new_data_type);

    new_reduce_op =
        instruction->parent()->AddInstruction(HloInstruction::CreateAllReduce(
            new_all_reduce_shape, {new_convert},
            all_reduce->called_computations()[0], all_reduce->replica_groups(),
            all_reduce->constrain_layout(), all_reduce->channel_id(),
            all_reduce->use_global_device_ids()));
    VLOG(2) << "Created new_reduce_op (AllReduce): "
            << new_reduce_op->ToString();
  }

  // Create a new add instruction with the new reduce-scatter/all-reduce and the
  // other operand
  HloInstruction* new_add =
      instruction->parent()->AddInstruction(HloInstruction::CreateBinary(
          add->shape(), HloOpcode::kAdd, new_reduce_op, other_operand));

  VLOG(2) << "Created new_add: " << new_add->ToString();
  VLOG(2) << "Leaving ExpandInstruction";

  return new_add;
}

}  // namespace xla

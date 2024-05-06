/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/gpu/dot_operand_converter.h"

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"

namespace xla::gpu {

bool DotOperandConverter::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDot) {
    return false;
  }
  HloInstruction* lhs = instruction->mutable_operand(0);
  HloInstruction* rhs = instruction->mutable_operand(1);

  PrimitiveType lhs_type = lhs->shape().element_type();
  PrimitiveType rhs_type = rhs->shape().element_type();

  if (lhs_type == rhs_type) {
    return false;
  }

  // Exclude conversions between FP8 types.
  absl::flat_hash_set<PrimitiveType> non_converting = {F8E4M3FN, F8E5M2};
  if (non_converting.contains(lhs_type) && non_converting.contains(rhs_type)) {
    return false;
  }

  PrimitiveType desired_type =
      ShapeUtil::HigherPrecisionElementType(lhs->shape(), rhs->shape());

  return desired_type == lhs_type || desired_type == rhs_type;
}

absl::StatusOr<HloInstruction*> DotOperandConverter::ExpandInstruction(
    HloInstruction* instruction) {
  HloInstruction* lhs = instruction->mutable_operand(0);
  HloInstruction* rhs = instruction->mutable_operand(1);

  // Find the higher precision type among the two operands, and add a convert
  // instruction to convert the lesser-precise operand to that type.
  PrimitiveType desired_type =
      ShapeUtil::HigherPrecisionElementType(lhs->shape(), rhs->shape());
  int operand_index = desired_type == lhs->shape().element_type() ? 1 : 0;
  HloInstruction* inst_to_replace =
      desired_type == lhs->shape().element_type() ? rhs : lhs;
  auto upcast_shape = inst_to_replace->shape();
  upcast_shape.set_element_type(desired_type);
  auto* convert_inst = instruction->AddInstruction(
      HloInstruction::CreateConvert(upcast_shape, inst_to_replace));
  TF_RETURN_IF_ERROR(instruction->ReplaceOperandWithDifferentShape(
      operand_index, convert_inst));
  return nullptr;
}

}  // namespace xla::gpu

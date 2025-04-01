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

#include "xla/service/gpu/transforms/dot_normalizer.h"

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"

namespace xla::gpu {

bool DotNormalizer::InstructionMatchesPattern(HloInstruction* instruction) {
  if (HloPredicateIsNotOp<HloOpcode::kDot>(instruction)) {
    return false;
  }
  return instruction->dot_dimension_numbers()
      .lhs_contracting_dimensions()
      .empty();
}

absl::StatusOr<HloInstruction*> DotNormalizer::ExpandInstruction(
    HloInstruction* instruction) {
  HloDotInstruction* dot = Cast<HloDotInstruction>(instruction);
  HloInstruction* lhs = dot->mutable_operand(0);
  Shape new_lhs_shape = lhs->shape();
  ShapeUtil::AppendMinorDimension(1, &new_lhs_shape);
  HloInstruction* normalized_lhs =
      dot->AddInstruction(HloInstruction::CreateBitcast(new_lhs_shape, lhs));
  TF_RETURN_IF_ERROR(dot->ReplaceOperandWithDifferentShape(0, normalized_lhs));
  HloInstruction* rhs = dot->mutable_operand(1);
  Shape new_rhs_shape = rhs->shape();
  ShapeUtil::AppendMinorDimension(1, &new_rhs_shape);
  HloInstruction* normalized_rhs =
      dot->AddInstruction(HloInstruction::CreateBitcast(new_rhs_shape, rhs));
  TF_RETURN_IF_ERROR(dot->ReplaceOperandWithDifferentShape(1, normalized_rhs));
  DotDimensionNumbers* dnums = dot->mutable_dot_dimension_numbers();
  dnums->add_lhs_contracting_dimensions(new_lhs_shape.dimensions_size() - 1);
  dnums->add_rhs_contracting_dimensions(new_rhs_shape.dimensions_size() - 1);
  return nullptr;
}

}  // namespace xla::gpu

// Copyright 2025 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "xla/hlo/transforms/expanders/convolution_type_canonicalizer.h"

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"

namespace xla {

bool ConvolutionTypeCanonicalizer::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return (instruction->opcode() == HloOpcode::kDot ||
          instruction->opcode() == HloOpcode::kConvolution) &&
         (primitive_util::IsFloatingPointType(
              instruction->operand(0)->shape().element_type()) &&
          primitive_util::IsFloatingPointType(
              instruction->operand(1)->shape().element_type())) &&
         primitive_util::IsIntegralType(instruction->shape().element_type());
}

absl::StatusOr<HloInstruction*> ConvolutionTypeCanonicalizer::ExpandInstruction(
    HloInstruction* instruction) {
  auto original_shape = instruction->shape();
  auto new_shape = ShapeUtil::ChangeElementType(original_shape, F32);
  HloInstruction* replacement_instruction;
  if (instruction->opcode() == HloOpcode::kDot) {
    replacement_instruction = instruction->parent()->AddInstruction(
        HloInstruction::CreateDot(new_shape, instruction->mutable_operand(0),
                                  instruction->mutable_operand(1),
                                  instruction->dot_dimension_numbers(),
                                  instruction->precision_config()));
  } else {
    replacement_instruction =
        instruction->parent()->AddInstruction(HloInstruction::CreateConvolve(
            new_shape, instruction->mutable_operand(0),
            instruction->mutable_operand(1), instruction->feature_group_count(),
            instruction->batch_group_count(), instruction->window(),
            instruction->convolution_dimension_numbers(),
            instruction->precision_config()));
  }
  HloInstruction* output_cast = instruction->parent()->AddInstruction(
      HloInstruction::CreateConvert(original_shape, replacement_instruction));
  return output_cast;
}

}  // namespace xla

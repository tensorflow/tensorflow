/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/operand_upcaster.h"

#include <optional>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

StatusOr<std::optional<Shape>> MaybeInferShape(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kDot:
      return ShapeInference::InferDotOpShape(
          instruction->operand(0)->shape(), instruction->operand(1)->shape(),
          instruction->dot_dimension_numbers(),
          /*preferred_element_type=*/std::nullopt);
    case HloOpcode::kConvolution:
      return ShapeInference::InferConvolveShape(
          instruction->operand(0)->shape(), instruction->operand(1)->shape(),
          instruction->feature_group_count(), instruction->batch_group_count(),
          instruction->window(), instruction->convolution_dimension_numbers(),
          /*preferred_element_type=*/std::nullopt);
    default:
      return std::optional<Shape>(std::nullopt);
  }
}

}  // namespace

bool OperandUpcaster::InstructionMatchesPattern(HloInstruction* instruction) {
  auto status_or_inferred_shape = MaybeInferShape(instruction);
  if (!status_or_inferred_shape.ok() ||
      !status_or_inferred_shape->has_value()) {
    return false;
  }

  // Always expand packed nibble precision mode.
  if (absl::c_count(instruction->precision_config().operand_precision(),
                    PrecisionConfig::PACKED_NIBBLE) == 2) {
    return true;
  }

  const Shape& inferred_shape = status_or_inferred_shape.value().value();
  if (inferred_shape.element_type() == instruction->shape().element_type() &&
      absl::c_all_of(instruction->operands(),
                     [&](const HloInstruction* operand) {
                       return operand->shape().element_type() ==
                              inferred_shape.element_type();
                     })) {
    return false;
  }
  return ShapeUtil::ElementCanUpcast(inferred_shape, instruction->shape());
}

StatusOr<HloInstruction*> OperandUpcaster::ExpandInstruction(
    HloInstruction* instruction) {
  const bool packed_nibble =
      absl::c_count(instruction->precision_config().operand_precision(),
                    PrecisionConfig::PACKED_NIBBLE) == 2;
  auto type = instruction->shape().element_type();

  // If the precision is packed nibble create clone the linear op for each
  // nibble of lhs and rhs.
  if (packed_nibble) {
    HloInstruction *lhs_n0 = instruction->mutable_operand(0), *lhs_n1 = lhs_n0,
                   *rhs_n0 = instruction->mutable_operand(1), *rhs_n1 = rhs_n0;

    TF_ASSIGN_OR_RETURN(lhs_n0, MakeBinaryHlo(HloOpcode::kShiftLeft, lhs_n0,
                                              MakeScalarLike(lhs_n0, 4)));
    HloOpcode lhs_shift = ShapeUtil::ElementIsSigned(lhs_n0->shape())
                              ? HloOpcode::kShiftRightArithmetic
                              : HloOpcode::kShiftRightLogical;
    TF_ASSIGN_OR_RETURN(
        lhs_n0, MakeBinaryHlo(lhs_shift, lhs_n0, MakeScalarLike(lhs_n0, 4)));
    lhs_n0 = MakeConvertToHlo(lhs_n0, type);

    TF_ASSIGN_OR_RETURN(
        lhs_n1, MakeBinaryHlo(lhs_shift, lhs_n1, MakeScalarLike(lhs_n1, 4)));
    lhs_n1 = MakeConvertToHlo(lhs_n1, type);

    TF_ASSIGN_OR_RETURN(rhs_n0, MakeBinaryHlo(HloOpcode::kShiftLeft, rhs_n0,
                                              MakeScalarLike(rhs_n0, 4)));
    HloOpcode rhs_shift = ShapeUtil::ElementIsSigned(rhs_n0->shape())
                              ? HloOpcode::kShiftRightArithmetic
                              : HloOpcode::kShiftRightLogical;
    TF_ASSIGN_OR_RETURN(
        rhs_n0, MakeBinaryHlo(rhs_shift, rhs_n0, MakeScalarLike(rhs_n0, 4)));
    rhs_n0 = MakeConvertToHlo(rhs_n0, type);

    TF_ASSIGN_OR_RETURN(
        rhs_n1, MakeBinaryHlo(rhs_shift, rhs_n1, MakeScalarLike(rhs_n1, 4)));
    rhs_n1 = MakeConvertToHlo(rhs_n1, type);

    HloInstruction* linear_n0 =
        instruction->parent()->AddInstruction(instruction->CloneWithNewOperands(
            instruction->shape(), {lhs_n0, rhs_n0}));
    linear_n0->mutable_precision_config()->mutable_operand_precision()->Set(
        0, PrecisionConfig::DEFAULT);
    linear_n0->mutable_precision_config()->mutable_operand_precision()->Set(
        1, PrecisionConfig::DEFAULT);
    HloInstruction* linear_n1 =
        instruction->parent()->AddInstruction(linear_n0->CloneWithNewOperands(
            instruction->shape(), {lhs_n1, rhs_n1}));
    return MakeBinaryHlo(HloOpcode::kAdd, linear_n0, linear_n1);
  }

  for (int i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (operand->shape().element_type() == type) {
      continue;
    }
    auto upcast_shape = operand->shape();
    upcast_shape.set_element_type(type);
    auto* convert_inst = instruction->AddInstruction(
        HloInstruction::CreateConvert(upcast_shape, operand));
    TF_RETURN_IF_ERROR(
        instruction->ReplaceOperandWithDifferentShape(i, convert_inst));
  }
  return nullptr;
}

}  // namespace xla

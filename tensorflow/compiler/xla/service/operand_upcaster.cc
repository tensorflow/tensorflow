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

#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {
namespace {

StatusOr<absl::optional<Shape>> MaybeInferShape(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kDot:
      return ShapeInference::InferDotOpShape(
          instruction->operand(0)->shape(), instruction->operand(1)->shape(),
          instruction->dot_dimension_numbers(),
          /*preferred_element_type=*/absl::nullopt);
    case HloOpcode::kConvolution:
      return ShapeInference::InferConvolveShape(
          instruction->operand(0)->shape(), instruction->operand(1)->shape(),
          instruction->feature_group_count(), instruction->batch_group_count(),
          instruction->window(), instruction->convolution_dimension_numbers(),
          /*preferred_element_type=*/absl::nullopt);
    default:
      return absl::optional<Shape>(absl::nullopt);
  }
}

}  // namespace

bool OperandUpcaster::InstructionMatchesPattern(HloInstruction* instruction) {
  auto status_or_inferred_shape = MaybeInferShape(instruction);
  if (!status_or_inferred_shape.ok() ||
      !status_or_inferred_shape->has_value()) {
    return false;
  }
  const Shape& inferred_shape = status_or_inferred_shape.ValueOrDie().value();
  if (inferred_shape.element_type() == instruction->shape().element_type()) {
    return false;
  }
  return ShapeUtil::ElementCanUpcast(inferred_shape, instruction->shape());
}

StatusOr<HloInstruction*> OperandUpcaster::ExpandInstruction(
    HloInstruction* instruction) {
  auto* computation = instruction->parent();
  auto type = instruction->shape().element_type();
  for (int i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (operand->shape().element_type() == type) {
      continue;
    }
    auto upcast_shape = operand->shape();
    upcast_shape.set_element_type(type);
    auto* convert_inst = computation->AddInstruction(
        HloInstruction::CreateConvert(upcast_shape, operand));
    TF_RETURN_IF_ERROR(
        instruction->ReplaceOperandWithDifferentShape(i, convert_inst));
  }
  return nullptr;
}

}  // namespace xla

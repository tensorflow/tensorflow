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

#include "xla/hlo/transforms/operand_upcaster.h"

#include <optional>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

absl::StatusOr<std::optional<Shape>> MaybeInferShape(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kDot:
      return ShapeInference::InferDotOpShape(
          instruction->operand(0)->shape(), instruction->operand(1)->shape(),
          instruction->dot_dimension_numbers(),
          /*preferred_element_type=*/std::nullopt,
          Cast<HloDotInstruction>(instruction)->sparsity());
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

  PrimitiveType inferred_type = (*status_or_inferred_shape)->element_type();
  if (instruction->shape().element_type() == inferred_type &&
      instruction->operand(0)->shape().element_type() == inferred_type &&
      instruction->operand(1)->shape().element_type() == inferred_type) {
    return false;
  }
  return ShapeUtil::ElementCanUpcast(**status_or_inferred_shape,
                                     instruction->shape());
}

absl::StatusOr<HloInstruction*> OperandUpcaster::ExpandInstruction(
    HloInstruction* instruction) {
  auto type = instruction->shape().element_type();

  for (int i = 0; i < HloDotInstruction::kOperands; ++i) {
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

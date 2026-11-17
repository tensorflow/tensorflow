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

#include "xla/hlo/transforms/simplifiers/convert_operand_folder.h"

#include <optional>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

// Returns the pre-convert element type if `hlo` is a value-preserving convert
// (possibly wrapped in formatting ops), or std::nullopt otherwise.
std::optional<PrimitiveType> UpcastSourceType(const HloInstruction* hlo) {
  if (!hlo->shape().IsArray()) {
    return std::nullopt;
  }
  switch (hlo->opcode()) {
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      return UpcastSourceType(hlo->operand(0));
    }
    case HloOpcode::kReduce: {
      if (ShapeUtil::ElementsIn(hlo->shape()) ==
          ShapeUtil::ElementsIn(hlo->operand(0)->shape())) {
        return UpcastSourceType(hlo->operand(0));
      }
      return std::nullopt;
    }
    case HloOpcode::kConvert:
      if (primitive_util::CastPreservesValues(
              hlo->operand(0)->shape().element_type(),
              hlo->shape().element_type())) {
        return hlo->operand(0)->shape().element_type();
      }
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

HloInstruction* EffectiveOperand(HloInstruction* hlo) {
  switch (hlo->opcode()) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      HloInstruction* operand = EffectiveOperand(hlo->mutable_operand(0));
      HloInstruction* clone = hlo->AddInstruction(hlo->Clone());
      *(clone->mutable_shape()) = ShapeUtil::ChangeElementType(
          clone->shape(), operand->shape().element_type());
      clone->ReplaceOperandWithDifferentShape(0, operand).IgnoreError();
      return clone;
    }
    case HloOpcode::kReduce: {
      // Reduce is a reshape in the case the the hlo chain was an upcast.
      HloInstruction* operand = EffectiveOperand(hlo->mutable_operand(0));
      return hlo->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::ChangeElementType(hlo->shape(),
                                       operand->shape().element_type()),
          operand));
    }
    case HloOpcode::kConvert:
      return hlo->mutable_operand(0);
    default:
      return nullptr;
  }
}

}  // namespace

bool ConvertOperandFolding::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDot &&
      instruction->opcode() != HloOpcode::kConvolution) {
    return false;
  }
  std::optional<PrimitiveType> lhs_upcast =
      UpcastSourceType(instruction->operand(0));
  std::optional<PrimitiveType> rhs_upcast =
      UpcastSourceType(instruction->operand(1));
  bool has_upcast = lhs_upcast.has_value() || rhs_upcast.has_value();
  for (int i = 2; !has_upcast && i < instruction->operand_count(); ++i) {
    has_upcast = UpcastSourceType(instruction->operand(i)).has_value();
  }
  if (!has_upcast) {
    return false;
  }
  return CanFold(
      *instruction,
      lhs_upcast.value_or(instruction->operand(0)->shape().element_type()),
      rhs_upcast.value_or(instruction->operand(1)->shape().element_type()));
}

absl::StatusOr<HloInstruction*> ConvertOperandFolding::ExpandInstruction(
    HloInstruction* instruction) {
  for (int i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (UpcastSourceType(operand).has_value()) {
      ABSL_RETURN_IF_ERROR(instruction->ReplaceOperandWithDifferentShape(
          i, EffectiveOperand(operand)));
    }
  }
  return nullptr;
}

}  // namespace xla

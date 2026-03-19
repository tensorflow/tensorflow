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

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

bool IsUpcastConvert(const HloInstruction* hlo) {
  if (!hlo->shape().IsArray()) {
    return false;
  }
  switch (hlo->opcode()) {
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      return IsUpcastConvert(hlo->operand(0));
    }
    case HloOpcode::kReduce: {
      if (ShapeUtil::ElementsIn(hlo->shape()) ==
          ShapeUtil::ElementsIn(hlo->operand(0)->shape())) {
        return IsUpcastConvert(hlo->operand(0));
      }
      return false;
    }
    case HloOpcode::kConvert:
      return primitive_util::CastPreservesValues(
          hlo->operand(0)->shape().element_type(), hlo->shape().element_type());
    default:
      return false;
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
  for (auto* operand : instruction->operands()) {
    if (IsUpcastConvert(operand)) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<HloInstruction*> ConvertOperandFolding::ExpandInstruction(
    HloInstruction* instruction) {
  for (int i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (IsUpcastConvert(operand)) {
      TF_RETURN_IF_ERROR(instruction->ReplaceOperandWithDifferentShape(
          i, EffectiveOperand(operand)));
    }
  }
  return nullptr;
}

}  // namespace xla

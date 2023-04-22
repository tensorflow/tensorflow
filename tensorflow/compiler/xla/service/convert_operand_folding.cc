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

#include "tensorflow/compiler/xla/service/convert_operand_folding.h"

#include "absl/base/attributes.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

bool IsUpcastConvert(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kConvert) {
    return false;
  }
  return primitive_util::CastPreservesValues(
      hlo->operand(0)->shape().element_type(), hlo->shape().element_type());
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

StatusOr<HloInstruction*> ConvertOperandFolding::ExpandInstruction(
    HloInstruction* instruction) {
  for (int i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (IsUpcastConvert(operand)) {
      TF_RETURN_IF_ERROR(instruction->ReplaceOperandWithDifferentShape(
          i, operand->mutable_operand(0)));
    }
  }
  return nullptr;
}

}  // namespace xla

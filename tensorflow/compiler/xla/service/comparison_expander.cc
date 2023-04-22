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

#include "tensorflow/compiler/xla/service/comparison_expander.h"

#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

HloInstruction* BitcastConvertFloatingPointToIntegral(
    HloComputation* computation, HloInstruction* value,
    const Shape& signed_shape, const Shape& unsigned_shape,
    HloInstruction* zero, HloInstruction* max_value) {
  // Switch from a floating point value to a integer value in such a way that
  // when using the integer value to compare, we get the same result for normal
  // values, and -Nan is treated as the smallest value, and Nan is treated as
  // the largest value.
  // If f is a float, and
  // x = bit_cast<int32>(f);
  // y = x < 0 ? numeric_limits<int32>::max() - x : x;
  // then y is ordered as an int32 such that finite values have the obvious
  // order, -0 is ordered before 0, and -NaN and NaN appear at the beginning
  // and end of the ordering.
  // Note that in order to avoid -x to overflow, we calculate
  // numeric_limits<int32>::max() - x as unsigned, and then convert back to
  // signed.
  auto signed_value = computation->AddInstruction(
      HloInstruction::CreateBitcastConvert(signed_shape, value));
  auto unsigned_value = computation->AddInstruction(
      HloInstruction::CreateBitcastConvert(unsigned_shape, value));
  auto flipped_value = computation->AddInstruction(HloInstruction::CreateBinary(
      unsigned_shape, HloOpcode::kSubtract, max_value, unsigned_value));
  flipped_value = computation->AddInstruction(
      HloInstruction::CreateBitcastConvert(signed_shape, flipped_value));
  auto compare_shape = signed_shape;
  compare_shape.set_element_type(PRED);
  auto is_negative = computation->AddInstruction(HloInstruction::CreateCompare(
      compare_shape, signed_value, zero, ComparisonDirection::kLt));
  return computation->AddInstruction(
      HloInstruction::CreateTernary(signed_shape, HloOpcode::kSelect,
                                    is_negative, flipped_value, signed_value));
}

bool ComparisonExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (HloCompareInstruction* compare =
          dynamic_cast<HloCompareInstruction*>(instruction)) {
    HloInstruction* lhs = instruction->operands()[0];
    if (compare->type() == Comparison::Type::kFloatTotalOrder &&
        primitive_util::IsFloatingPointType(lhs->shape().element_type())) {
      return true;
    }
  }
  return false;
}

StatusOr<HloInstruction*> ComparisonExpander::ExpandInstruction(
    HloInstruction* instruction) {
  CHECK(instruction->opcode() == HloOpcode::kCompare);
  HloCompareInstruction* compare =
      static_cast<HloCompareInstruction*>(instruction);
  CHECK(compare->type() == Comparison::Type::kFloatTotalOrder);
  HloComputation* computation = instruction->parent();
  HloInstruction* lhs = instruction->operands()[0];
  HloInstruction* rhs = instruction->operands()[1];
  Shape compare_shape = lhs->shape();
  PrimitiveType compare_type = compare_shape.element_type();
  CHECK(primitive_util::IsFloatingPointType(compare_type));
  // Special-case handling for BF16. We currently do not support direct
  // comparisons with BF16, so we convert to F32 and then use the F32
  // comparison logic.
  if (compare_type == BF16) {
    compare_type = F32;
    compare_shape.set_element_type(compare_type);
    lhs = computation->AddInstruction(
        HloInstruction::CreateConvert(compare_shape, lhs));
    rhs = computation->AddInstruction(
        HloInstruction::CreateConvert(compare_shape, rhs));
  }

  int64 bit_width = primitive_util::BitWidth(compare_type);
  PrimitiveType signed_type =
      primitive_util::SignedIntegralTypeForBitWidth(bit_width);
  PrimitiveType unsigned_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(bit_width);
  auto signed_shape = compare_shape;
  signed_shape.set_element_type(signed_type);
  auto unsigned_shape = compare_shape;
  unsigned_shape.set_element_type(unsigned_type);
  auto zero_value = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(signed_type)));
  zero_value = computation->AddInstruction(HloInstruction::CreateBroadcast(
      signed_shape, zero_value, zero_value->shape().dimensions()));
  auto max_signed = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MaxValue(signed_type)));
  auto max_shape = max_signed->shape();
  max_shape.set_element_type(unsigned_type);
  auto max_unsigned = computation->AddInstruction(
      HloInstruction::CreateConvert(max_shape, max_signed));
  auto max_value = computation->AddInstruction(HloInstruction::CreateBroadcast(
      unsigned_shape, max_unsigned, max_shape.dimensions()));
  lhs = BitcastConvertFloatingPointToIntegral(
      computation, lhs, signed_shape, unsigned_shape, zero_value, max_value);
  rhs = BitcastConvertFloatingPointToIntegral(
      computation, rhs, signed_shape, unsigned_shape, zero_value, max_value);
  auto new_compare = computation->AddInstruction(HloInstruction::CreateCompare(
      instruction->shape(), lhs, rhs, compare->direction(),
      Comparison::Type::kSigned));
  VLOG(2) << "New comparison instruction for total order:"
          << new_compare->ToString() << "\n";
  return new_compare;
}

}  // namespace xla

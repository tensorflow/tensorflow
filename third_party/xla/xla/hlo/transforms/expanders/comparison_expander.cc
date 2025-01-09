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

#include "xla/hlo/transforms/expanders/comparison_expander.h"

#include <cstdint>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

HloInstruction* BitcastConvertFloatingPointToIntegral(
    HloComputation* computation, HloInstruction* value, HloInstruction* zero,
    HloInstruction* min_value, HloInstruction* max_value) {
  // Switch from a floating point value to a integer value in such a way that
  // when using the integer value to compare, we get the same result for normal
  // values, and -Nan is treated as the smallest value, and Nan is treated as
  // the largest value.
  // If f is a float, and
  // x = bit_cast<int32_t>(f);
  // y = x < 0 ? numeric_limits<int32_t>::max() ^ x : x;
  // then y is ordered as an int32_t such that finite values have the obvious
  // order, -0 is ordered before 0, and -NaN and NaN appear at the beginning
  // and end of the ordering.
  auto signed_shape = max_value->shape();
  auto signed_value = computation->AddInstruction(
      HloInstruction::CreateBitcastConvert(signed_shape, value));
  auto compare_shape = ShapeUtil::ChangeElementType(signed_shape, PRED);
  HloInstruction* flipped_value;
  if (primitive_util::HasNegativeZero(value->shape().element_type())) {
    flipped_value = computation->AddInstruction(HloInstruction::CreateBinary(
        signed_shape, HloOpcode::kXor, max_value, signed_value));
  } else {
    // There is no -0 so min_denorm() must take its place, this is the same as
    // adding one to flipped_value.
    flipped_value = computation->AddInstruction(HloInstruction::CreateBinary(
        signed_shape, HloOpcode::kSubtract, min_value, signed_value));

    // NaN is the smallest value as it is negative.
    auto nan_bit_pattern = min_value;
    auto is_nan = computation->AddInstruction(HloInstruction::CreateCompare(
        compare_shape, signed_value, nan_bit_pattern,
        ComparisonDirection::kEq));
    flipped_value = computation->AddInstruction(HloInstruction::CreateTernary(
        signed_shape, HloOpcode::kSelect, is_nan, min_value, flipped_value));
  }
  auto is_negative = computation->AddInstruction(HloInstruction::CreateCompare(
      compare_shape, signed_value, zero, ComparisonDirection::kLt));
  return computation->AddInstruction(
      HloInstruction::CreateTernary(signed_shape, HloOpcode::kSelect,
                                    is_negative, flipped_value, signed_value));
}

bool ComparisonExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (HloCompareInstruction* compare =
          DynCast<HloCompareInstruction>(instruction)) {
    HloInstruction* lhs = instruction->operands()[0];
    if (compare->order() == Comparison::Order::kTotal &&
        primitive_util::IsFloatingPointType(lhs->shape().element_type())) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<HloInstruction*> ComparisonExpander::ExpandInstruction(
    HloInstruction* instruction) {
  CHECK_EQ(instruction->opcode(), HloOpcode::kCompare);
  HloCompareInstruction* compare =
      static_cast<HloCompareInstruction*>(instruction);
  CHECK(compare->order() == Comparison::Order::kTotal)
      << ComparisonOrderToString(compare->order());
  HloComputation* computation = instruction->parent();
  HloInstruction* lhs = instruction->operands()[0];
  HloInstruction* rhs = instruction->operands()[1];
  PrimitiveType compare_type = lhs->shape().element_type();
  CHECK(primitive_util::IsFloatingPointType(compare_type));
  if (auto do_upcast = absl::c_find_if(
          expand_via_upcast_,
          [compare_type](std::pair<PrimitiveType, PrimitiveType> upcast) {
            return upcast.first == compare_type;
          });
      do_upcast != expand_via_upcast_.end()) {
    CHECK(primitive_util::CastPreservesValues(do_upcast->first,
                                              do_upcast->second));
    compare_type = do_upcast->second;
    lhs = computation->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::ChangeElementType(lhs->shape(), compare_type), lhs));
    rhs = computation->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::ChangeElementType(rhs->shape(), compare_type), rhs));
  }

  int64_t bit_width = primitive_util::BitWidth(lhs->shape().element_type());
  PrimitiveType signed_type =
      primitive_util::SignedIntegralTypeForBitWidth(bit_width);
  auto signed_shape = ShapeUtil::ChangeElementType(lhs->shape(), signed_type);

  auto zero_value = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(signed_type)));
  zero_value = computation->AddInstruction(
      HloInstruction::CreateBroadcast(signed_shape, zero_value, {}));

  auto min_value = computation->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::MinValue(signed_shape.element_type())));
  min_value = computation->AddInstruction(
      HloInstruction::CreateBroadcast(signed_shape, min_value, {}));

  auto max_value = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MaxValue(signed_type)));
  max_value = computation->AddInstruction(
      HloInstruction::CreateBroadcast(signed_shape, max_value, {}));

  lhs = BitcastConvertFloatingPointToIntegral(computation, lhs, zero_value,
                                              min_value, max_value);
  rhs = BitcastConvertFloatingPointToIntegral(computation, rhs, zero_value,
                                              min_value, max_value);

  auto new_compare = computation->AddInstruction(HloInstruction::CreateCompare(
      instruction->shape(), lhs, rhs, compare->direction(),
      Comparison::Type::kSigned));

  VLOG(2) << "New comparison instruction for total order:"
          << new_compare->ToString();
  return new_compare;
}

}  // namespace xla

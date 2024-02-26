/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/value_range.h"

#include <optional>
#include <string>

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

std::optional<int64_t> Range::GetSingleSignedValue() const {
  if (!IsSingleValue()) {
    return std::nullopt;
  }
  return min_.GetSignedValue();
}

std::optional<int64_t> Range::GetSingleUnsignedValue() const {
  if (!IsSingleValue()) {
    return std::nullopt;
  }
  return min_.GetUnsignedValue();
}

std::string Range::ToString() const {
  if (IsEmpty()) {
    return std::string("Empty");
  }
  return absl::StrCat("min: ", min_.ToString(), " max: ", max_.ToString());
}

// Identify the value ranges of a scalar HLO with a integer type. It returns
// a range of values that the instruction can have.
Range RecursivelyIdentifyRange(
    const HloInstruction* instr,
    const absl::flat_hash_map<const HloInstruction*, Range>&
        predefined_ranges) {
  // Non scalar or non-integer HLO. Abort.
  if ((!instr->shape().IsInteger() && instr->shape().element_type() != PRED) ||
      instr->shape().dimensions_size() != 0) {
    return Range{};
  }
  VLOG(5) << "Computing Range for " << instr->ToString();
  auto it = predefined_ranges.find(instr);
  if (it != predefined_ranges.end()) {
    VLOG(5) << "Found range! " << it->second.max().GetSignedValue() << " "
            << it->second.min().GetSignedValue();
    return it->second;
  }
  switch (instr->opcode()) {
    case HloOpcode::kCompare: {
      VLOG(5) << "Handling Compare";
      Range lhs =
          RecursivelyIdentifyRange(instr->operand(0), predefined_ranges);
      Range rhs =
          RecursivelyIdentifyRange(instr->operand(1), predefined_ranges);
      VLOG(5) << "Returned Rhs: " << rhs.ToString()
              << " Lhs: " << lhs.ToString();
      // Only kLt supported right now.
      if (instr->comparison_direction() != ComparisonDirection::kLt) {
        return Range{};
      }
      if (lhs.max().lt(rhs.min())) {
        return Range{ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                     ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                     /*is_linear=*/true};
      }
      if (!lhs.min().lt(rhs.max())) {
        return Range{
            ConstantValue::GetZero(/*bitwidth=*/1, /*is_signed=*/false),
            ConstantValue::GetZero(/*bitwidth=*/1, /*is_signed=*/false),
            /*is_linear=*/true};
      }
      VLOG(5) << "Compare failed";
      VLOG(5) << "rhs max " << rhs.max().GetSignedValue() << " rhs min "
              << rhs.min().GetSignedValue() << " lhs max "
              << lhs.max().GetSignedValue() << " lhs min "
              << lhs.min().GetSignedValue();
      return Range{};
    }
    case HloOpcode::kConstant: {
      if (!instr->shape().IsInteger()) {
        return Range{};
      }
      VLOG(5) << "Handling Constant";
      const int64_t bitwidth =
          primitive_util::BitWidth(instr->shape().element_type());
      const bool is_signed =
          primitive_util::IsSignedIntegralType(instr->shape().element_type());
      if (is_signed) {
        const int64_t value = *instr->literal().GetFirstInteger();
        return Range{ConstantValue::GetSigned(value, bitwidth),
                     ConstantValue::GetSigned(value, bitwidth),
                     /*is_linear=*/true};
      }
      const uint64_t value = *instr->literal().GetFirstInteger();
      return Range{ConstantValue::GetUnsigned(value, bitwidth),
                   ConstantValue::GetUnsigned(value, bitwidth),
                   /*is_linear=*/true};
    }
    case HloOpcode::kAdd: {
      if (!instr->shape().IsInteger()) {
        return Range{};
      }
      VLOG(5) << "Handling Add";
      Range lhs =
          RecursivelyIdentifyRange(instr->operand(0), predefined_ranges);
      Range rhs =
          RecursivelyIdentifyRange(instr->operand(1), predefined_ranges);
      VLOG(5) << "Returned Rhs: " << rhs.ToString()
              << " Lhs: " << lhs.ToString();
      if (lhs.IsEmpty() || rhs.IsEmpty()) {
        return Range{};
      }
      ConstantValue min = lhs.min().add(rhs.min());
      ConstantValue max = lhs.max().add(rhs.max());
      if (max.lt(min)) {
        VLOG(5) << "Add wrapped";
        return Range{};
      }
      return Range{min, max, lhs.IsLinear() && rhs.IsLinear()};
    }
    case HloOpcode::kSelect: {
      VLOG(5) << "Handling Select";
      const HloInstruction* cmp = instr->operand(0);
      Range cmp_range = RecursivelyIdentifyRange(cmp, predefined_ranges);
      // Support only when the select has a constant value as condition.
      if (cmp_range.IsEmpty() || !cmp_range.IsSingleValue()) {
        VLOG(5) << "Select failed";
        return Range{};
      }
      if (cmp_range.GetSingleSignedValue() == 0) {
        return RecursivelyIdentifyRange(instr->operand(2), predefined_ranges);
      }
      return RecursivelyIdentifyRange(instr->operand(1), predefined_ranges);
    }
    case HloOpcode::kSubtract: {
      if (!instr->shape().IsInteger()) {
        return Range{};
      }
      VLOG(5) << "Handling Subtract";
      Range lhs =
          RecursivelyIdentifyRange(instr->operand(0), predefined_ranges);
      Range rhs =
          RecursivelyIdentifyRange(instr->operand(1), predefined_ranges);
      VLOG(5) << "Returned Rhs: " << rhs.ToString()
              << " Lhs: " << lhs.ToString();
      if (lhs.IsEmpty() || rhs.IsEmpty()) {
        return Range{};
      }
      ConstantValue min = lhs.min().sub(rhs.max());
      ConstantValue max = lhs.max().sub(rhs.min());
      if (max.lt(min)) {
        VLOG(5) << "Subtract wrapped";
        return Range{};
      }
      return Range{min, max, lhs.IsLinear() && rhs.IsLinear()};
    }
    default:
      break;
  }
  VLOG(5) << "Unsupported instruction: " << instr->ToString();
  return Range{};
}

}  // namespace xla

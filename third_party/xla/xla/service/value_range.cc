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

#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/constant_value.h"
#include "xla/service/hlo_value.h"

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
  if (IsSingleValue()) {
    return min_.ToString();
  }
  return absl::StrCat(
      "min: ", min_.ToString(), " max: ", max_.ToString(),
      " step: ", IsStepKnown() ? step_.value().ToString() : "Unknown");
}

std::optional<ConstantValue> FindStepForBinaryOp(const Range& lhs,
                                                 const Range& rhs) {
  if (!lhs.IsStepKnown() || !rhs.IsStepKnown()) {
    return std::nullopt;
  }
  if (lhs.IsSingleValue()) {
    return rhs.step();
  }
  if (rhs.IsSingleValue()) {
    return lhs.step();
  }
  if (lhs.step().eq(rhs.step())) {
    return lhs.step();
  }
  return std::nullopt;
}

// Identify the value ranges of a scalar HLO with a integer type. It returns
// a range of values that the instruction can have.
Range RecursivelyIdentifyRange(
    const HloInstruction* instr,
    const absl::flat_hash_map<const HloInstruction*, Range>& predefined_ranges,
    const HloAliasAnalysis* alias_analysis) {
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
  } else if (alias_analysis != nullptr) {
    auto value_set =
        alias_analysis->dataflow_analysis().GetFlattenedValueSet(instr);
    for (const auto& value : value_set.TakeValues()) {
      for (const HloPosition& position : value->positions()) {
        auto it = predefined_ranges.find(position.instruction);
        if (it != predefined_ranges.end()) {
          VLOG(5) << "Found range in defining instruction! "
                  << it->second.max().GetSignedValue() << " "
                  << it->second.min().GetSignedValue();
          return it->second;
        }
      }
    }
  }
  switch (instr->opcode()) {
    case HloOpcode::kCompare: {
      VLOG(5) << "Handling Compare";
      Range lhs = RecursivelyIdentifyRange(instr->operand(0), predefined_ranges,
                                           alias_analysis);
      Range rhs = RecursivelyIdentifyRange(instr->operand(1), predefined_ranges,
                                           alias_analysis);
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
      if (instr->shape().element_type() == PRED &&
          instr->shape().dimensions_size() == 0) {
        if (instr->literal().IsAll(true)) {
          return Range{
              ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
              ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
              /*is_linear=*/true};
        }
        return Range{
            ConstantValue::GetZero(/*bitwidth=*/1, /*is_signed=*/false),
            ConstantValue::GetZero(/*bitwidth=*/1, /*is_signed=*/false),
            /*is_linear=*/true};
      }
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
                     ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                     /*is_linear=*/true};
      }
      const uint64_t value = *instr->literal().GetFirstInteger();
      return Range{ConstantValue::GetUnsigned(value, bitwidth),
                   ConstantValue::GetUnsigned(value, bitwidth),
                   ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                   /*is_linear=*/true};
    }
    case HloOpcode::kAdd: {
      if (!instr->shape().IsInteger()) {
        return Range{};
      }
      VLOG(5) << "Handling Add";
      Range lhs = RecursivelyIdentifyRange(instr->operand(0), predefined_ranges,
                                           alias_analysis);
      Range rhs = RecursivelyIdentifyRange(instr->operand(1), predefined_ranges,
                                           alias_analysis);
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
      return Range{min, max, FindStepForBinaryOp(lhs, rhs),
                   lhs.IsLinear() && rhs.IsLinear()};
    }
    case HloOpcode::kMultiply: {
      if (!instr->shape().IsInteger()) {
        return Range{};
      }
      VLOG(5) << "Handling Multiply";
      Range lhs = RecursivelyIdentifyRange(instr->operand(0), predefined_ranges,
                                           alias_analysis);
      Range rhs = RecursivelyIdentifyRange(instr->operand(1), predefined_ranges,
                                           alias_analysis);
      VLOG(5) << "Returned Rhs: " << rhs.ToString()
              << " Lhs: " << lhs.ToString();
      if (lhs.IsEmpty() || rhs.IsEmpty()) {
        return Range{};
      }
      // We only handle multiplication of a single value with a range.
      if (!lhs.IsSingleValue() && !rhs.IsSingleValue()) {
        return Range{};
      }
      ConstantValue single_value = lhs.IsSingleValue() ? lhs.min() : rhs.min();
      ConstantValue min = lhs.IsSingleValue() ? rhs.min().mul(single_value)
                                              : lhs.min().mul(single_value);
      ConstantValue max = lhs.IsSingleValue() ? rhs.max().mul(single_value)
                                              : lhs.max().mul(single_value);
      return Range{min, max, single_value, lhs.IsLinear() && rhs.IsLinear()};
    }
    case HloOpcode::kSelect: {
      VLOG(5) << "Handling Select: " << instr->ToString();
      const HloInstruction* cmp = instr->operand(0);
      Range cmp_range =
          RecursivelyIdentifyRange(cmp, predefined_ranges, alias_analysis);
      // Support only when the select has a constant value as condition.
      if (cmp_range.IsEmpty() || !cmp_range.IsSingleValue()) {
        VLOG(5) << "Select failed";
        return Range{};
      }
      if (cmp_range.GetSingleSignedValue() == 0) {
        return RecursivelyIdentifyRange(instr->operand(2), predefined_ranges,
                                        alias_analysis);
      }
      return RecursivelyIdentifyRange(instr->operand(1), predefined_ranges,
                                      alias_analysis);
    }
    case HloOpcode::kSubtract: {
      if (!instr->shape().IsInteger()) {
        return Range{};
      }
      VLOG(5) << "Handling Subtract";
      Range lhs = RecursivelyIdentifyRange(instr->operand(0), predefined_ranges,
                                           alias_analysis);
      Range rhs = RecursivelyIdentifyRange(instr->operand(1), predefined_ranges,
                                           alias_analysis);
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
      return Range{min, max, FindStepForBinaryOp(lhs, rhs),
                   lhs.IsLinear() && rhs.IsLinear()};
    }
    default:
      break;
  }
  VLOG(5) << "Unsupported instruction: " << instr->ToString();
  return Range{};
}

}  // namespace xla

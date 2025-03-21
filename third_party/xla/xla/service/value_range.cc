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
#include <vector>

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
      "min: ", min_.ToString(),
      " max: ", IsBounded() ? max_.value().ToString() : "Unknown",
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
  if (lhs.step()->eq(rhs.step().value())) {
    return lhs.step();
  }
  return std::nullopt;
}

// Helper function that updates the known_ranges map and returns the range.
Range RecordAndReturnRange(
    const Range& range, const HloInstruction* instr,
    absl::flat_hash_map<const HloInstruction*, Range>& known_ranges) {
  known_ranges[instr] = range;
  VLOG(5) << "Computed range for: " << instr->name() << " -> "
          << range.ToString();
  return range;
}

// Identify the value ranges of a scalar HLO with a integer type. It returns
// a range of values that the instruction can have.
Range RecursivelyIdentifyRange(
    const HloInstruction* instr,
    absl::flat_hash_map<const HloInstruction*, Range>& known_ranges,
    const HloAliasAnalysis* alias_analysis) {
  // Non scalar or non-integer HLO. Abort.
  if ((!instr->shape().AreAllLeavesIntegers() &&
       instr->shape().element_type() != PRED) ||
      instr->shape().dimensions_size() != 0) {
    return Range{};
  }
  VLOG(5) << "Computing Range for " << instr->ToString();
  auto it = known_ranges.find(instr);
  if (it != known_ranges.end()) {
    VLOG(5) << "Found range: " << it->second.ToString();
    return it->second;
  } else if (alias_analysis != nullptr) {
    auto value_set =
        alias_analysis->dataflow_analysis().GetFlattenedValueSet(instr);
    for (const auto& value : value_set.TakeValues()) {
      for (const HloPosition& position : value->positions()) {
        auto it = known_ranges.find(position.instruction);
        if (it != known_ranges.end()) {
          VLOG(5) << "Found range in defining instruction: "
                  << it->second.ToString();
          return it->second;
        }
      }
    }
  }
  switch (instr->opcode()) {
    case HloOpcode::kGetTupleElement: {
      if (alias_analysis != nullptr) {
        auto value_set =
            alias_analysis->dataflow_analysis().GetFlattenedValueSet(instr);
        std::vector<const HloValue*> values = value_set.TakeValues();
        if (values.size() != 1) {
          VLOG(5) << "Ambiguous value set";
          return Range{};
        }
        HloInstruction* defining_instruction =
            values.at(0)->defining_instruction();
        if (defining_instruction != nullptr) {
          return RecursivelyIdentifyRange(defining_instruction, known_ranges,
                                          alias_analysis);
        }
      }
      return Range{};
    }
    case HloOpcode::kCompare: {
      VLOG(5) << "Handling Compare";
      Range lhs = RecursivelyIdentifyRange(instr->operand(0), known_ranges,
                                           alias_analysis);
      Range rhs = RecursivelyIdentifyRange(instr->operand(1), known_ranges,
                                           alias_analysis);
      VLOG(5) << "Returned Rhs: " << rhs.ToString()
              << " Lhs: " << lhs.ToString();
      // Only kLt supported right now.
      if (instr->comparison_direction() != ComparisonDirection::kLt) {
        return Range{};
      }
      if (lhs.IsBounded() && lhs.max()->lt(rhs.min())) {
        return RecordAndReturnRange(
            Range{ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                  ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                  /*is_linear=*/true},
            instr, known_ranges);
      }
      if (rhs.IsBounded() && !lhs.min().lt(rhs.max().value())) {
        return RecordAndReturnRange(
            Range{ConstantValue::GetZero(/*bitwidth=*/1, /*is_signed=*/false),
                  ConstantValue::GetZero(/*bitwidth=*/1, /*is_signed=*/false),
                  /*is_linear=*/true},
            instr, known_ranges);
      }
      return Range{};
    }
    case HloOpcode::kConstant: {
      if (instr->shape().element_type() == PRED &&
          instr->shape().dimensions_size() == 0) {
        if (instr->literal().IsAll(true)) {
          return RecordAndReturnRange(
              Range{ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                    ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                    /*is_linear=*/true},
              instr, known_ranges);
        }
        return RecordAndReturnRange(
            Range{ConstantValue::GetZero(/*bitwidth=*/1, /*is_signed=*/false),
                  ConstantValue::GetZero(/*bitwidth=*/1, /*is_signed=*/false),
                  /*is_linear=*/true},
            instr, known_ranges);
      }
      if (!instr->shape().AreAllLeavesIntegers()) {
        return Range{};
      }
      VLOG(5) << "Handling Constant";
      const int64_t bitwidth =
          primitive_util::BitWidth(instr->shape().element_type());
      const bool is_signed =
          primitive_util::IsSignedIntegralType(instr->shape().element_type());
      if (is_signed) {
        const int64_t value = *instr->literal().GetFirstInteger();
        return RecordAndReturnRange(
            Range{ConstantValue::GetSigned(value, bitwidth),
                  ConstantValue::GetSigned(value, bitwidth),
                  ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                  /*is_linear=*/true},
            instr, known_ranges);
      }
      const uint64_t value = *instr->literal().GetFirstInteger();
      return RecordAndReturnRange(
          Range{ConstantValue::GetUnsigned(value, bitwidth),
                ConstantValue::GetUnsigned(value, bitwidth),
                ConstantValue::GetOne(/*bitwidth=*/1, /*is_signed=*/false),
                /*is_linear=*/true},
          instr, known_ranges);
    }
    case HloOpcode::kAdd: {
      if (!instr->shape().AreAllLeavesIntegers()) {
        return Range{};
      }
      VLOG(5) << "Handling Add";
      Range lhs = RecursivelyIdentifyRange(instr->operand(0), known_ranges,
                                           alias_analysis);
      Range rhs = RecursivelyIdentifyRange(instr->operand(1), known_ranges,
                                           alias_analysis);
      VLOG(5) << "Returned Rhs: " << rhs.ToString()
              << " Lhs: " << lhs.ToString();
      if (lhs.IsEmpty() || rhs.IsEmpty()) {
        return Range{};
      }
      ConstantValue min = lhs.min().add(rhs.min());
      std::optional<ConstantValue> step = FindStepForBinaryOp(lhs, rhs);
      if (lhs.IsBounded() && rhs.IsBounded()) {
        ConstantValue max = lhs.max()->add(rhs.max().value());
        if (max.lt(min)) {
          VLOG(5) << "Add wrapped";
          return Range{};
        }
        return RecordAndReturnRange(
            Range{min, max, step, lhs.IsLinear() && rhs.IsLinear()}, instr,
            known_ranges);
      }
      return RecordAndReturnRange(
          Range{min, std::nullopt, step, lhs.IsLinear() && rhs.IsLinear()},
          instr, known_ranges);
    }
    case HloOpcode::kMultiply: {
      if (!instr->shape().AreAllLeavesIntegers()) {
        return Range{};
      }
      VLOG(5) << "Handling Multiply";
      Range lhs = RecursivelyIdentifyRange(instr->operand(0), known_ranges,
                                           alias_analysis);
      Range rhs = RecursivelyIdentifyRange(instr->operand(1), known_ranges,
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
      Range operand_range = lhs.IsSingleValue() ? rhs : lhs;
      // When multiplying with a constant, min, max, and step are all
      // multiplied by the single value.
      ConstantValue min = operand_range.min().mul(single_value);
      if (operand_range.IsBounded()) {
        ConstantValue max = operand_range.max()->mul(single_value);
        if (!operand_range.IsStepKnown()) {
          return RecordAndReturnRange(Range{min, max, operand_range.IsLinear()},
                                      instr, known_ranges);
        }
        ConstantValue step = operand_range.step()->mul(single_value);
        return RecordAndReturnRange(
            Range{min, max, step, operand_range.IsLinear()}, instr,
            known_ranges);
      }
      if (!operand_range.IsStepKnown()) {
        return RecordAndReturnRange(
            Range{min, std::nullopt, operand_range.IsLinear()}, instr,
            known_ranges);
      }
      ConstantValue step = operand_range.step()->mul(single_value);
      return RecordAndReturnRange(
          Range{min, std::nullopt, step, operand_range.IsLinear()}, instr,
          known_ranges);
    }
    case HloOpcode::kSelect: {
      VLOG(5) << "Handling Select: " << instr->ToString();
      const HloInstruction* cmp = instr->operand(0);
      Range cmp_range =
          RecursivelyIdentifyRange(cmp, known_ranges, alias_analysis);
      // Support only when the select has a constant value as condition.
      if (cmp_range.IsEmpty() || !cmp_range.IsSingleValue()) {
        VLOG(5) << "Select failed";
        return Range{};
      }
      if (cmp_range.GetSingleSignedValue() == 0) {
        return RecordAndReturnRange(
            RecursivelyIdentifyRange(instr->operand(2), known_ranges,
                                     alias_analysis),
            instr, known_ranges);
      }
      return RecordAndReturnRange(
          RecursivelyIdentifyRange(instr->operand(1), known_ranges,
                                   alias_analysis),
          instr, known_ranges);
    }
    case HloOpcode::kSubtract: {
      if (!instr->shape().AreAllLeavesIntegers()) {
        return Range{};
      }
      VLOG(5) << "Handling Subtract";
      Range lhs = RecursivelyIdentifyRange(instr->operand(0), known_ranges,
                                           alias_analysis);
      Range rhs = RecursivelyIdentifyRange(instr->operand(1), known_ranges,
                                           alias_analysis);
      VLOG(5) << "Returned Rhs: " << rhs.ToString()
              << " Lhs: " << lhs.ToString();
      if (lhs.IsEmpty() || rhs.IsEmpty()) {
        return Range{};
      }
      if (lhs.IsBounded() && rhs.IsBounded()) {
        ConstantValue min = lhs.min().sub(rhs.max().value());
        ConstantValue max = lhs.max()->sub(rhs.min());
        if (max.lt(min)) {
          VLOG(5) << "Subtract wrapped";
          return Range{};
        }
        return RecordAndReturnRange(
            Range{min, max, FindStepForBinaryOp(lhs, rhs),
                  lhs.IsLinear() && rhs.IsLinear()},
            instr, known_ranges);
      } else if (lhs.IsBounded()) {  // bounded - unbounded -> Empty range
        VLOG(5) << "Subtract unbounded from bounded is not represntable with a "
                   "range";
        return Range{};
      } else {  // unbounded - bounded -> Unbounded range
        ConstantValue min = lhs.min().sub(rhs.max().value());
        return RecordAndReturnRange(
            Range{min, std::nullopt, FindStepForBinaryOp(lhs, rhs),
                  lhs.IsLinear() && rhs.IsLinear()},
            instr, known_ranges);
      }
    }
    default:
      break;
  }
  VLOG(5) << "Unsupported instruction: " << instr->ToString();
  return Range{};
}

}  // namespace xla

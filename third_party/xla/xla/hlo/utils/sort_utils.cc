/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/utils/sort_utils.h"

#include <cstdint>
#include <utility>

#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"

namespace xla {
namespace {

namespace m = match;

// Returns whether the given instruction is a floating-point constant literal
// representing NaN.
bool MatchConstNan(const HloInstruction* op) {
  const auto* const_nan = DynCast<HloConstantInstruction>(op);
  if (const_nan == nullptr) {
    return false;
  }
  return const_nan->literal().GetAsString({}) == "nan";
}

// Matches the HLO pattern used to ensure NumPy sort order (canonicalizing
// zeros and NaNs). This is how JAX lowers `lax.sort` to HLO comparators:
//   Select(Compare(param, param, NE), NaN,
//                                     Select(Compare(param, 0, EQ), 0, param))
// Returns the parameter number if matched, or -1 otherwise.
int64_t MatchCanonicalizedZerosAndNans(const HloInstruction* select) {
  const HloInstruction* param_nan1 = nullptr;
  const HloInstruction* param_nan2 = nullptr;
  const HloInstruction* param_zero = nullptr;
  const HloInstruction* param_fallback = nullptr;
  const HloInstruction* maybe_const_nan = nullptr;
  if (!Match(
          select,
          m::Select(
              m::Compare(m::Parameter(&param_nan1), m::Parameter(&param_nan2))
                  .WithComparisonDirection(ComparisonDirection::kNe),
              m::Constant(&maybe_const_nan),
              m::Select(m::Compare(m::Parameter(&param_zero),
                                   m::ConstantEffectiveScalar(0))
                            .WithComparisonDirection(ComparisonDirection::kEq),
                        m::ConstantEffectiveScalar(0),
                        m::Parameter(&param_fallback)))) ||
      param_nan1 != param_nan2 || param_nan1 != param_zero ||
      param_nan1 != param_fallback) {
    return -1;
  }
  if (!MatchConstNan(maybe_const_nan)) {
    return -1;
  }
  return param_nan1->parameter_number();
}

// Matches an operand of a NumPy sort comparator. This handles both:
// 1. Pre-ComparisonExpander: Select(isnan, nan, Select(is_zero, 0, param))
// 2. Post-ComparisonExpander: Select(is_neg, xor/sub,
//                                            bitcast(select_canonical))
// Returns the parameter number if matched, or -1 otherwise.
int64_t MatchNumpySortParameter(const HloInstruction* operand) {
  if (operand == nullptr) {
    return -1;
  }

  // 1. Direct canonicalized select (pre-ComparisonExpander).
  int64_t param_idx = MatchCanonicalizedZerosAndNans(operand);
  if (param_idx != -1) {
    return param_idx;
  }

  // 2. Expanded TotalOrder integer mapping (post-ComparisonExpander):
  // Select(Compare(bitcast, 0, LT), Xor(max_val, bitcast), bitcast)
  if (operand->opcode() != HloOpcode::kSelect) {
    return -1;
  }

  const HloInstruction* bitcast = operand->operand(2);
  if (bitcast->opcode() != HloOpcode::kBitcastConvert) {
    return -1;
  }

  const HloInstruction* cond = operand->operand(0);
  if (cond->opcode() != HloOpcode::kCompare) {
    return -1;
  }
  auto* compare_cond = Cast<HloCompareInstruction>(cond);
  if (compare_cond->comparison_direction() != ComparisonDirection::kLt ||
      compare_cond->operand(0) != bitcast) {
    return -1;
  }

  const HloInstruction* flipped = operand->operand(1);
  if (flipped->opcode() != HloOpcode::kXor &&
      flipped->opcode() != HloOpcode::kSubtract) {
    return -1;
  }
  if (flipped->operand(0) != bitcast && flipped->operand(1) != bitcast) {
    return -1;
  }

  return MatchCanonicalizedZerosAndNans(bitcast->operand(0));
}

}  // namespace

std::pair<int64_t, int64_t> MatchNumpySortComparator(
    const HloCompareInstruction* compare) {
  if (compare == nullptr) {
    return {-1, -1};
  }
  return {MatchNumpySortParameter(compare->operand(0)),
          MatchNumpySortParameter(compare->operand(1))};
}

std::pair<int64_t, int64_t> MatchSimpleSortComparator(
    const HloCompareInstruction* compare) {
  if (compare == nullptr) {
    return {-1, -1};
  }
  const auto* param0 = DynCast<HloParameterInstruction>(compare->operand(0));
  const auto* param1 = DynCast<HloParameterInstruction>(compare->operand(1));
  if (param0 && param1) {
    return {param0->parameter_number(), param1->parameter_number()};
  }
  return {-1, -1};
}

}  // namespace xla

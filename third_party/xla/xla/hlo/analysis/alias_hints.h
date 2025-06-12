/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_ALIAS_HINTS_H_
#define XLA_HLO_ANALYSIS_ALIAS_HINTS_H_

#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"

namespace xla {
class AliasHints {
 public:
  AliasHints() = default;
  virtual ~AliasHints() = default;

  // Must-alias hints. HloDataflowAnalysis::GetInPlaceInputOutputPairs will
  // check first whether this function returns true.
  virtual bool IsPotentialInPlaceOperation(const HloInstruction* hlo) const;

  // May-alias hints. If an empty optional is returned, default rules in
  // HloDataflowAnalysis are used. The first parameter of the function should be
  // the instruction, the second parameter should be an operand of the
  // instruction. The third parameter should be the output index of the
  // instruction.
  virtual std::optional<bool> CanShareBuffer(
      const HloInstruction* instr, const HloInstruction* operand,
      const ShapeIndex& user_index) const;

  // Whether an instruction defines a new value.
  //
  // The first parameter is the instruction and the second parameter is the
  // output index. If an empty optional is used, default rules are used. If a
  // ForwardedOperand object is returned, the value at the corresponding
  // operand's index is used for the output, overriding all default logic.
  struct ForwardedOperand {
    int64_t operand_number;
    ShapeIndex operand_index;
  };
  virtual std::optional<ForwardedOperand> ForwardsValue(
      const HloInstruction* instr, const ShapeIndex& index) const;
};
}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_ALIAS_HINTS_H_

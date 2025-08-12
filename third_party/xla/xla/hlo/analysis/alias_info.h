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

#ifndef XLA_HLO_ANALYSIS_ALIAS_INFO_H_
#define XLA_HLO_ANALYSIS_ALIAS_INFO_H_

#include <optional>
#include <utility>
#include <vector>

#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"

namespace xla {
class AliasInfo {
 public:
  AliasInfo() = default;
  virtual ~AliasInfo() = default;

  // Returns whether the buffer assigned to `operand` at ShapeIndex
  // `operand_index` needs to share the buffer assigned to `user` at ShapeIndex
  // `user_index`. This is a wrapper around GetInPlaceInputOutputPairs() that
  // checks whether the given operand and user appear as part of what
  // GetInPlaceInputOutputPairs() returns.
  bool MustAlias(const HloInstruction* operand, const ShapeIndex& operand_index,
                 const HloInstruction* user,
                 const ShapeIndex& user_index) const;

  // Returns the pairs of inputs and outputs that must share the same buffer,
  // according to the aliasing rules for the `user` instruction.
  //
  // This function only considers array values as inputs and outputs, so
  // when tuples are present it "sees through" to the array values inside. The
  // HloUse describing the input parameter contains not only the operand number
  // but also a shape index describing its position inside a nested tuple shape
  // (if any). Similarly, the output parameter is described by a shape index
  // into the nested tuple shape (if any) of the output value.
  //
  // For example, for this hypothetical op:
  //   %foo = (f32[1], (f32[2], f32[3]))
  //              op((f32[4], f32[5]) %arg0, f32[6] %arg1)
  //
  // ... the results can include any of the 3 * 3 = 9 possible pairs of
  // input and output arrays.
  // TODO(b/424109294): Move the implementation from HloDataflowAnalysis to
  // here. Currently, this does not implement any default must-alias rules.
  std::vector<std::pair<HloOperandIndex, ShapeIndex>>
  GetInPlaceInputOutputPairs(const HloInstruction* user) const;

  // Backend-specific may-alias hint. If an empty optional is returned, the
  // default rules in HloDataflowAnalysis are used. `operand` should be an
  // operand of `user`. `operand_index` should be the output index of `operand`,
  // `user_index` should be the output index of `user`.
  virtual std::optional<bool> MayAlias(const HloInstruction* operand,
                                       const ShapeIndex& operand_index,
                                       const HloInstruction* user,
                                       const ShapeIndex& user_index) const {
    return std::nullopt;
  }

 protected:
  // Backend-specific hook that allows to deviate from the default must-alias
  // rules in GetInPlaceInputOutputPairs(). If an empty optional is returned,
  // the default rules are used. Otherwise, the return value of this function is
  // used.
  virtual std::optional<std::vector<std::pair<HloOperandIndex, ShapeIndex>>>
  GetNonDefaultInPlaceInputOutputPairs(const HloInstruction* user) const {
    return std::nullopt;
  }
};
}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_ALIAS_INFO_H_

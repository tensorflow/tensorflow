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

#ifndef XLA_HLO_ANALYSIS_HLO_OPERAND_INDEX_H_
#define XLA_HLO_ANALYSIS_HLO_OPERAND_INDEX_H_

#include <cstdint>
#include <functional>
#include <tuple>

#include "xla/shape_util.h"

namespace xla {

// Identifies one array input of an HloInstruction. It bundles the operand
// number with the ShapeIndex for indexing into the operand array.
struct HloOperandIndex {
  using MyTuple = std::tuple<int64_t, const ShapeIndex&>;

  template <typename H>
  friend H AbslHashValue(H h, const HloOperandIndex& hlo_operand_index) {
    return H::combine(std::move(h), hlo_operand_index.ToTuple());
  }

  friend bool operator==(const HloOperandIndex& lhs,
                         const HloOperandIndex& rhs) {
    return lhs.ToTuple() == rhs.ToTuple();
  }

  bool operator!=(const HloOperandIndex& other) const {
    return !(*this == other);
  }

  MyTuple ToTuple() const {
    return std::make_tuple(operand_number, std::cref(operand_index));
  }

  // The operand number in which the array value appears.
  int64_t operand_number;

  // The shape index within the operand in which the array value appears.
  ShapeIndex operand_index;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_OPERAND_INDEX_H_

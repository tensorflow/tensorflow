/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_IR_HLO_INSTRUCTION_UTILS_H_
#define XLA_HLO_IR_HLO_INSTRUCTION_UTILS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"

namespace xla {
namespace hlo_instruction_utils {
// Returns true if the given HLO is a slice operation which has a unit stride in
// all dimensions.
bool IsUnstridedSlice(const HloInstruction* hlo);

// Checks that all instruction operands have the same bitwidth as its output.
bool KeepsBitwidth(const HloInstruction&);

// Adds or updates the attributes for an instruction. If the attribute is
// already present, then it is overwritten. Otherwise, this is added as another
// attribute.
void AddOrUpdateVectorOfPairsAsAttribute(
    HloInstruction* instr, std::string attr_name,
    std::vector<std::pair<int64_t, int64_t>> intervals);

// Returns the nesting depth in computations from the top-level computation of
// `hlo`. i.e. 0 = in the top-level computation, ...
int32_t NestingDepth(const HloInstruction* hlo);

namespace async {

// Utilities for async instructions.

// Determines if the operands and output of the async
// instruction is fully bound at the given shape
// index, which is empty by default.
// Returns an error if the index is invalid, or index does not start with 0 or
// 1.
absl::StatusOr<bool> AreOperandsAndOutputFullyBound(
    const HloInstruction* async_op, const ShapeIndex& index = {});
}  // namespace async

}  // namespace hlo_instruction_utils
}  // namespace xla

#endif  // XLA_HLO_IR_HLO_INSTRUCTION_UTILS_H_

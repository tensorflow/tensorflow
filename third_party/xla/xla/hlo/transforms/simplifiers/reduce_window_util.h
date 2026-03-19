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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCE_WINDOW_UTIL_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCE_WINDOW_UTIL_H_

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace reduce_window_util {

// Transform reduce-win(x) ->
//   if rank(x) == 1:
//   then: reshape_r2_r1(reduce-win(reshape_r1_r2(x)))
//   else: no change
absl::Status Replace1DReduceWindowWithReshape(
    HloReduceWindowInstruction* reduce_window);

// Returns the tuple element shape at given index.
// Wrapper for ShapeUtil::GetTupleElementShape.
Shape ShapeAtIndex(const Shape& shape, const ShapeIndex& shape_index);

// Creates a get tuple element instruction and returns it.
HloInstruction* GetAtIndex(HloInstruction* hlo, const ShapeIndex& shape_index);

}  // namespace reduce_window_util
}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_REDUCE_WINDOW_UTIL_H_

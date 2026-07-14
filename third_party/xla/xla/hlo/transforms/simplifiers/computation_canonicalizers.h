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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_COMPUTATION_CANONICALIZERS_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_COMPUTATION_CANONICALIZERS_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"

namespace xla {

// This function moves kParameter and kConstant instructions in a computation to
// the beginning of the computation. This simplifies other transformations like
// the construction of command buffer computations because we don't need to deal
// with parameters and constants that have users outside of a command buffer.
// Returns true if there is a change in the order of instructions, false
// otherwise.
absl::StatusOr<bool> MoveParametersAndConstantsToFront(HloComputation&);

// Moves GetTupleElement instructions to right after the instruction that
// produces the tuple. Returns whether the computation was changed. This is run,
// for instance, before command buffer scheduling.
absl::StatusOr<bool> MoveGTEsRightAfterTupleDefinition(HloComputation&);

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_COMPUTATION_CANONICALIZERS_H_

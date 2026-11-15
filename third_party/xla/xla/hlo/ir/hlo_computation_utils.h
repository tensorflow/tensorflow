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

#ifndef XLA_HLO_IR_HLO_COMPUTATION_UTILS_H_
#define XLA_HLO_IR_HLO_COMPUTATION_UTILS_H_

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::hlo_computation_utils {

// Clones `computation` and adds the clone as an embedded computation to its
// parent module. If `computation` is scheduled in its parent module's schedule,
// the cloned computation is also scheduled with the corresponding cloned
// instruction sequence. If `suffix` is empty, uses `HloComputation::Clone`'s
// default suffix.
// Precondition: `computation` and `computation->parent()` must be non-null.
HloComputation* CloneComputation(HloComputation* computation,
                                 absl::string_view suffix = "");

// Specializes computations on the entry computation's execution thread that are
// reachable from the entry computation via both pure ordinary call paths and
// paths containing at least one special call, ensuring each reachable
// computation has at most two versions:
// - Ordinary version (original): all call paths from the entry computation
//   consist solely of calls where `is_special_call` is false.
// - Special version (cloned with `suffix`): every call
//   path from the entry computation contains at least one call where
//   `is_special_call` is true.
// Returns true if any computation was cloned.
// Precondition: `module` must be non-null.
bool SpecializeComputations(
    HloModule* module,
    absl::FunctionRef<bool(const HloInstruction*)> is_special_call,
    absl::string_view suffix = "");

}  // namespace xla::hlo_computation_utils

#endif  // XLA_HLO_IR_HLO_COMPUTATION_UTILS_H_

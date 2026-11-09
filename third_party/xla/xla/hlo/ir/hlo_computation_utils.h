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

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"

namespace xla::hlo_computation_utils {

// Clones `computation` and adds the clone as an embedded computation to its
// parent module. If `computation` is scheduled in its parent module's schedule,
// the cloned computation is also scheduled with the corresponding cloned
// instruction sequence. If `suffix` is empty, uses `HloComputation::Clone`'s
// default suffix.
// Precondition: `computation` and `computation->parent()` must be non-null.
HloComputation* CloneComputation(HloComputation* computation,
                                 absl::string_view suffix = "");

}  // namespace xla::hlo_computation_utils

#endif  // XLA_HLO_IR_HLO_COMPUTATION_UTILS_H_

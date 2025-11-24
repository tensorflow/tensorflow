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

#ifndef XLA_HLO_UTILS_HLO_LONGEST_PREFIX_H_
#define XLA_HLO_UTILS_HLO_LONGEST_PREFIX_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace hlo_longest_prefix {

template <typename Visitor>
absl::Status VisitInstAndCalledButNotOperands(Visitor& visitor,
                                              const HloInstruction& inst) {
  // Visit the given instruction, and the things it calls, but not its operands.
  TF_RETURN_IF_ERROR(visitor.DefaultAction(&inst));
  for (const HloComputation* called : inst.called_computations()) {
    const HloInstruction* const root = called->root_instruction();
    TF_RETURN_IF_ERROR(root->Accept(&visitor, /*call_finish_visit=*/false,
                                    /*ignore_control_predecessors=*/true,
                                    /*cross_computation=*/true));
  }
  return absl::OkStatus();
}

absl::string_view GetLongestOpNamePrefix(
    const HloInstruction& inst, bool ignore_malformed_op_names = false);

absl::string_view GetLongestOpNamePrefix(
    const HloModule& mod, bool ignore_malformed_op_names = false);

}  // namespace hlo_longest_prefix
}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_LONGEST_PREFIX_H_

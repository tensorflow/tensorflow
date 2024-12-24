/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/op_expander_pass.h"

#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<bool> OpExpanderPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> matching_instructions;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    absl::c_copy_if(computation->MakeInstructionPostOrder(),
                    std::back_inserter(matching_instructions),
                    [&](HloInstruction* inst) {
                      return InstructionMatchesPattern(inst) &&
                             (!extra_filter_ || extra_filter_(inst));
                    });
  }

  for (HloInstruction* inst : matching_instructions) {
    TF_ASSIGN_OR_RETURN(HloInstruction * expanded_root,
                        ExpandInstruction(inst));
    if (expanded_root == nullptr) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool changed,
                        inst->parent()->ReplaceInstruction(
                            inst, expanded_root, preserve_sharding_,
                            relay_control_dependency_));
    DCHECK(changed);
  }

  return !matching_instructions.empty();
}
}  // namespace xla

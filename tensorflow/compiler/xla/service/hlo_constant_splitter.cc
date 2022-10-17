/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/compiler/xla/service/hlo_constant_splitter.h"

namespace xla {

StatusOr<bool> HloConstantSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kConstant) {
        continue;
      }
      auto users = instruction->users();
      for (int i = 1; i < users.size(); ++i) {
        HloInstruction* user = users[i];
        HloInstruction* constant =
            computation->AddInstruction(instruction->Clone(""));
        VLOG(4) << "Replacing " << instruction->name() << " with "
                << constant->name() << " on user " << user->name();
        TF_RETURN_IF_ERROR(instruction->ReplaceUseWith(user, constant));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla

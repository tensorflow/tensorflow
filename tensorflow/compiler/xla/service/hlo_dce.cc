/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_dce.h"

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<bool> HloDCE::Run(HloModule* module) {
  bool changed = false;

  for (auto& computation : module->computations()) {
    std::unordered_set<HloInstruction*> live_instructions;
    TF_RETURN_IF_ERROR(computation->root_instruction()->Accept(
        [&live_instructions](HloInstruction* instruction) {
          live_instructions.insert(instruction);
          return Status::OK();
        }));

    // Remove any dead roots and their dead transitive operands. Collect them
    // into a separate list first to avoid problems with iterating through the
    // computation's instruction while simultaneously removing instructions.
    std::vector<HloInstruction*> dead_roots;
    for (auto& instruction : computation->instructions()) {
      if (instruction->user_count() == 0 &&
          live_instructions.count(instruction.get()) == 0 &&
          computation->IsRemovable(instruction.get())) {
        dead_roots.push_back(instruction.get());
      }
    }

    for (HloInstruction* dead_root : dead_roots) {
      TF_RETURN_IF_ERROR(
          computation->RemoveInstructionAndUnusedOperands(dead_root));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla

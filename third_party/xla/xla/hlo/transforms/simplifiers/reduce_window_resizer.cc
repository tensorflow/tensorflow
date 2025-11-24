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

#include "xla/hlo/transforms/simplifiers/reduce_window_resizer.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> ReduceWindowResizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const auto& computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      HloReduceWindowInstruction* reduce_window =
          DynCast<HloReduceWindowInstruction>(instruction);
      if (!reduce_window) {
        continue;
      }

      if (reduce_window->inputs().front()->shape().dimensions().size() != 1) {
        continue;
      }
      TF_RETURN_IF_ERROR(
          reduce_window_util::Replace1DReduceWindowWithReshape(reduce_window));

      changed = true;
    }
  }
  return changed;
}

}  // namespace xla

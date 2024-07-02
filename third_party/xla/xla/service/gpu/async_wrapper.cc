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

#include "xla/service/gpu/async_wrapper.h"

#include <algorithm>
#include <deque>
#include <iterator>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"

namespace xla::gpu {

absl::StatusOr<bool> AsyncWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  std::deque<HloComputation*> computations;
  computations.push_back(module->entry_computation());
  while (!computations.empty()) {
    HloComputation* computation = computations.front();
    computations.pop_front();

    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (predicate_(instruction)) {
        // If the predicate matches, then wrap the instructions in async blocks.
        TF_RETURN_IF_ERROR(
            computation
                ->CreateAsyncInstructions(instruction,
                                          {ShapeUtil::MakeScalarShape(U32)})
                .status());
        changed = true;
        continue;
      }

      // Otherwise, follow any `calls` to discover other instructions that can
      // potentially be made async.
      if (instruction->opcode() == HloOpcode::kCall) {
        std::copy(instruction->called_computations().begin(),
                  instruction->called_computations().end(),
                  std::back_inserter(computations));
      }
    }
  }
  return changed;
}

}  // namespace xla::gpu

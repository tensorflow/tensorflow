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

#include "xla/backends/cpu/transforms/embedded_while_loop_unroller.h"

#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/while_loop_unroller.h"

namespace xla::cpu {
namespace {

std::vector<HloInstruction*> EmbeddedWhileLoops(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> loops;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (!InstructionFusion::IsEmbeddedComputation(computation)) {
      continue;
    }
    absl::c_copy_if(computation->instructions(), std::back_inserter(loops),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  return loops;
}

}  // namespace

absl::StatusOr<bool> EmbeddedWhileLoopUnroller::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Unrolling runs DCE on the module and may expose loops nested in the
  // unrolled body, so re-collect the loops after every successful unroll.
  bool changed = false;
  bool unrolled = true;
  while (unrolled) {
    unrolled = false;
    for (HloInstruction* loop : EmbeddedWhileLoops(module, execution_threads)) {
      // No `prepare`: PrepareModuleForUnrolling would also rewrite loops in
      // the entry computation.
      ABSL_ASSIGN_OR_RETURN(UnrollResult result,
                       WhileLoopUnroller::UnrollAndReturnReplacement(
                           loop, /*unroll_factor=*/-1,
                           /*wrap_in_trivial_loop=*/false,
                           /*force_unroll=*/false, /*prepare=*/false));
      if (result.unrolled) {
        changed = unrolled = true;
        break;
      }
      VLOG(2) << "Could not unroll " << loop->name()
              << " in embedded computation " << loop->parent()->name();
    }
  }
  return changed;
}

}  // namespace xla::cpu

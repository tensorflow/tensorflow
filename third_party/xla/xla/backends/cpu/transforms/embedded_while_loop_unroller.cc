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
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/while_loop_unroller.h"
#include "xla/service/call_graph.h"

namespace xla::cpu {
namespace {

std::vector<HloInstruction*> EmbeddedWhileLoops(const CallGraph& call_graph) {
  std::vector<HloInstruction*> loops;
  CHECK_OK(call_graph.VisitNodes([&](const CallGraphNode& node) {
    if (node.IsEmbedded()) {
      absl::c_copy_if(node.computation()->instructions(),
                      std::back_inserter(loops),
                      HloPredicateIsOp<HloOpcode::kWhile>);
    }
    return absl::OkStatus();
  }));
  return loops;
}

}  // namespace

absl::StatusOr<bool> EmbeddedWhileLoopUnroller::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Unrolling runs DCE on the module and may clone called computations or
  // expose loops nested in the unrolled body, so rebuild the call graph and
  // re-collect the loops after every successful unroll.
  bool changed = false;
  bool unrolled = true;
  while (unrolled) {
    unrolled = false;
    std::unique_ptr<CallGraph> call_graph =
        CallGraph::Build(module, execution_threads);
    for (HloInstruction* loop : EmbeddedWhileLoops(*call_graph)) {
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

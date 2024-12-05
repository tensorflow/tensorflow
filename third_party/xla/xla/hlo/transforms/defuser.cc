/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/defuser.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla {

absl::StatusOr<bool> Defuser::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Defusing module " << module->name();
  XLA_VLOG_LINES(2, "Before defusion:\n" + module->ToString());

  bool changed = false;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(
      [&](const CallGraphNode& call_graph_node) -> absl::Status {
        if (call_graph_node.computation()->IsFusionComputation()) {
          TF_RET_CHECK(call_graph_node.caller_callsites().size() == 1);
          HloInstruction* fusion_instruction =
              call_graph_node.caller_callsites()[0].instruction();
          TF_RETURN_IF_ERROR(fusion_instruction->Defuse());
          changed = true;
        }
        return absl::OkStatus();
      },
      /*visit_unreachable_nodes=*/true));

  XLA_VLOG_LINES(2, "After defusion:\n" + module->ToString());

  return changed;
}

}  // namespace xla

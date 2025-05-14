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

#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/call_graph.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace {

// Flatten a single call graph node. Expects to visit nodes in postorder.
absl::Status FlattenNode(const CallGraphNode& node) {
  HloComputation* computation = node.computation();
  HloModule* module = computation->parent();
  // Clone callee for all call-sites except the first one.
  for (int i = 0; i < node.caller_callsites().size(); ++i) {
    CallSite call_site = node.caller_callsites()[i];
    // Only consider sequential call contexts.
    if (call_site.context() == CallContext::kEmbedded) {
      continue;
    }
    CHECK_EQ(call_site.context(), CallContext::kControlFlow);

    // Skip first element if this computation is only called from a sequential
    // context.
    if (node.context() != CallContext::kBoth && i == 0) {
      continue;
    }

    std::vector<HloComputation*> worklist;
    // Clone computation for the remaining sequential context call sites.
    call_site.instruction()->ReplaceCalledComputations(
        [&](HloComputation* callee) {
          if (callee == computation) {
            HloComputation* clone =
                module->AddEmbeddedComputation(callee->Clone());
            worklist.push_back(clone);
            return clone;
          }
          return callee;
        });

    // Clone the sub-tree of all computations called from this node.
    while (!worklist.empty()) {
      auto current = worklist.back();
      worklist.pop_back();
      for (auto* instruction : current->instructions()) {
        if (GetInstructionCallContext(instruction->opcode()) !=
            CallContext::kControlFlow) {
          continue;
        }

        instruction->ReplaceCalledComputations([&](HloComputation* callee) {
          return module->AddEmbeddedComputation(callee->Clone());
        });
        for (auto* callee_clone : instruction->called_computations()) {
          worklist.push_back(callee_clone);
        }
      }
    }
  }
  return absl::OkStatus();
}

// Annotates flatten computations with callee instruction types.
absl::Status AnnotateNode(const CallGraphNode& node) {
  for (auto& callsite : node.callsites()) {
    HloInstruction* instruction = callsite.instruction();

    if (instruction->opcode() == HloOpcode::kFusion) {
      for (HloComputation* computation : instruction->called_computations()) {
        computation->SetFusionInstruction(instruction);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> FlattenCallGraph::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3, "Before flatten call graph:\n" + module->ToString());

  {  // Flatten original call graph.
    std::unique_ptr<CallGraph> call_graph =
        CallGraph::Build(module, execution_threads);
    TF_RETURN_IF_ERROR(call_graph->VisitNodes(FlattenNode));
  }

  {  // Annotate flattened computations with callee types.
    std::unique_ptr<CallGraph> call_graph =
        CallGraph::Build(module, execution_threads);
    TF_RETURN_IF_ERROR(call_graph->VisitNodes(AnnotateNode));
  }

  XLA_VLOG_LINES(3, "After flatten call graph:\n" + module->ToString());
  return true;
}

}  // namespace xla

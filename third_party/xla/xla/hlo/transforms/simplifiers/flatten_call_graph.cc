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
  for (int i = 0; i < node.caller_callsites().size(); ++i) {
    CallSite call_site = node.caller_callsites()[i];
    std::vector<HloComputation*> worklist;
    // If this is the first (or only) callsite, and it only refers to the
    // computation once, no need to clone.
    if (i == 0) {
      int computation_count = 0;
      for (auto* callee : call_site.instruction()->called_computations()) {
        if (callee == computation) {
          ++computation_count;
        }
      }
      if (computation_count <= 1) {
        continue;
      }
    }
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

  // Correctly handle dead code: if a fusion computation is no longer used, it
  // should not have a fusion instruction set.
  if (node.callers().empty() &&
      node.computation()->FusionInstruction() != nullptr) {
    node.computation()->SetFusionInstruction(nullptr);
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

  // TODO(b/418034360): Remove this step once the fusion instruction is
  // automatically maintained.
  {  // Annotate flattened computations with callee types.
    std::unique_ptr<CallGraph> call_graph =
        CallGraph::Build(module, execution_threads);
    TF_RETURN_IF_ERROR(call_graph->VisitNodes(AnnotateNode));
  }

  XLA_VLOG_LINES(3, "After flatten call graph:\n" + module->ToString());
  return true;
}

}  // namespace xla

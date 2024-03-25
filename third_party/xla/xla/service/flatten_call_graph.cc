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

#include "xla/service/flatten_call_graph.h"

#include <memory>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/call_graph.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace {

// Helper to replace the called computation at a while, call, conditional or
// async instruction. This function replaces exactly one instance of
// 'computation' with 'new_computation' even if 'instruction' calls
// 'computation' more than once.
void ReplaceCalledComputation(HloInstruction* instruction,
                              HloComputation* computation,
                              HloComputation* new_computation) {
  switch (instruction->opcode()) {
    case HloOpcode::kWhile: {
      if (computation == instruction->while_condition()) {
        instruction->set_while_condition(new_computation);
      } else {
        CHECK_EQ(computation, instruction->while_body());
        instruction->set_while_body(new_computation);
      }
      break;
    }
    case HloOpcode::kCall: {
      CHECK_EQ(instruction->to_apply(), computation);
      instruction->set_to_apply(new_computation);
      break;
    }
    case HloOpcode::kConditional: {
      for (int b = 0; b < instruction->branch_count(); ++b) {
        if (b == instruction->branch_count() - 1) {
          CHECK_EQ(computation, instruction->branch_computation(b));
        }
        if (computation == instruction->branch_computation(b)) {
          instruction->set_branch_computation(b, new_computation);
          break;
        }
      }
      break;
    }
    default:
      LOG(FATAL) << "unexpected opcode: " << instruction->opcode();
  }
}

// Flatten a single call graph node. Expects to visit nodes in postorder.
Status FlattenNode(const CallGraphNode& node) {
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

    if (computation->IsAsyncComputation()) {
      continue;
    }
    // Clone computation for the remaining sequential context call sites.
    HloComputation* clone =
        module->AddEmbeddedComputation(computation->Clone());
    ReplaceCalledComputation(call_site.instruction(), computation, clone);
    // Clone the sub-tree of all computations called from this node.
    std::vector<HloComputation*> worklist;
    worklist.push_back(clone);
    while (!worklist.empty()) {
      auto current = worklist.back();
      worklist.pop_back();
      for (auto* instruction : current->instructions()) {
        if (GetInstructionCallContext(instruction->opcode()) !=
            CallContext::kControlFlow) {
          continue;
        }
        for (auto callee : instruction->called_computations()) {
          HloComputation* callee_clone =
              module->AddEmbeddedComputation(callee->Clone());
          ReplaceCalledComputation(instruction, callee, callee_clone);
          worklist.push_back(callee_clone);
        }
      }
    }
  }
  return OkStatus();
}

}  // namespace

absl::StatusOr<bool> FlattenCallGraph::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3, "Before flatten call graph:\n" + module->ToString());

  std::unique_ptr<CallGraph> call_graph =
      CallGraph::Build(module, execution_threads);
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(FlattenNode));

  XLA_VLOG_LINES(3, "After flatten call graph:\n" + module->ToString());
  return true;
}

}  // namespace xla

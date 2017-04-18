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

#include "tensorflow/compiler/xla/service/flatten_call_graph.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

// Helper to replace the called computation at a while- or call-instruction.
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
    default:
      LOG(FATAL) << "unexpected opcode: "
                 << HloOpcodeString(instruction->opcode());
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
    if (call_site.context() == CallContext::kParallel) {
      continue;
    }
    CHECK_EQ(call_site.context(), CallContext::kSequential);

    // Skip first element if this computation is only called from a sequential
    // context.
    if (node.context() != CallContext::kBoth && i == 0) {
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
      for (auto& instruction : current->instructions()) {
        if (GetInstructionCallContext(instruction.get()) !=
            CallContext::kSequential) {
          continue;
        }
        for (auto callee : instruction->called_computations()) {
          HloComputation* callee_clone =
              module->AddEmbeddedComputation(callee->Clone());
          ReplaceCalledComputation(instruction.get(), callee, callee_clone);
          worklist.push_back(callee_clone);
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace

StatusOr<bool> FlattenCallGraph::Run(HloModule* module) {
  XLA_VLOG_LINES(3, "Before flatten call graph:\n" + module->ToString());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<CallGraph> call_graph,
                      CallGraph::Build(module));
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(FlattenNode));

  XLA_VLOG_LINES(3, "After flatten call graph:\n" + module->ToString());
  return true;
}

}  // namespace xla

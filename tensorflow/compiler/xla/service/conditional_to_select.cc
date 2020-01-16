/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/conditional_to_select.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

static StatusOr<bool> DoConditionalToSelect(HloInstruction* conditional) {
  // Only allow conditional to select if the called computations
  // do not have side effects.
  if (conditional->true_computation()->HasSideEffect() ||
      conditional->false_computation()->HasSideEffect()) {
    VLOG(1) << "Not transforming conditional; branches have side effects:"
            << conditional->ToString();
    return false;
  }

  auto computation = conditional->parent();

  // Create new instructions
  HloInstruction* if_call_op =
      computation->AddInstruction(HloInstruction::CreateCall(
          conditional->shape(), {conditional->mutable_operand(1)},
          conditional->true_computation()));
  conditional->SetupDerivedInstruction(if_call_op);
  HloInstruction* else_call_op =
      computation->AddInstruction(HloInstruction::CreateCall(
          conditional->shape(), {conditional->mutable_operand(2)},
          conditional->false_computation()));
  conditional->SetupDerivedInstruction(else_call_op);
  HloInstruction* condition = conditional->mutable_operand(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * select_op,
      MakeSelectHlo(condition, if_call_op, else_call_op, conditional));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(conditional, select_op));
  TF_RETURN_IF_ERROR(CallInliner::Inline(if_call_op).status());
  TF_RETURN_IF_ERROR(CallInliner::Inline(else_call_op).status());
  return true;
}

StatusOr<bool> ConditionalToSelect::Run(HloModule* module) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  bool did_mutate = false;
  VLOG(1) << "Running conditional-to-select pass";
  TF_RETURN_IF_ERROR(
      call_graph->VisitNodes([&](const CallGraphNode& node) -> Status {
        std::vector<HloInstruction*> ToInline;
        if (node.context() != CallContext::kParallel) {
          return Status::OK();
        }
        for (const CallSite& callsite : node.callsites()) {
          if (callsite.instruction()->opcode() == HloOpcode::kConditional) {
            VLOG(1) << "Visiting conditional: " << callsite.ToString();
            HloInstruction* conditional = callsite.instruction();
            TF_ASSIGN_OR_RETURN(bool result,
                                DoConditionalToSelect(conditional));
            did_mutate |= result;
          }
        }
        return Status::OK();
      }));
  return did_mutate;
}

}  // namespace xla

/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/conditional_to_select.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla {

static absl::StatusOr<bool> DoConditionalToSelect(HloInstruction* conditional) {
  if (conditional->shape().IsTuple()) {
    VLOG(1) << "Not transforming tuples to 'select'";
    return false;
  }
  bool is_form2 = false;
  HloInstruction* true_operand;
  HloInstruction* false_operand;
  HloComputation* true_computation;
  HloComputation* false_computation;

  if (ShapeUtil::IsScalarWithElementType(conditional->operand(0)->shape(),
                                         PrimitiveType::PRED)) {
    // Form 1
    // Conditional(pred, true_oprnd, true_comp, false_oprnd, false_comp)
    true_operand = conditional->mutable_operand(1);
    false_operand = conditional->mutable_operand(2);
    true_computation = conditional->true_computation();
    false_computation = conditional->false_computation();
  } else if (ShapeUtil::IsScalarWithElementType(
                 conditional->operand(0)->shape(), PrimitiveType::S32)) {
    // Form 2
    // Conditional(branch_index, branch_computations, branch_operands)
    if (conditional->branch_computations().size() != 2) {
      VLOG(1)
          << "Not transforming conditional; branch_computations size is not 2";
      return false;
    }
    false_operand = conditional->mutable_operand(1);
    true_operand = conditional->mutable_operand(2);
    false_computation = conditional->branch_computations()[0];
    true_computation = conditional->branch_computations()[1];
    is_form2 = true;
  } else {
    VLOG(1) << "Not transforming conditional; Unexpected operand0 type";
    return false;
  }

  // Only allow conditional to select if the called computations
  // do not have side effects.
  if (true_computation->HasSideEffect() || false_computation->HasSideEffect()) {
    VLOG(1) << "Not transforming conditional; branches have side effects:"
            << conditional->ToString();
    return false;
  }

  auto computation = conditional->parent();

  // Create new instructions
  HloInstruction* if_call_op =
      computation->AddInstruction(HloInstruction::CreateCall(
          conditional->shape(), {true_operand}, true_computation));
  conditional->SetupDerivedInstruction(if_call_op);
  HloInstruction* else_call_op =
      computation->AddInstruction(HloInstruction::CreateCall(
          conditional->shape(), {false_operand}, false_computation));
  conditional->SetupDerivedInstruction(else_call_op);
  HloInstruction* condition = conditional->mutable_operand(0);
  if (is_form2) {
    // If Form2 => convert operand0 of type S32 to PRED
    condition = computation->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::ChangeElementType(condition->shape(), PrimitiveType::PRED),
        condition));
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * select_op,
      MakeSelectHlo(condition, if_call_op, else_call_op, conditional));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(conditional, select_op));
  TF_RETURN_IF_ERROR(CallInliner::Inline(if_call_op).status());
  TF_RETURN_IF_ERROR(CallInliner::Inline(else_call_op).status());
  return true;
}

absl::StatusOr<bool> ConditionalToSelect::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  bool did_mutate = false;
  VLOG(1) << "Running conditional-to-select pass";
  TF_RETURN_IF_ERROR(
      call_graph->VisitNodes([&](const CallGraphNode& node) -> absl::Status {
        std::vector<HloInstruction*> ToInline;
        if (node.context() != CallContext::kEmbedded) {
          return absl::OkStatus();
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
        return absl::OkStatus();
      }));
  return did_mutate;
}

}  // namespace xla
